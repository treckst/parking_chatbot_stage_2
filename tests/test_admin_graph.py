from datetime import datetime, timezone
from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src import admin_graph
from src.schema import AdminDecision, ReservationPeriod, UserState


def _sample_input():
    return {
        "user_info": UserState(
            name="Anna Nowak",
            car_number="EL99887",
            place="Manufaktura",
            reservation_period=ReservationPeriod(
                start_date="2026-11-18T09:00",
                end_date="2026-11-18T12:00",
            ),
        ),
        "assigned_slot": "B01",
        "sql_result": "Slot B01 at Manufaktura is available on 2026-11-18 (06:00-22:00).",
    }


def _fake_httpx_post(sent_payloads: list[dict], request_id: str):
    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"request_id": request_id, "status": "pending"}

    def fake_post(url, json, timeout):
        sent_payloads.append({"url": url, "json": json})
        return DummyResponse()

    return fake_post


def test_admin_subgraph_interrupt_and_approve(monkeypatch):
    sent_payloads: list[dict] = []

    class FakeDraftLLM:
        def invoke(self, _messages):
            return SimpleNamespace(text="Please approve Anna Nowak for Manufaktura slot B01.")

    class FakeDecisionLLM:
        def invoke(self, _messages):
            return AdminDecision(
                status="approved",
                reason="Confirmed for B01",
                approval_time=datetime(2026, 11, 18, 8, 30, tzinfo=timezone.utc),
            )

    monkeypatch.setattr(admin_graph, "_get_llm", lambda: FakeDraftLLM())
    monkeypatch.setattr(admin_graph, "_get_decision_llm", lambda: FakeDecisionLLM())
    monkeypatch.setattr(
        admin_graph.httpx,
        "post",
        _fake_httpx_post(sent_payloads, "REQ-TEST0001"),
    )

    compiled = admin_graph.admin_builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t-approve-1"}}

    # 1st invoke drafts + sends request to FastAPI and pauses on interrupt()
    paused = compiled.invoke(_sample_input(), config=config)
    assert "__interrupt__" in paused
    interrupt_payload = paused["__interrupt__"][0].value
    assert interrupt_payload["request_id"] == "REQ-TEST0001"
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["url"].endswith("/requests")
    assert sent_payloads[0]["json"]["thread_id"] == "t-approve-1"

    # Resuming processes the decision WITHOUT re-sending the POST /requests call
    resumed = compiled.invoke(
        Command(resume={"decision": "approved", "comment": "Confirmed for B01"}),
        config=config,
    )
    assert len(sent_payloads) == 1
    assert resumed["admin_decision"].status == "approved"
    assert resumed["admin_decision"].reason == "Confirmed for B01"
    assert resumed["user_info"].approval_time is not None


def test_admin_subgraph_refused_decision(monkeypatch):
    sent_payloads: list[dict] = []

    class FakeDraftLLM:
        def invoke(self, _messages):
            return SimpleNamespace(text="Please review Anna Nowak for slot B01.")

    class FakeDecisionLLM:
        def invoke(self, _messages):
            return AdminDecision(
                status="refused",
                reason="VIP event on 2026-11-18",
                approval_time=None,
            )

    monkeypatch.setattr(admin_graph, "_get_llm", lambda: FakeDraftLLM())
    monkeypatch.setattr(admin_graph, "_get_decision_llm", lambda: FakeDecisionLLM())
    monkeypatch.setattr(
        admin_graph.httpx,
        "post",
        _fake_httpx_post(sent_payloads, "REQ-TEST0002"),
    )

    compiled = admin_graph.admin_builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t-refuse-1"}}

    paused = compiled.invoke(_sample_input(), config=config)
    assert "__interrupt__" in paused

    resumed = compiled.invoke(
        Command(resume="Sorry, refused due to VIP event on 2026-11-18"),
        config=config,
    )
    assert len(sent_payloads) == 1
    assert resumed["admin_decision"].status == "refused"
    assert resumed["admin_decision"].reason == "VIP event on 2026-11-18"
    assert resumed["user_info"].approval_time is None
