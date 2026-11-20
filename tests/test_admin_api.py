from fastapi.testclient import TestClient
import pytest

from src import admin_api

client = TestClient(admin_api.app)

SAMPLE_REQUEST = {
    "user_name": "Jan Kowalski",
    "car_number": "EL12345",
    "place": "Galeria Łódzka",
    "slot_code": "A01",
    "start_date": "2026-11-15T10:00",
    "end_date": "2026-11-15T14:00",
    "summary": "Reservation request for Jan Kowalski at Galeria Łódzka (slot A01).",
}


@pytest.fixture(autouse=True)
def clean_store():
    admin_api.reset_store()
    yield
    admin_api.reset_store()


def test_create_list_and_approve_request():
    created = client.post("/requests", json=SAMPLE_REQUEST)
    assert created.status_code == 201
    req = created.json()
    assert req["request_id"].startswith("REQ-")
    assert req["status"] == "pending"

    pending = client.get("/requests", params={"status": "pending"})
    assert pending.status_code == 200
    assert len(pending.json()) == 1

    decision = client.post(
        f"/requests/{req['request_id']}/decision",
        json={"decision": "approved", "comment": "Bay A01 confirmed"},
    )
    assert decision.status_code == 200
    body = decision.json()
    assert body["request"]["status"] == "approved"
    assert body["request"]["comment"] == "Bay A01 confirmed"
    assert body["request"]["approval_time"] is not None
    assert body["graph_resumed"] is False


def test_refuse_and_reject_duplicate_decision():
    req_id = client.post("/requests", json=SAMPLE_REQUEST).json()["request_id"]

    first = client.post(
        f"/requests/{req_id}/decision",
        json={"decision": "refused", "comment": "Maintenance on row A"},
    )
    assert first.status_code == 200
    assert first.json()["request"]["status"] == "refused"
    assert first.json()["request"]["approval_time"] is None

    second = client.post(
        f"/requests/{req_id}/decision",
        json={"decision": "approved"},
    )
    assert second.status_code == 409


def test_unknown_request_returns_404():
    assert client.get("/requests/REQ-DOESNOTEXIST").status_code == 404
    assert (
        client.post(
            "/requests/REQ-DOESNOTEXIST/decision",
            json={"decision": "approved"},
        ).status_code
        == 404
    )


def test_decision_auto_resumes_langgraph_thread(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return DummyResponse()

    monkeypatch.setattr(admin_api.httpx, "post", fake_post)

    payload = {**SAMPLE_REQUEST, "thread_id": "thread-42"}
    req_id = client.post("/requests", json=payload).json()["request_id"]

    res = client.post(
        f"/requests/{req_id}/decision",
        json={"decision": "approved", "comment": "OK"},
    )
    assert res.status_code == 200
    assert res.json()["graph_resumed"] is True
    assert captured["url"].endswith("/threads/thread-42/runs")
    assert captured["json"]["command"]["resume"]["decision"] == "approved"
