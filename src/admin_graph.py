"""Stage 2: Human-in-the-Loop administrator approval subgraph.

Uses a second LangChain agent + tools to send reservation requests to the
FastAPI Admin API, pauses execution cleanly via LangGraph `interrupt()`,
and interprets the administrator's confirm/refuse reply into AdminDecision.
"""

from datetime import datetime, timezone
from functools import lru_cache
import json
import os

import httpx
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.prompts import ADMIN_DECISION_PROMPT, ADMIN_REQUEST_PROMPT
from src.schema import (
    AdminAgentInput,
    AdminAgentOutput,
    AdminAgentState,
    AdminDecision,
)


def _admin_api_url() -> str:
    return os.getenv("ADMIN_API_URL", "http://127.0.0.1:8000").rstrip("/")


@tool
def send_admin_request(
    user_name: str,
    car_number: str,
    place: str,
    start_date: str,
    end_date: str,
    summary: str,
    slot_code: str | None = None,
    thread_id: str | None = None,
) -> dict:
    """Send a pending parking reservation request to the administrator REST API."""
    payload = {
        "user_name": user_name,
        "car_number": car_number,
        "place": place,
        "slot_code": slot_code,
        "start_date": start_date,
        "end_date": end_date,
        "summary": summary,
        "thread_id": thread_id,
    }
    try:
        response = httpx.post(
            f"{_admin_api_url()}/requests",
            json=payload,
            timeout=5.0,
        )
        response.raise_for_status()
        return {"ok": True, **response.json()}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc)}


@tool
def fetch_admin_decision(request_id: str) -> dict:
    """Fetch the current administrator decision for a request from the REST API."""
    try:
        response = httpx.get(
            f"{_admin_api_url()}/requests/{request_id}",
            timeout=5.0,
        )
        response.raise_for_status()
        return {"ok": True, **response.json()}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc)}


@lru_cache(maxsize=1)
def _get_llm():
    from src.models_vector import llm

    return llm


@lru_cache(maxsize=1)
def _get_decision_llm():
    return _get_llm().with_structured_output(AdminDecision)


def send_request(state: AdminAgentState, config: RunnableConfig):
    """Draft the confirmation request via LangChain and send it to the Admin API."""
    details = {
        "user_name": state.user_info.name,
        "car_number": state.user_info.car_number,
        "place": state.user_info.place,
        "slot_code": state.assigned_slot,
        "start_date": state.user_info.reservation_period.start_date,
        "end_date": state.user_info.reservation_period.end_date,
        "sql_availability_note": state.sql_result,
    }
    draft = _get_llm().invoke([
        SystemMessage(content=ADMIN_REQUEST_PROMPT),
        HumanMessage(content=json.dumps(details, ensure_ascii=False)),
    ])
    summary_text = draft.text if hasattr(draft, "text") else str(draft.content)

    thread_id = (config.get("configurable") or {}).get("thread_id")
    api_result = send_admin_request.invoke({
        "user_name": state.user_info.name or "",
        "car_number": state.user_info.car_number or "",
        "place": state.user_info.place or "",
        "slot_code": state.assigned_slot,
        "start_date": state.user_info.reservation_period.start_date or "",
        "end_date": state.user_info.reservation_period.end_date or "",
        "summary": summary_text,
        "thread_id": thread_id,
    })

    return {
        "request_id": api_result.get("request_id"),
        "admin_request_message": summary_text,
    }


def wait_for_decision(state: AdminAgentState):
    """Pause graph execution until the human administrator approves or refuses.

    Kept in its own side-effect-free node so resuming never re-sends the request.
    """
    api_url = _admin_api_url()
    payload = {
        "type": "admin_approval_required",
        "request_id": state.request_id,
        "message": state.admin_request_message,
        "approve_endpoint": (
            f"{api_url}/requests/{state.request_id}/decision"
            if state.request_id
            else None
        ),
        "example_payload": {"decision": "approved", "comment": "Approved"},
    }
    admin_reply = interrupt(payload)
    return {"raw_admin_response": admin_reply}


def process_decision(state: AdminAgentState):
    """Normalize the administrator's reply into a structured AdminDecision."""
    raw = state.raw_admin_response

    if (
        isinstance(raw, str)
        and raw.strip().lower() in {"", "fetch", "check"}
        and state.request_id
    ):
        raw = fetch_admin_decision.invoke({"request_id": state.request_id})

    raw_text = (
        json.dumps(raw, ensure_ascii=False, default=str)
        if not isinstance(raw, str)
        else raw
    )
    decision: AdminDecision = _get_decision_llm().invoke([
        SystemMessage(content=ADMIN_DECISION_PROMPT),
        HumanMessage(content=f"Administrator response:\n{raw_text}"),
    ])

    if decision.status == "approved" and decision.approval_time is None:
        decision = decision.model_copy(
            update={"approval_time": datetime.now(timezone.utc)}
        )

    updated_user = state.user_info.model_copy(
        update={
            "approval_time": (
                decision.approval_time if decision.status == "approved" else None
            )
        }
    )
    return {
        "admin_decision": decision,
        "user_info": updated_user,
    }


admin_builder = StateGraph(
    AdminAgentState,
    input_schema=AdminAgentInput,
    output_schema=AdminAgentOutput,
)
admin_builder.add_node("send_request", send_request)
admin_builder.add_node("wait_for_decision", wait_for_decision)
admin_builder.add_node("process_decision", process_decision)

admin_builder.add_edge(START, "send_request")
admin_builder.add_edge("send_request", "wait_for_decision")
admin_builder.add_edge("wait_for_decision", "process_decision")
admin_builder.add_edge("process_decision", END)

admin_agent = admin_builder.compile()
