"""FastAPI REST API for human administrator reservation approvals.

Run locally:
    uv run uvicorn src.admin_api:app --port 8000

Admin can view pending requests and submit approve/refuse decisions directly
via FastAPI's built-in Swagger docs at http://127.0.0.1:8000/docs or via curl.
"""

from datetime import datetime, timezone
import os
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
import httpx
from pydantic import BaseModel, Field

app = FastAPI(
    title="Parking Reservation Admin API",
    description="REST API for reviewing and approving/refusing parking reservation requests.",
    version="1.0.0",
)


class AdminRequestCreate(BaseModel):
    user_name: str
    car_number: str
    place: str
    slot_code: str | None = None
    start_date: str
    end_date: str
    summary: str = Field(description="Request summary drafted by the admin-liaison agent.")
    thread_id: str | None = Field(
        default=None,
        description="Optional LangGraph thread ID to auto-resume after a decision.",
    )


class AdminDecisionInput(BaseModel):
    decision: Literal["approved", "refused"]
    comment: str = ""


class AdminRequestRecord(AdminRequestCreate):
    request_id: str
    status: Literal["pending", "approved", "refused"] = "pending"
    comment: str | None = None
    created_at: str
    approval_time: str | None = None


class DecisionResponse(BaseModel):
    request: AdminRequestRecord
    graph_resumed: bool
    resume_error: str | None = None


# In-memory store for Stage 2.
REQUESTS: dict[str, AdminRequestRecord] = {}


def reset_store() -> None:
    """Clear stored requests (useful in tests)."""
    REQUESTS.clear()


def _resume_langgraph_thread(
    thread_id: str,
    decision_payload: dict,
) -> tuple[bool, str | None]:
    """Resume an interrupted LangGraph Server thread if reachable."""
    base_url = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024").rstrip("/")
    assistant_id = os.getenv("LANGGRAPH_ASSISTANT_ID", "graph")
    if not base_url:
        return False, None

    try:
        response = httpx.post(
            f"{base_url}/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "command": {"resume": decision_payload},
            },
            timeout=5.0,
        )
        response.raise_for_status()
        return True, None
    except httpx.HTTPError as exc:
        return False, str(exc)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/requests", response_model=AdminRequestRecord, status_code=201)
def create_request(payload: AdminRequestCreate) -> AdminRequestRecord:
    request_id = f"REQ-{uuid4().hex[:8].upper()}"
    record = AdminRequestRecord(
        request_id=request_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        **payload.model_dump(),
    )
    REQUESTS[request_id] = record
    return record


@app.get("/requests", response_model=list[AdminRequestRecord])
def list_requests(
    status: Literal["pending", "approved", "refused"] | None = Query(default=None),
) -> list[AdminRequestRecord]:
    records = list(REQUESTS.values())
    if status is not None:
        records = [r for r in records if r.status == status]
    return records


@app.get("/requests/{request_id}", response_model=AdminRequestRecord)
def get_request(request_id: str) -> AdminRequestRecord:
    record = REQUESTS.get(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found.")
    return record


@app.post("/requests/{request_id}/decision", response_model=DecisionResponse)
def submit_decision(
    request_id: str,
    payload: AdminDecisionInput,
) -> DecisionResponse:
    record = REQUESTS.get(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found.")
    if record.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Request '{request_id}' was already {record.status}.",
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    record.status = payload.decision
    record.comment = payload.comment or None
    record.approval_time = now_iso if payload.decision == "approved" else None

    graph_resumed = False
    resume_error = None
    if record.thread_id:
        graph_resumed, resume_error = _resume_langgraph_thread(
            record.thread_id,
            {
                "request_id": record.request_id,
                "decision": record.status,
                "comment": record.comment,
                "approval_time": record.approval_time,
            },
        )

    return DecisionResponse(
        request=record,
        graph_resumed=graph_resumed,
        resume_error=resume_error,
    )
