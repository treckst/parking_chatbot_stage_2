import json
from pathlib import Path
import sqlite3
import time

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.functions import redact_pii_rows
from src.guardrails import SQLValidationError, safe_sql
from src.models_vector import llm
from src.prompts import SQL_AGENT_PROMPT, SQL_AVAILABILITY_PROMPT
from src.schema import (
    SQLAgentInput,
    SQLAgentOutput,
    SQLAgentState,
    SQLAvailabilityResult,
)

DB_URI = (
    (Path(__file__).parent.parent / "data" / "parking.db")
    .resolve()
    .as_uri()
    + "?mode=ro"
)
MAX_SQL_CALLS = 3


def authorize_sql(action, arg1, arg2, database, source):
    if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION}:
        return sqlite3.SQLITE_OK

    if action == sqlite3.SQLITE_READ:
        if database == "main" and arg1 == "parking_slots":
            return sqlite3.SQLITE_OK

    return sqlite3.SQLITE_DENY


@tool
def execute_readonly_sql(sql: str) -> dict:
    """Execute one read-only SQLite query against the public parking table.

    Returns either rows or a database error that can help correct the SQL.
    """
    connection = None

    try:
        sql = safe_sql(sql)
        connection = sqlite3.connect(DB_URI, uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        connection.set_authorizer(authorize_sql)
        deadline = time.monotonic() + 2
        connection.set_progress_handler(
            lambda: time.monotonic() > deadline, 1000,
        )
        rows = connection.execute(sql).fetchmany(51)

        return {
            "ok": True,
            "rows": redact_pii_rows([dict(row) for row in rows[:50]]),
            "truncated": len(rows) > 50,
        }

    except SQLValidationError as exc:
        return {
            "ok": False,
            "error_category": "validation_error",
            "error": f"Query rejected: {exc}",
            "retryable": True,
        }

    except sqlite3.Error as exc:
        message = str(exc)

        if "no such column" in message:
            category = "unknown_column"
        elif "no such table" in message:
            category = "unknown_table"
        elif "syntax error" in message:
            category = "syntax_error"
        else:
            # Avoid exposing arbitrary operational error details.
            return {
                "ok": False,
                "error_category": "execution_error",
                "error": "Query was denied, timed out, or could not execute.",
                "retryable": False,
            }
        return {
            "ok": False,
            "error_category": category,
            "error": message,
            "retryable": True,
        }

    finally:
        if connection is not None:
            connection.close()


model_with_tools = llm.bind_tools(
    [execute_readonly_sql],
    parallel_tool_calls=False,
)
model_with_required_tool = llm.bind_tools(
    [execute_readonly_sql],
    parallel_tool_calls=False,
    tool_choice="execute_readonly_sql",
)
tool_node = ToolNode([execute_readonly_sql], messages_key="sql_messages")
availability_llm = llm.with_structured_output(SQLAvailabilityResult)


def agent(state: SQLAgentState):
    reviewing_reservation = state.wants_reservation and not state.missing_info

    if reviewing_reservation:
        instructions = SQL_AGENT_PROMPT + """
Review the supplied venue and booking period. Check that a slot is available
for the entire period (at least the intented time should overlap within the hours range).
A reservation still requires approval.
If no available slot matches the requested venue or period (including if the
venue name returns no rows), query what parking venues and slots ARE available
on that parking_date (including location, opening hours, price per hour, and
number of available slots) and report them in your answer.
If an information question is supplied, answer it too. Reuse query results
when they cover both requests. Keep the venue and dates of each request distinct.
"""
        request = state.user_info.model_dump(include={"place", "reservation_period"})
        if state.wants_dynamic_info:
            request["information_question"] = state.dynamic_question
        question = json.dumps(request, ensure_ascii=False)
    else:
        instructions = SQL_AGENT_PROMPT + """
Answer the public parking question about hours, price, or availability and also research it in db.
If the requested venue or period has no available slots on a given date, also
query and report what parking venues, hours, and prices ARE available on that day.
If a venue or date is needed, ask for it. Do not ask for a name or plate,
start a reservation, or suggest that a reservation has been made.
"""
        question = state.dynamic_question

    if state.sql_calls >= MAX_SQL_CALLS:
        instructions += """
No tool calls remain. Answer using successful results already obtained.
If no adequate result exists, explain that the lookup could not be completed.
"""
        model = llm
    elif reviewing_reservation and state.sql_calls == 0:
        model = model_with_required_tool
    else:
        model = model_with_tools

    response = model.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=question),
        *state.sql_messages,
    ])

    if response.tool_calls:
        return {"sql_messages": [response]}

    if reviewing_reservation:
        check = availability_llm.invoke([
            SystemMessage(content=SQL_AVAILABILITY_PROMPT),
            HumanMessage(content=question),
            *state.sql_messages,
            response,
        ])
        return {
            "sql_messages": [response],
            "sql_result": check.summary,
            "slot_available": check.slot_available,
            "assigned_slot": check.assigned_slot if check.slot_available else None,
        }

    return {
        "sql_messages": [response],
        "sql_result": response.text,
        "slot_available": False,
        "assigned_slot": None,
    }


def tools(state: SQLAgentState):
    calls = state.sql_messages[-1].tool_calls
    remaining = MAX_SQL_CALLS - state.sql_calls

    if len(calls) > remaining:
        return {
            "sql_messages": [
                ToolMessage(
                    content=json.dumps({
                        "ok": False,
                        "error": "SQL call budget exceeded. No query executed.",
                        "retryable": False,
                    }),
                    tool_call_id=call["id"],
                )
                for call in calls
            ],
            "sql_calls": MAX_SQL_CALLS,
        }

    result = tool_node.invoke(state)
    return {
        "sql_messages": result["sql_messages"],
        "sql_calls": state.sql_calls + len(calls),
    }


def route_after_agent(state: SQLAgentState):
    if state.sql_messages[-1].tool_calls:
        return "tools"
    return END


builder = StateGraph(
    SQLAgentState,
    input_schema=SQLAgentInput,
    output_schema=SQLAgentOutput,
)
builder.add_node("agent", agent)
builder.add_node("tools", tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
builder.add_edge("tools", "agent")

sql_agent = builder.compile()
