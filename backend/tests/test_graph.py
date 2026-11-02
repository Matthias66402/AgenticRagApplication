"""LangGraph-Steuerung (app.agent.graph) und Runner mit geskriptetem Modell statt echtem LLM."""

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx
import pytest
from pydantic_ai import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agent.deps import AgentDeps
from app.agent.graph import planner_graph
from app.agent.planner import planner_agent
from app.agent.runner import RunOutcome, run_planner
from app.config import get_settings
from app.db import SessionLocal
from tests.test_agent import scripted_model

QUESTION = ("ask_user", {"question": "Wo startest Du?", "missing_fields": ["start"]})


@pytest.fixture
async def deps() -> AsyncIterator[AgentDeps]:
    async with httpx.AsyncClient() as http:
        yield AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Test")


async def collect(message: str, deps: AgentDeps, history: list[Any] | None = None) -> tuple[list[dict[str, Any]], RunOutcome]:
    outcome = RunOutcome()
    events = [event async for event in run_planner(message, history or [], deps, outcome)]
    return events, outcome


async def visited_nodes(deps: AgentDeps) -> list[str]:
    updates = planner_graph.astream({"message": "Hallo", "history_json": []}, context=deps, stream_mode="updates")
    return [node async for update in updates for node in update]


@pytest.mark.parametrize(
    ("step", "node"),
    [
        (QUESTION, "finish_question"),
        (("answer_user", {"message": "Gern geschehen."}), "finish_answer"),
    ],
)
async def test_graph_routes_by_output(deps: AgentDeps, step: tuple[str, dict[str, Any]], node: str) -> None:
    with planner_agent.override(model=scripted_model(step)):
        assert await visited_nodes(deps) == ["planner", node]


async def test_graph_routes_errors(deps: AgentDeps) -> None:
    async def fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("Anbieter nicht erreichbar")
        yield ""  # pragma: no cover

    with planner_agent.override(model=FunctionModel(stream_function=fail)):
        assert await visited_nodes(deps) == ["planner", "finish_error"]


async def test_runner_streams_events_and_fills_outcome(deps: AgentDeps) -> None:
    model = scripted_model(
        ("estimate_hiking_time", {"distance_km": 12, "ascent_m": 300}),
        "Ich frage kurz nach.",  # Text statt Output-Tool → PydanticAI verlangt ein Output-Tool
        QUESTION,
    )
    with planner_agent.override(model=model):
        events, outcome = await collect("Wanderung morgen", deps)

    types = [e["type"] for e in events]
    assert types[:2] == ["tool_call", "tool_result"]
    assert "token" in types and "error" not in types
    assert outcome.text == "Ich frage kurz nach."
    [call] = outcome.tool_calls
    assert call["tool"] == "estimate_hiking_time" and call["args"] == {"distance_km": 12, "ascent_m": 300}
    assert call["id"] == events[0]["id"] and call["failed"] is False
    assert call["result"] == events[1]["result"] != ""
    assert outcome.question is not None and outcome.question.question == "Wo startest Du?"
    assert outcome.plan is None and outcome.answer is None and outcome.error is None
    assert outcome.history_json and outcome.history_json[0]["kind"] == "request"


async def test_runner_continues_history(deps: AgentDeps) -> None:
    with planner_agent.override(model=scripted_model(QUESTION)):
        _, first = await collect("Wanderung morgen", deps)
    with planner_agent.override(model=scripted_model(("answer_user", {"message": "  Alles klar.  "}))):
        _, second = await collect("Ab Garmisch", deps, first.history_json)
    assert second.answer == "Alles klar."
    assert second.history_json is not None and len(second.history_json) > len(first.history_json or [])


async def test_runner_reports_agent_errors(deps: AgentDeps) -> None:
    async def fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("Anbieter nicht erreichbar")
        yield ""  # pragma: no cover

    with planner_agent.override(model=FunctionModel(stream_function=fail)):
        events, outcome = await collect("Hallo", deps)
    assert events == [{"type": "error", "message": "Fehler bei der Planung: RuntimeError: Anbieter nicht erreichbar"}]
    assert outcome.error == events[0]["message"]
    assert outcome.history_json is None


async def test_runner_keeps_partial_outcome_when_client_disconnects(deps: AgentDeps) -> None:
    model = scripted_model("Moment …", QUESTION)
    with planner_agent.override(model=model):
        outcome = RunOutcome()
        async with aclosing(run_planner("Hallo", [], deps, outcome)) as events:
            async for event in events:
                assert event["type"] == "token"
                break
    assert outcome.text.startswith("Moment")
    assert outcome.question is None and outcome.history_json is None and outcome.error is None
