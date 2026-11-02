"""LangGraph-Steuerung (app.agent.graph) und Runner mit geskriptetem Modell statt echtem LLM."""

import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx
import pytest
import respx
from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

from app.agent.deps import AgentDeps
from app.agent.graph import planner_graph
from app.agent.intake import conversation_context, intake_agent, intake_prompt, places_to_geocode
from app.agent.planner import planner_agent
from app.agent.tools import geocode as geocode_module
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


async def visited_nodes(deps: AgentDeps, message: str = "Hallo") -> list[str]:
    updates = planner_graph.astream({"message": message, "history_json": []}, context=deps, stream_mode="updates")
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
        assert await visited_nodes(deps) == ["intake", "planner", node]


async def test_graph_routes_errors(deps: AgentDeps) -> None:
    async def fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("Anbieter nicht erreichbar")
        yield ""  # pragma: no cover

    with planner_agent.override(model=FunctionModel(stream_function=fail)):
        assert await visited_nodes(deps) == ["intake", "planner", "finish_error"]


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


GARMISCH = {"display_name": "Garmisch-Partenkirchen", "lat": "47.49", "lon": "11.09", "category": "place", "type": "town"}


def intake_finds(*places: dict[str, Any]) -> Any:
    return intake_agent.override(model=TestModel(custom_output_args={"places": list(places)}))


@respx.mock
async def test_prefetched_places_are_in_planner_history(deps: AgentDeps) -> None:
    nominatim = respx.get(geocode_module.NOMINATIM_URL).respond(json=[GARMISCH])
    geocode_module._cache.clear()
    seen: dict[str, Any] = {}

    async def planner_step(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        seen["messages"] = messages
        seen["instructions"] = info.instructions
        yield {0: DeltaToolCall(name=QUESTION[0], json_args=json.dumps(QUESTION[1]))}

    with intake_finds({"place_name": "Garmisch-Partenkirchen", "country_codes": "de"}):
        with planner_agent.override(model=FunctionModel(stream_function=planner_step)):
            assert await visited_nodes(deps, "Wanderung ab Garmisch") == [
                "intake", "prefetch_geocode", "planner", "finish_question"
            ]  # fmt: skip
            events, outcome = await collect("Wanderung ab Garmisch", deps)

    assert nominatim.call_count == 1  # zweiter Lauf aus dem Cache
    # Der Planer sieht Nachricht, Vorab-Aufruf und Ergebnis, als hätte er geocode selbst aufgerufen
    kinds = [[p.part_kind for p in m.parts] for m in seen["messages"]]
    assert kinds == [["user-prompt"], ["tool-call"], ["tool-return"]]
    assert seen["messages"][0].parts[0].content == "Wanderung ab Garmisch"
    call, result = seen["messages"][1].parts[0], seen["messages"][2].parts[0]
    assert (call.tool_name, call.args, call.tool_call_id) == ("geocode", {"place_name": "Garmisch-Partenkirchen", "country_codes": "de"}, result.tool_call_id)  # fmt: skip
    assert result.content[0]["lat"] == 47.49
    assert seen["instructions"] and "Outdoor-Tourenplaner" in seen["instructions"]  # Anweisungen trotz fehlendem Prompt

    assert [e["type"] for e in events] == ["tool_call", "tool_result"]
    assert events[0]["tool"] == "geocode" and events[1]["failed"] is False
    assert [c["tool"] for c in outcome.tool_calls] == ["geocode"] and outcome.tool_calls[0]["result"]
    # Gespeicherter Verlauf: Nachricht genau einmal, danach Vorab-Aufruf und Rückfrage
    prompts = [p for m in outcome.history_json or [] for p in m["parts"] if p["part_kind"] == "user-prompt"]
    assert [p["content"] for p in prompts] == ["Wanderung ab Garmisch"]
    assert outcome.question is not None


@respx.mock
async def test_failed_prefetch_is_passed_to_planner(deps: AgentDeps) -> None:
    respx.get(geocode_module.NOMINATIM_URL).respond(status_code=503)
    geocode_module._cache.clear()
    with intake_finds({"place_name": "Nirgendwo-Vorab"}), planner_agent.override(model=scripted_model(QUESTION)):
        events, outcome = await collect("Tour ab Nirgendwo-Vorab", deps)
    assert events[1]["type"] == "tool_result" and events[1]["failed"] is True
    assert outcome.question is not None and outcome.error is None


async def test_intake_failure_falls_back_to_planner(deps: AgentDeps) -> None:
    async def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("Intake nicht erreichbar")

    with intake_agent.override(model=FunctionModel(fail)), planner_agent.override(model=scripted_model(QUESTION)):
        assert await visited_nodes(deps) == ["intake", "planner", "finish_question"]


def test_intake_context_from_history(deps: AgentDeps) -> None:
    history = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Wanderung morgen"}]},
        {"kind": "response", "parts": [
            {"part_kind": "tool-call", "tool_name": "geocode", "args": {"place_name": "Garmisch-Partenkirchen"}},
            {"part_kind": "tool-call", "tool_name": "ask_user", "args": {"question": "Wohin soll es gehen?"}},
        ]},
    ]  # fmt: skip
    earlier, question, geocoded = conversation_context(history)
    assert (earlier, question, geocoded) == (["Wanderung morgen"], "Wohin soll es gehen?", {"garmisch-partenkirchen"})
    deps.home = "Nelkenstraße 10, 26121 Oldenburg"
    prompt, _ = intake_prompt("Nach Mittenwald", history, deps)
    assert "Wohnadresse: Nelkenstraße 10" in prompt and "Letzte Rückfrage des Planers: Wohin" in prompt
    assert prompt.endswith("NEUE Nachricht: Nach Mittenwald")


async def test_already_geocoded_places_are_skipped(deps: AgentDeps) -> None:
    history = [{"kind": "response", "parts": [
        {"part_kind": "tool-call", "tool_name": "geocode", "args": {"place_name": "Garmisch-Partenkirchen"}},
    ]}]  # fmt: skip
    found = ({"place_name": "garmisch-partenkirchen"}, {"place_name": "Mittenwald"}, {"place_name": " Mittenwald "})
    with intake_finds(*found):
        places = await places_to_geocode("Weiter nach Mittenwald", history, deps)
    assert [p.place_name for p in places] == ["Mittenwald"]


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
