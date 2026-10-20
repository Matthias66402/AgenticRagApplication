"""Agent- und Chat-Tests mit einem geskripteten Modell statt echtem LLM."""

import datetime as dt
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from app.agent.plan_schema import AdventurePlan
from app.agent.planner import planner_agent

TOMORROW = dt.date.today() + dt.timedelta(days=1)


def make_plan(distance_km: float = 15.0, days: int = 1, accommodation: str | None = None) -> dict[str, Any]:
    place = {"name": "Garmisch", "lat": 47.49, "lon": 11.09}
    return {
        "title": "Testtour",
        "summary": "Eine Testtour.",
        "transport": "hiking",
        "start_date": TOMORROW.isoformat(),
        "duration_days": days,
        "start": place,
        "destination": place,
        "accommodation_type": accommodation,
        "stages": [
            {
                "day": d + 1,
                "date": (TOMORROW + dt.timedelta(days=d)).isoformat(),
                "start": place,
                "end": place,
                "distance_km": distance_km,
            }
            for d in range(days)
        ],
        "route_ids": [],
        "sources": ["Testquelle"],
    }


Step = tuple[str, dict[str, Any]] | str


def streaming_model(respond: Callable[[list[ModelMessage]], Step]) -> FunctionModel:
    """FunctionModel für den Streaming-Pfad: `respond` liefert (tool_name, args) oder Text."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        step = respond(messages)
        if isinstance(step, str):
            yield step
        else:
            name, args = step
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    return FunctionModel(stream_function=stream)


def scripted_model(*steps: Step) -> FunctionModel:
    """Jeder Modellaufruf liefert den nächsten Schritt."""
    calls = iter(steps)
    return streaming_model(lambda _: next(calls))


def test_multi_day_requires_accommodation() -> None:
    with pytest.raises(ValidationError, match="accommodation_type"):
        AdventurePlan.model_validate(make_plan(days=2))
    with pytest.raises(ValidationError, match="genau 2 Etappen"):
        AdventurePlan.model_validate(make_plan(days=2, accommodation="hut") | {"stages": make_plan()["stages"]})
    assert AdventurePlan.model_validate(make_plan(days=2, accommodation="hut")).duration_days == 2


async def test_question_when_information_missing(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    model = scripted_model(("ask_user", {"question": "Wo startest Du?", "missing_fields": ["start"]}))
    with planner_agent.override(model=model):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Radtour morgen an den Bodensee"}
        )
    events = parse_sse(response.text)
    assert [e["type"] for e in events][-2:] == ["question", "done"], events
    assert events[-2]["question"] == "Wo startest Du?"

    stored = (await client.get(f"/api/adventures/{adventure['id']}", headers=user_headers)).json()
    assert stored["status"] == "draft"
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]


async def test_plan_with_tool_call_and_self_correction(
    client: httpx.AsyncClient, user_headers: dict[str, str]
) -> None:
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    model = scripted_model(
        ("estimate_hiking_time", {"distance_km": 15, "ascent_m": 600}),
        ("final_plan", make_plan(distance_km=15)),  # ohne RAG-Suche → Validator verlangt search_knowledge
        ("search_knowledge", {"query": "Tageswanderung Ausrüstung"}),
        ("final_plan", make_plan(distance_km=80)),  # unplausibel lange Etappe → erneute Korrektur
        ("final_plan", make_plan(distance_km=15)),
    )
    with planner_agent.override(model=model):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Tageswanderung ab Garmisch morgen"}
        )
    events = parse_sse(response.text)
    types = [e["type"] for e in events]
    assert "tool_call" in types and "tool_result" in types
    assert types[-2:] == ["plan", "done"]
    assert events[-2]["plan"]["stages"][0]["distance_km"] == 15

    stored = (await client.get(f"/api/adventures/{adventure['id']}", headers=user_headers)).json()
    assert stored["status"] == "planned"
    assert stored["title"] == "Testtour"
    assert [c["tool"] for c in stored["messages"][1]["tool_calls"]] == ["estimate_hiking_time", "search_knowledge"]


async def test_follow_up_uses_history(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    seen_history: list[int] = []

    def respond(messages: list[ModelMessage]) -> Step:
        seen_history.append(len(messages))
        return ("ask_user", {"question": "Und wann?"})

    with planner_agent.override(model=streaming_model(respond)):
        for text in ["Wanderung zur Zugspitze", "Ab Garmisch"]:
            await client.post(f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": text})
    # Der zweite Lauf sieht den Verlauf des ersten
    assert seen_history[1] > seen_history[0]


async def test_cannot_chat_in_foreign_adventure(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    from tests.conftest import auth_headers, create_user

    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    await create_user("other@example.com")
    other = await auth_headers(client, "other@example.com")
    assert (await client.get(f"/api/adventures/{adventure['id']}", headers=other)).status_code == 404
    response = await client.post(f"/api/chat/{adventure['id']}", headers=other, json={"message": "hi"})
    assert response.status_code == 404


def parse_sse(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for block in body.strip().split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]
