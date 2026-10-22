"""Agent- und Chat-Tests mit einem geskripteten Modell statt echtem LLM."""

import datetime as dt
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
import respx
from pydantic import ValidationError
from pydantic_ai import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from app.agent.plan_schema import AdventurePlan
from app.agent.planner import planner_agent
from app.agent.tools.routing import BROUTER_URL, ORS_URL
from app.config import get_settings
from tests.test_tools import BROUTER_OK, END, START, TRANSIT_OK

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


JOURNEY = {
    "departure": f"{TOMORROW}T07:12",
    "arrival": f"{TOMORROW}T09:28",
    "duration_minutes": 136,
    "transfers": 0,
    "legs": [
        {
            "line": "RB6",
            "from_stop": "München Hbf",
            "to_stop": "Garmisch-Partenkirchen",
            "departure": f"{TOMORROW}T07:12",
            "arrival": f"{TOMORROW}T09:28",
            "track": "28",
        }
    ],
}


def test_journey_times_are_local_without_timezone() -> None:
    plan = make_plan() | {"return_journey": JOURNEY}
    assert AdventurePlan.model_validate(plan).return_journey.departure == f"{TOMORROW}T07:12"
    with pytest.raises(ValidationError, match="pattern"):
        AdventurePlan.model_validate(plan | {"return_journey": JOURNEY | {"departure": f"{TOMORROW}T05:12:00Z"}})


@respx.mock
async def test_journey_only_from_train_connections(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": []})
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    plan = make_plan() | {"outbound_journey": JOURNEY}
    connection_args = {
        "from_name": "München", "from_lat": 48.14, "from_lon": 11.56,
        "to_name": "Garmisch", "to_lat": 47.49, "to_lon": 11.09, "date": TOMORROW.isoformat(),
    }  # fmt: skip
    model = scripted_model(
        ("search_knowledge", {"query": "Anreise Bahn"}),
        ("final_plan", plan),  # Verbindung ohne Fahrplanabfrage → Validator verlangt find_train_connections
        ("find_train_connections", connection_args),
        ("final_plan", plan),
    )
    with planner_agent.override(model=model):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Wanderung, Anreise ab München"}
        )
    events = parse_sse(response.text)
    assert events[-2]["type"] == "plan", events
    assert events[-2]["plan"]["outbound_journey"]["legs"][0]["line"] == "RB6"
    assert events[-2]["plan"]["return_journey"] is None
    stored = (await client.get(f"/api/adventures/{adventure['id']}", headers=user_headers)).json()
    assert [c["tool"] for c in stored["messages"][1]["tool_calls"]] == ["search_knowledge", "find_train_connections"]


@respx.mock
async def test_train_connection_answer_without_plan(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    transit = respx.get(get_settings().transit_api_url).respond(json=TRANSIT_OK)
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    args = {
        "from_name": "München", "from_lat": 48.14, "from_lon": 11.56,
        "to_name": "Garmisch-Partenkirchen", "to_lat": 47.49, "to_lon": 11.1, "date": TOMORROW.isoformat(),
    }  # fmt: skip
    model = scripted_model(
        ("find_train_connections", args),
        ("find_train_connections", args),  # Wiederholung geht nicht erneut an die API
        ("answer_user", {"message": "Die Direktverbindung mit dem RB6 ist am schnellsten."}),
    )
    with planner_agent.override(model=model):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Zug München → Garmisch morgen"}
        )
    events = parse_sse(response.text)
    assert [e["type"] for e in events][-2:] == ["answer", "done"], events
    answer = events[-2]["content"]
    assert answer.startswith("Die Direktverbindung mit dem RB6 ist am schnellsten.")
    assert "🚆 München → Garmisch-Partenkirchen" in answer
    assert "08:28 RB6: München Hbf (Gl. 28) → Garmisch-Partenkirchen 09:52" in answer
    assert "– fällt aus!" in answer
    assert "Fahrplandaten: Transitous" in answer
    assert transit.call_count == 1

    stored = (await client.get(f"/api/adventures/{adventure['id']}", headers=user_headers)).json()
    assert stored["status"] == "draft" and stored["plan"] is None
    assert stored["messages"][1]["content"] == answer


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


@respx.mock
async def test_plan_uses_route_from_run_with_question(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    """Route berechnet, dann Rückfrage; der Plan im nächsten Lauf verweist auf diese route_id und zeigt ihre Geometrie."""
    respx.post(ORS_URL.format(profile="foot-hiking")).respond(status_code=503)
    respx.get(BROUTER_URL).respond(json=BROUTER_OK)
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()

    def route_id(messages: list[ModelMessage]) -> str:
        return next(
            part.content["route_id"]
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "plan_route"
        )

    steps: list[Callable[[list[ModelMessage]], Step]] = [
        lambda _: ("plan_route", {"waypoints": [START.model_dump(), END.model_dump()], "transport": "hiking"}),
        lambda _: ("ask_user", {"question": "Wann möchtest Du los?"}),
        lambda _: ("search_knowledge", {"query": "Tageswanderung Ausrüstung"}),
        lambda messages: ("final_plan", make_plan() | {"route_ids": [route_id(messages)]}),
    ]
    calls = iter(steps)
    with planner_agent.override(model=streaming_model(lambda messages: next(calls)(messages))):
        for text in ["Wanderung morgen", "Um 9 Uhr"]:
            response = await client.post(f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": text})

    plan = parse_sse(response.text)[-2]["plan"]
    assert len(plan["route_ids"]) == 1
    assert [p[:2] for p in plan["routes"][plan["route_ids"][0]]] == [[53.1, 8.2], [52.9, 8.2]]
    stored = (await client.get(f"/api/adventures/{adventure['id']}", headers=user_headers)).json()
    assert stored["plan"]["routes"] == plan["routes"]


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
