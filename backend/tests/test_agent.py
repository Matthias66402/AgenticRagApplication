"""Agent- und Chat-Tests mit einem geskripteten Modell statt echtem LLM."""

import datetime as dt
import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx
from pydantic import ValidationError
from pydantic_ai import Agent, ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from app.agent.plan_schema import AdventurePlan
from app.agent.planner import build_model, model_settings, planner_agent
from app.agent.rate_limit import RateLimitedModel, RequestBudget
from app.agent.tools.routing import BROUTER_URL, ORS_URL
from app.agent.tools.transit import WEEKDAYS, next_timetable_change
from app.evaluation.planner import retry_feedback
from app.config import get_settings
from tests.test_tools import BROUTER_OK, END, START, TRANSIT_OK, add_test_route, one_trip

TOMORROW = dt.date.today() + dt.timedelta(days=1)


def _tomorrow_same_local_time(match: re.Match[str]) -> str:
    berlin = ZoneInfo("Europe/Berlin")
    local = dt.datetime.fromisoformat(match.group(0)).astimezone(berlin)
    moved = dt.datetime.combine(TOMORROW, local.time(), berlin)
    return moved.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Fahrplanantwort mit den Verbindungen der Fixture aus test_tools (festes, inzwischen vergangenes Datum) zur selben
# Ortszeit morgen; über die Umstellung auf Sommer- bzw. Winterzeit hinweg ändern sich dabei die UTC-Zeiten
TRANSIT_TOMORROW = json.loads(
    re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", _tomorrow_same_local_time, json.dumps(TRANSIT_OK))
)


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


def test_reasoning_summary_only_for_reasoning_models() -> None:
    assert model_settings("openai:gpt-5-mini")["openai_reasoning_summary"] == "detailed"  # type: ignore[typeddict-item]
    assert "openai_reasoning_summary" not in model_settings("openai:gpt-4.1-mini")
    assert "openai_reasoning_summary" not in model_settings("test")


async def test_request_budget_keeps_limit_in_every_minute() -> None:
    now = [0.0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    budget = RequestBudget(3, clock=lambda: now[0], sleep=sleep)
    sent = []
    for step in [0, 10, 10, 0, 0, 50]:  # Abstände zwischen den Anfragen in Sekunden
        now[0] += step
        await budget.acquire()
        sent.append(now[0])
    # Die 4. Anfrage wartet, bis die 1. (t=0) eine Minute zurückliegt, die 5. auf die 2. (t=10)
    assert sent == [0, 10, 20, 61, 71, 121]
    assert all(sum(t - 60 < s <= t for s in sent) <= 3 for t in sent)


async def test_rate_limited_model_waits_before_each_request() -> None:
    model = RateLimitedModel(scripted_model("ok", "ok"), per_minute=60)
    acquired = []
    original = model.budget.acquire

    async def acquire() -> float:
        acquired.append(True)
        return await original()

    model.budget.acquire = acquire  # type: ignore[method-assign]
    async with Agent(model).run_stream_events("Hallo") as events:
        async for _ in events:
            pass
    assert acquired == [True]


def test_model_is_throttled_only_with_limit() -> None:
    assert build_model("openai:gpt-5-mini") == "openai:gpt-5-mini"
    assert isinstance(build_model("test", requests_per_minute=5), RateLimitedModel)


def test_reasoning_effort_only_when_configured() -> None:
    assert model_settings("openai:gpt-5-mini", "low")["openai_reasoning_effort"] == "low"  # type: ignore[typeddict-item]
    assert "openai_reasoning_effort" not in model_settings("openai:gpt-5-mini")
    assert "openai_reasoning_effort" not in model_settings("openai:gpt-4.1-mini", "low")


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


def utc(local: str) -> str:
    """Ortszeit in Berlin → UTC-Zeitstempel wie bei der MOTIS-API."""
    moment = dt.datetime.fromisoformat(local).replace(tzinfo=ZoneInfo("Europe/Berlin"))
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:00Z")


@respx.mock
async def test_journey_only_from_train_connections(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    trip = one_trip(utc(JOURNEY["departure"]), utc(JOURNEY["arrival"]))
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": [trip]})
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
        # nicht gefundene Verbindung → Validator verlangt eine gefundene
        ("final_plan", plan | {"return_journey": JOURNEY | {"departure": f"{TOMORROW}T08:00"}}),
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
async def test_plan_links_bahn_search_without_timetable(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    """Nach dem Fahrplanwechsel ohne Fahrplandaten hängt das Backend den bahn.de-Link an den Plan, nicht das LLM; auch
    wenn der Agent die Bahn in einer früheren Nachricht abgefragt hat oder nur antwortet."""
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": []})
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    start = next_timetable_change(dt.date.today()) + dt.timedelta(days=7)
    plan = make_plan() | {"start_date": start.isoformat()}
    plan["stages"][0]["date"] = start.isoformat()
    route = {
        "from_name": "München", "from_lat": 48.14, "from_lon": 11.56,
        "to_name": "Garmisch", "to_lat": 47.49, "to_lon": 11.09,
    }  # fmt: skip

    async def send(message: str, *steps: Step) -> None:
        with planner_agent.override(model=scripted_model(*steps)):
            await client.post(f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": message})

    async def links() -> list[dict[str, Any]]:
        stored = (await client.get(f"/api/adventures/{adventure['id']}", headers=user_headers)).json()
        return stored["plan"]["timetable_links"]

    # Abfrage, dann Rückfrage; der Plan folgt erst mit der nächsten Nachricht ohne neue Abfrage
    outside = route | {"date": (start + dt.timedelta(days=30)).isoformat()}  # gehört nicht zur Tour
    await send(
        "Anreise ab München",
        ("find_train_connections", route | {"date": start.isoformat(), "time": "06:00"}),
        ("find_train_connections", outside),
        ("ask_user", {"question": "Wie lange?"}),
    )
    await send("Einen Tag", ("search_knowledge", {"query": "Anreise Bahn"}), ("final_plan", plan))
    [link] = await links()
    assert link["label"] == f"München → Garmisch · {WEEKDAYS[start.weekday()]} {start:%d.%m.%Y}, ab 06:00"
    assert link["url"].startswith("https://www.bahn.de/buchung/fahrplan/suche#")
    assert link["missing"] is True

    # Rückreise abgefragt, aber nur geantwortet: Link kommt trotzdem in den Plan
    back = {**route, "from_name": "Garmisch", "to_name": "München", "date": (start + dt.timedelta(days=1)).isoformat()}
    await send("Und zurück?", ("find_train_connections", back), ("answer_user", {"message": "Noch kein Fahrplan."}))
    assert [link["label"][:18] for link in await links()] == ["München → Garmisch", "Garmisch → München"]


@respx.mock
async def test_train_connection_answer_without_plan(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    transit = respx.get(get_settings().transit_api_url).respond(json=TRANSIT_TOMORROW)
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


@respx.mock
async def test_train_trip_card_for_chosen_connection(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    """Folgenachricht "Plan für Verbindung 1": Reisekarte statt Verweis auf nicht angezeigte Verbindungen."""
    respx.get(get_settings().transit_api_url).respond(json=TRANSIT_TOMORROW)
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    args = {
        "from_name": "München", "from_lat": 48.14, "from_lon": 11.56,
        "to_name": "Garmisch-Partenkirchen", "to_lat": 47.49, "to_lon": 11.1, "date": TOMORROW.isoformat(),
    }  # fmt: skip

    def chosen(messages: list[ModelMessage]) -> dict[str, Any]:
        return next(
            part.content["connections"][0]
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "find_train_connections"
        )

    def without_lines(journey: dict[str, Any]) -> dict[str, Any]:
        return journey | {"legs": [leg | {"line": "?"} for leg in journey["legs"]]}

    def trip(journey: dict[str, Any]) -> dict[str, Any]:
        return {"title": "Bahnfahrt München → Garmisch", "summary": "Mit dem RB6.", "outbound_journey": journey}

    steps: list[Callable[[list[ModelMessage]], Step]] = [
        lambda _: ("find_train_connections", args),
        lambda _: ("answer_user", {"message": "Die Verbindungen siehst Du unten."}),
        # Folgenachricht ohne neue Abfrage: Verweis auf "unten" und erfundene Verbindung gehen zurück an das Modell
        lambda _: ("answer_user", {"message": "Die Details siehst Du unten."}),
        lambda _: ("final_train_trip", trip(JOURNEY)),
        # Teilstrecken ohne Linie: Die Reisekarte übernimmt sie unverändert aus dem Fahrplanergebnis
        lambda messages: ("final_train_trip", trip(without_lines(chosen(messages)))),
    ]
    calls = iter(steps)

    async def send(message: str) -> list[dict[str, Any]]:
        response = await client.post(f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": message})
        return parse_sse(response.text)

    with planner_agent.override(model=streaming_model(lambda messages: next(calls)(messages))):
        first = await send("Zug morgen")
        second = await send("Plan für Verbindung 1")

    assert first[-2]["type"] == "answer" and "🚆 München → Garmisch-Partenkirchen" in first[-2]["content"]
    assert [e["type"] for e in second][-2:] == ["plan", "done"], second
    plan = second[-2]["plan"]
    assert plan["kind"] == "train_trip"
    assert plan["outbound_journey"]["legs"][0]["line"] == "RB74"  # erste gefundene, nicht die erfundene Verbindung

    stored = (await client.get(f"/api/adventures/{adventure['id']}", headers=user_headers)).json()
    assert stored["status"] == "planned" and stored["title"] == "Bahnfahrt München → Garmisch"
    assert stored["plan"]["kind"] == "train_trip"
    # Ohne Weg zum Bahnhof: PDF mit der Verbindung, aber kein GPX (Verbindungen haben keine Koordinaten)
    pdf = await client.get(f"/api/adventures/{adventure['id']}/pdf", headers=user_headers)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    gpx = await client.get(f"/api/adventures/{adventure['id']}/gpx", headers=user_headers)
    assert gpx.status_code == 409


@respx.mock
async def test_train_trip_with_bike_to_station(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    """Mit dem Rad zum Bahnhof, dann Zug: Reisekarte mit berechneter Route, die vor der Abfahrt ankommt."""
    respx.post(ORS_URL.format(profile="cycling-regular")).respond(status_code=503)
    respx.get(BROUTER_URL).respond(json=BROUTER_OK)
    respx.get(get_settings().transit_api_url).respond(json=TRANSIT_TOMORROW)
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    args = {
        "from_name": "Oldenburg Hbf", "from_lat": 53.14, "from_lon": 8.22,
        "to_name": "Bremen", "to_lat": 53.08, "to_lon": 8.88, "date": TOMORROW.isoformat(),
    }  # fmt: skip

    def tool_result(messages: list[ModelMessage], tool: str) -> Any:
        return next(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == tool
        )

    def trip(messages: list[ModelMessage], minutes_before: int, route_id: str | None = None) -> dict[str, Any]:
        journey = tool_result(messages, "find_train_connections")["connections"][0]
        leave = dt.datetime.fromisoformat(journey["departure"]) - dt.timedelta(minutes=minutes_before)
        access = {
            "transport": "cycling", "start": START.model_dump() | {"name": "Zuhause"},
            "end": END.model_dump() | {"name": "Oldenburg Hbf"}, "departure": leave.strftime("%Y-%m-%dT%H:%M"),
            "distance_km": 2.1, "duration_hours": 0.25,
            "route_id": route_id or tool_result(messages, "plan_route")["route_id"],
        }  # fmt: skip
        return {"title": "Rad und Zug nach Bremen", "summary": "Mit dem Rad zum Bahnhof.", "outbound_journey": journey,
                "access": access}  # fmt: skip

    steps: list[Callable[[list[ModelMessage]], Step]] = [
        lambda _: ("plan_route", {"waypoints": [START.model_dump(), END.model_dump()], "transport": "cycling"}),
        lambda _: ("find_train_connections", args),
        lambda messages: ("final_train_trip", trip(messages, 30, route_id="r-erfunden")),  # keine berechnete Route
        lambda messages: ("final_train_trip", trip(messages, 10)),  # 15 min Fahrzeit → zu spät am Bahnhof
        lambda messages: ("final_train_trip", trip(messages, 25)),
    ]
    calls = iter(steps)
    with planner_agent.override(model=streaming_model(lambda messages: next(calls)(messages))):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Mit dem Rad zum Bahnhof, dann Zug"}
        )
    events = parse_sse(response.text)
    assert [e["type"] for e in events][-2:] == ["plan", "done"], events
    plan = events[-2]["plan"]
    assert plan["kind"] == "train_trip"
    assert plan["access"]["transport"] == "cycling"
    assert list(plan["routes"]) == [plan["access"]["route_id"]] and plan["routes"][plan["access"]["route_id"]]

    respx.get(url__startswith=get_settings().map_tile_url.split("{")[0]).respond(status_code=503)
    pdf = await client.get(f"/api/adventures/{adventure['id']}/pdf", headers=user_headers)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    gpx = await client.get(f"/api/adventures/{adventure['id']}/gpx", headers=user_headers)
    assert gpx.status_code == 200 and gpx.headers["x-gpx-has-track"] == "true"
    assert "<name>Start: Zuhause</name>" in gpx.text and "<name>Bahnhof: Oldenburg Hbf</name>" in gpx.text


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


async def test_agent_knows_home_and_default_transport(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    instructions: list[str] = []

    def respond(messages: list[ModelMessage]) -> Step:
        instructions.append(messages[-1].instructions or "")
        return ("ask_user", {"question": "Wann?"})

    with planner_agent.override(model=streaming_model(respond)):
        await client.post(f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Tour"})
        profile = {"home_address": "Nelkenstraße 10", "home_city": "26121 Oldenburg", "default_transport": "cycling"}
        await client.patch("/api/users/me", headers=user_headers, json=profile)
        await client.post(f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Morgen"})

    assert "Profil-Vorgaben des Nutzers" not in instructions[0]
    assert "Profil-Vorgaben des Nutzers" in instructions[1]
    assert "- Wohnadresse: Nelkenstraße 10, 26121 Oldenburg." in instructions[1]
    assert "- Standard-Fortbewegungsart: Radtour (transport=cycling)." in instructions[1]


async def set_default_transport(client: httpx.AsyncClient, headers: dict[str, str], transport: str | None) -> None:
    await client.patch("/api/users/me", headers=headers, json={"default_transport": transport})


async def test_default_transport_instead_of_question(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    """Mit Standard-Fortbewegungsart geht eine Rückfrage nach dem Verkehrsmittel an das Modell zurück."""
    await set_default_transport(client, user_headers, "cycling")
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    model = scripted_model(
        ("ask_user", {"question": "Wie möchtest Du fahren?", "missing_fields": ["transport"]}),
        ("ask_user", {"question": "Wann geht es los?", "missing_fields": ["start_date"]}),
    )
    with planner_agent.override(model=model):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Von Osnabrück nach Hause fahren"}
        )
    assert parse_sse(response.text)[-2]["question"] == "Wann geht es los?"


@respx.mock
async def test_default_transport_instead_of_train_answer(
    client: httpx.AsyncClient, user_headers: dict[str, str]
) -> None:
    respx.get(get_settings().transit_api_url).respond(json=TRANSIT_TOMORROW)
    connection_args = {
        "from_name": "Osnabrück", "from_lat": 52.27, "from_lon": 8.06,
        "to_name": "Oldenburg", "to_lat": 53.14, "to_lon": 8.21, "date": TOMORROW.isoformat(),
    }  # fmt: skip
    steps = [
        ("find_train_connections", connection_args),
        ("answer_user", {"message": "Der RE9 ist am schnellsten."}),
        ("ask_user", {"question": "Wann startest Du?", "missing_fields": ["start_date"]}),
    ]

    async def ask(message: str) -> list[dict[str, Any]]:
        adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
        with planner_agent.override(model=scripted_model(*steps)):
            response = await client.post(
                f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": message}
            )
        return parse_sse(response.text)

    # Ohne Verkehrsmittel und mit Standard Radtour ist eine reine Bahnauskunft unerwünscht
    await set_default_transport(client, user_headers, "cycling")
    assert (await ask("Morgen von Osnabrück nach Hause fahren"))[-2]["type"] == "question"
    # Nennt der Nutzer die Bahn bzw. hat er keine Vorgabe, bleibt die Bahnauskunft
    assert (await ask("Morgen mit der Bahn von Osnabrück nach Hause"))[-2]["type"] == "answer"
    await set_default_transport(client, user_headers, None)
    assert (await ask("Morgen von Osnabrück nach Hause fahren"))[-2]["type"] == "answer"


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


@respx.mock
async def test_plan_must_use_route_along_long_distance_route(
    client: httpx.AsyncClient, user_headers: dict[str, str]
) -> None:
    """Ist eine Route entlang des Fernwegs berechnet, weicht der Plan nicht auf eine ältere Route ohne Fernweg aus."""
    await add_test_route()
    respx.post(url__startswith="https://api.openrouteservice.org/").respond(status_code=503)
    respx.get(BROUTER_URL).respond(json=BROUTER_OK)
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    waypoints = [START.model_dump(), END.model_dump()]
    route_ids: list[str] = []
    retries: list[str] = []

    def plan(messages: list[ModelMessage], route_id: int) -> Step:
        route_ids[:] = [
            part.content["route_id"]
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "plan_route"
        ]
        retries.extend(
            feedback for message in messages for part in message.parts if (feedback := retry_feedback(part)) is not None
        )
        return "final_plan", make_plan() | {"transport": "cycling", "route_ids": [route_ids[route_id]]}

    steps: list[Callable[[list[ModelMessage]], Step]] = [
        lambda _: ("plan_route", {"waypoints": waypoints, "transport": "cycling"}),
        lambda _: ("plan_route", {"waypoints": waypoints, "transport": "cycling", "route_name": "Testweg"}),
        lambda _: ("search_knowledge", {"query": "Ausrüstung Radtour"}),
        lambda messages: plan(messages, 0),  # ältere Route ohne Fernweg
        lambda messages: plan(messages, 1),
    ]
    calls = iter(steps)
    with planner_agent.override(model=streaming_model(lambda messages: next(calls)(messages))):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Auf dem Testweg radeln"}
        )

    assert parse_sse(response.text)[-2]["plan"]["route_ids"] == [route_ids[1]]
    assert any("ältere Route statt der Route entlang Testweg" in text for text in retries), retries


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
