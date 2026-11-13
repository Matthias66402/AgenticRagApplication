"""Agent- und Chat-Tests mit einem geskripteten Modell statt echtem LLM."""

import datetime as dt
import json
import re
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx
from pydantic import ValidationError
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from app.agent.models import REQUEST_TIMEOUT_SECONDS, build_model, model_settings
from app.agent.plan_schema import AdventurePlan, ClarificationQuestion, TextAnswer
from app.agent.planner import (
    DAILY_DISTANCE,
    FIXED_DURATION,
    QUIET_WORDS,
    ROUTE_WORDS,
    check_blocked_sections,
    check_duplicate_routes,
    check_duration_question,
    check_no_plan_evasion,
    check_quiet_wish,
    check_round_trip,
    check_route_wish,
    check_stage_distances,
    check_stays,
    check_wild_camping_question,
    planner_agent,
)
from app.agent.rate_limit import RateLimitedModel, RequestBudget
from app.agent.research import research_agent
from app.agent.tools import weather as weather_module
from app.agent.tools.routing import BROUTER_URL, ORS_URL
from app.agent.tools.transit import WEEKDAYS, next_timetable_change
from app.config import get_settings
from app.evaluation.planner import retry_feedback
from app.schemas.adventure import plan_places
from tests.test_tools import BROUTER_OK, END, START, TRANSIT_OK, add_test_route, one_trip

TOMORROW = dt.date.today() + dt.timedelta(days=1)
# Auftrag an den Recherche-Agenten (in Tests per conftest.fake_research ohne LLM)
RESEARCH = {"region": "Garmisch", "transport": "hiking"}
# Wetter für make_plan (Open-Meteo per conftest.fake_weather)
WEATHER = ("get_weather", {"lat": 47.49, "lon": 11.09, "start_date": TOMORROW.isoformat(), "days": 1})
pytestmark = pytest.mark.usefixtures("fake_weather")


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
        "weather_summary": "Sonnig, 5-15 °C.",
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


def test_request_timeout_for_planner_and_research() -> None:
    """Ohne Timeout wartet das OpenAI-SDK bis zu 600 s je Versuch auf einen stehenden Stream."""
    for model in ("openai:gpt-5-mini", "openai:gpt-4.1-mini", "test"):
        assert model_settings(model)["timeout"] == REQUEST_TIMEOUT_SECONDS  # type: ignore[typeddict-item]
    assert planner_agent.model_settings and planner_agent.model_settings["timeout"] == REQUEST_TIMEOUT_SECONDS  # type: ignore[index]
    assert research_agent.model_settings and research_agent.model_settings["timeout"] == REQUEST_TIMEOUT_SECONDS  # type: ignore[index]


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


def test_stage_shorter_than_straight_line_is_retried() -> None:
    """Etappenorte und Distanz passen nicht zusammen (Chiemsee → Donauwörth mit 73 km statt ~125 km Luftlinie)."""
    plan = make_plan(distance_km=73.14)
    stage = plan["stages"][0]
    stage["start"] = {"name": "Campingplatz", "lat": 48.0795, "lon": 12.4113}
    stage["end"] = {"name": "Donauwörth", "lat": 48.7180, "lon": 10.7807}
    with pytest.raises(ModelRetry, match=r"Tag 1 \(Campingplatz → Donauwörth: 73.14 km angegeben, 1\d\d km Luftlinie"):
        check_stage_distances(AdventurePlan.model_validate(plan))
    stage["distance_km"] = 140  # Route länger als die Luftlinie
    check_stage_distances(AdventurePlan.model_validate(plan))


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


async def test_plan_with_tool_call_and_self_correction(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    model = scripted_model(
        ("estimate_hiking_time", {"distance_km": 15, "ascent_m": 600}),
        ("final_plan", make_plan(distance_km=15)),  # ohne Recherche → Validator verlangt research_knowledge
        ("research_knowledge", RESEARCH),
        ("final_plan", make_plan(distance_km=80)),  # ohne Wetter → Validator verlangt get_weather
        WEATHER,
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
    tools = [c["tool"] for c in stored["messages"][1]["tool_calls"]]
    assert tools == ["estimate_hiking_time", "research_knowledge", "get_weather"]


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
        ("research_knowledge", RESEARCH),
        WEATHER,
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
    tools = [c["tool"] for c in stored["messages"][1]["tool_calls"]]
    assert tools == ["research_knowledge", "get_weather", "find_train_connections"]
    # Datendienste ergänzt das Backend aus dem Verlauf, nicht das Modell
    assert sorted(s["name"] for s in stored["plan"]["data_sources"]) == ["Open-Meteo", "Transitous"]


@respx.mock
async def test_plan_links_bahn_search_without_timetable(
    client: httpx.AsyncClient, user_headers: dict[str, str]
) -> None:
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
    weather = ("get_weather", {"lat": 47.49, "lon": 11.09, "start_date": start.isoformat()})
    await send("Einen Tag", ("research_knowledge", RESEARCH), weather, ("final_plan", plan))
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


@respx.mock
async def test_train_trip_with_bike_from_arrival_station(
    client: httpx.AsyncClient, user_headers: dict[str, str]
) -> None:
    """Zug nach Augsburg, dann mit dem Rad vom Bahnhof zur Jugendherberge: Der Weg gehört in `egress`. Das Modell
    trug ihn erst in `access` ein und plante nach der Zurückweisung den Weg von zu Hause zum Bahnhof."""
    respx.post(ORS_URL.format(profile="cycling-regular")).respond(status_code=503)
    respx.get(BROUTER_URL).respond(json=BROUTER_OK)
    respx.get(get_settings().transit_api_url).respond(json=TRANSIT_TOMORROW)
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    args = {
        "from_name": "Oldenburg Hbf", "from_lat": 53.14, "from_lon": 8.22,
        "to_name": "Bremen", "to_lat": 53.08, "to_lon": 8.88, "date": TOMORROW.isoformat(),
    }  # fmt: skip
    seen: list[str] = []

    def trip(messages: list[ModelMessage], field: str, minutes_after: int) -> dict[str, Any]:
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        seen.extend(p.content for p in returns if p.tool_name == "final_train_trip" and isinstance(p.content, str))
        journey = next(p.content for p in returns if p.tool_name == "find_train_connections")["connections"][0]
        leave = dt.datetime.fromisoformat(journey["arrival"]) + dt.timedelta(minutes=minutes_after)
        leg = {
            "transport": "cycling", "start": START.model_dump() | {"name": "Bremen Hbf"},
            "end": END.model_dump() | {"name": "Jugendherberge"}, "departure": leave.strftime("%Y-%m-%dT%H:%M"),
            "distance_km": 1.6, "duration_hours": 0.1,
            "route_id": next(p.content for p in returns if p.tool_name == "plan_route")["route_id"],
        }  # fmt: skip
        return {"title": "Zug und Rad zur Jugendherberge", "summary": "Zug, dann mit dem Rad zur Jugendherberge.",
                "outbound_journey": journey, field: leg}  # fmt: skip

    steps: list[Callable[[list[ModelMessage]], Step]] = [
        lambda _: ("plan_route", {"waypoints": [START.model_dump(), END.model_dump()], "transport": "cycling"}),
        lambda _: ("find_train_connections", args),
        lambda messages: ("final_train_trip", trip(messages, "access", 10)),  # Weg am Ziel als Weg zum Bahnhof
        lambda messages: ("final_train_trip", trip(messages, "egress", -10)),  # vor Ankunft losfahren
        lambda messages: ("final_train_trip", trip(messages, "egress", 10)),
    ]
    calls = iter(steps)
    with planner_agent.override(model=streaming_model(lambda messages: next(calls)(messages))):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Zug, dann Rad zur Jugendherberge"}
        )
    events = parse_sse(response.text)
    assert [e["type"] for e in events][-2:] == ["plan", "done"], events
    assert "gehört er nicht in `access`, sondern in `egress`" in seen[0] and "vor der Ankunft" in seen[-1]
    plan = events[-2]["plan"]
    assert plan["access"] is None and plan["egress"]["end"]["name"] == "Jugendherberge"
    assert list(plan["routes"]) == [plan["egress"]["route_id"]]

    respx.get(url__startswith=get_settings().map_tile_url.split("{")[0]).respond(status_code=503)
    pdf = await client.get(f"/api/adventures/{adventure['id']}/pdf", headers=user_headers)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    gpx = await client.get(f"/api/adventures/{adventure['id']}/gpx", headers=user_headers)
    assert gpx.status_code == 200 and gpx.headers["x-gpx-has-track"] == "true"
    assert "<name>Start: Bremen Hbf</name>" in gpx.text and "<name>Ziel: Jugendherberge</name>" in gpx.text

    # Übersichtsliste: Orte der Reisekarte für die Suche
    listed = (await client.get("/api/adventures", headers=user_headers)).json()
    places = next(a["places"] for a in listed if a["id"] == adventure["id"])
    assert "Jugendherberge" in places and "Bremen Hbf" in places


def test_plan_places_for_search() -> None:
    """Orte für die Suche in der Übersicht: gekürzt, ohne Duplikate, mit Unterkünften; ohne Plan leer."""
    plan = make_plan(days=2, accommodation="hostel")
    plan["start"] = {"name": "Vareler Hafen, Hohenberge, Varel", "lat": 53.4, "lon": 8.17}
    plan["stages"][0]["end"] = {"name": "Dangast, Varel, Landkreis Friesland", "lat": 53.45, "lon": 8.12}
    plan["stages"][0]["accommodation"] = {"name": "Jugendherberge Dangast", "lat": 53.45, "lon": 8.12}
    assert plan_places(plan) == ["Vareler Hafen", "Garmisch", "Dangast", "Jugendherberge Dangast"]
    assert plan_places(None) == []


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


async def test_message_of_failed_run_stays_in_history(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    """Abenteuer 201: Nach einem Netzwerkfehler fehlte die Antwort des Nutzers, "bitte erneut versuchen!" führte zur
    selben Rückfrage."""
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    prompts: list[list[str]] = []
    calls = 0

    def respond(messages: list[ModelMessage]) -> Step:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise httpx.ConnectError("Verbindung abgelehnt")
        prompts.append([
            part.content for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart)
        ])  # fmt: skip
        return ("ask_user", {"question": "Bis wohin?"})

    with planner_agent.override(model=streaming_model(respond)):
        for text in ["Radtour ab Swinemünde", "Bis nach Kaliningrad", "bitte erneut versuchen!"]:
            await client.post(f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": text})

    assert prompts[1] == ["Radtour ab Swinemünde", "Bis nach Kaliningrad", "bitte erneut versuchen!"]


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


async def test_agent_knows_imported_long_distance_routes(
    client: httpx.AsyncClient, user_headers: dict[str, str]
) -> None:
    """Ohne die Namen aus dem Katalog rät das Modell Fernwegnamen, die eine langsame Live-Suche auslösen."""
    await add_test_route()
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    instructions: list[str] = []

    def respond(messages: list[ModelMessage]) -> Step:
        instructions.append(messages[-1].instructions or "")
        return ("ask_user", {"question": "Wann?"})

    with planner_agent.override(model=streaming_model(respond)):
        await client.post(f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Tour"})

    assert "Fernwege mit vorab importiertem Verlauf" in instructions[0]
    assert "- Testweg (bicycle), auch: Testradweg, T1; Verlauf: Nordheim – Südheim" in instructions[0]


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
        lambda _: ("research_knowledge", RESEARCH),
        lambda _: WEATHER,
        # Dieselbe Route je Tag genannt → nur einmal im Plan, sonst erscheint das Höhenprofil doppelt
        lambda messages: ("final_plan", make_plan() | {"route_ids": [route_id(messages)] * 2}),
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
    """Ist eine Route entlang des Fernwegs berechnet, weicht der Plan nicht auf eine ältere Route ohne Fernweg aus.

    Eine ältere Route entlang des Fernwegs (z. B. mit besser passenden Etappen) ist dagegen in Ordnung.
    """
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
        lambda _: ("plan_route", {"waypoints": waypoints, "transport": "cycling", "route_name": "Testweg"}),
        lambda _: ("research_knowledge", RESEARCH | {"transport": "cycling"}),  # Recherche gilt je Fortbewegungsart
        lambda _: WEATHER,
        lambda messages: plan(messages, 0),  # ältere Route ohne Fernweg
        lambda messages: plan(messages, 1),  # ältere Route entlang des Fernwegs
    ]
    calls = iter(steps)
    with planner_agent.override(model=streaming_model(lambda messages: next(calls)(messages))):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Auf dem Testweg radeln"}
        )

    assert parse_sse(response.text)[-2]["plan"]["route_ids"] == [route_ids[1]]
    assert any("Route ohne Fernweg statt der Route entlang Testweg" in text for text in retries), retries
    assert len(retries) == 1


async def test_plan_needs_weather_summary(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    """Wetter abgefragt, aber nicht übernommen → Korrektur; liefert Open-Meteo einen Fehler, geht der Plan ohne durch."""
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    without = make_plan() | {"weather_summary": None}
    model = scripted_model(
        ("research_knowledge", RESEARCH), WEATHER, ("final_plan", without), ("final_plan", make_plan())
    )
    with planner_agent.override(model=model):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Wanderung"}
        )
    assert parse_sse(response.text)[-2]["plan"]["weather_summary"] == "Sonnig, 5-15 °C."


async def test_plan_without_weather_when_service_fails(
    client: httpx.AsyncClient, user_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unavailable(*args: Any) -> list[dict[str, Any]]:
        raise httpx.ConnectError("Open-Meteo nicht erreichbar")

    monkeypatch.setattr(weather_module, "_forecast", unavailable)
    adventure = (await client.post("/api/adventures", headers=user_headers, json={})).json()
    without = make_plan() | {"weather_summary": None}
    with planner_agent.override(
        model=scripted_model(("research_knowledge", RESEARCH), WEATHER, ("final_plan", without))
    ):
        response = await client.post(
            f"/api/chat/{adventure['id']}", headers=user_headers, json={"message": "Wanderung"}
        )
    assert parse_sse(response.text)[-2]["plan"]["weather_summary"] is None


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


def route_wish_context(message: str, *calls: dict[str, Any]) -> SimpleNamespace:
    """Verlauf mit einer Nutzernachricht und plan_route-Aufrufen mit den angegebenen Argumenten."""
    return SimpleNamespace(messages=[
        ModelRequest(parts=[UserPromptPart(content=message)]),
        ModelResponse(parts=[ToolCallPart(tool_name="plan_route", args=args) for args in calls] or [ToolCallPart(tool_name="geocode", args={})]),
    ])  # fmt: skip


def test_route_wish_requires_named_route_attempt() -> None:
    """Wünscht der Nutzer einen Verlauf, muss plan_route einen Fernweg versucht haben, bevor der Plan durchgeht."""
    plan = AdventurePlan.model_validate(make_plan() | {"transport": "cycling"})
    coast = "Von Emden entlang der Nordseeküste nach Wilhelmshaven radeln"
    with pytest.raises(ModelRetry, match="route_name"):
        check_route_wish(route_wish_context(coast, {"waypoints": [], "transport": "cycling"}), plan)
    # Ein Versuch genügt, auch wenn das Tool keinen passenden Fernweg fand
    check_route_wish(route_wish_context(coast, {"transport": "cycling", "route_name": "Nordseeküsten-Radweg"}), plan)
    with pytest.raises(ModelRetry):
        check_route_wish(route_wish_context("Auf dem Weser-Radweg nach Bremen", {"transport": "cycling"}), plan)
    # Ohne Verlaufswunsch und für Autofahrten keine Pflicht
    check_route_wish(route_wish_context("Von Emden nach Wilhelmshaven radeln", {"transport": "cycling"}), plan)
    car = plan.model_copy(update={"transport": "car"})
    check_route_wish(route_wish_context(coast, {"transport": "car"}), car)


def quiet_context(messages: list[str], *calls: dict[str, Any]) -> SimpleNamespace:
    """Verlauf mit Nutzernachrichten und plan_route-Aufrufen samt Ergebnis (route_id r-0, r-1, ...)."""
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=m)]) for m in messages]
    for index, args in enumerate(calls):
        history.append(ModelResponse(parts=[ToolCallPart(tool_name="plan_route", args=args, tool_call_id=f"c{index}")]))
        history.append(ModelRequest(parts=[ToolReturnPart(
            tool_name="plan_route", content={"route_id": f"r-{index}", "distance_km": 61.2}, tool_call_id=f"c{index}",
        )]))  # fmt: skip
    settings = get_settings().model_copy(update={"avoid_main_roads_enabled": True})
    return SimpleNamespace(messages=history, deps=SimpleNamespace(settings=settings))


def test_quiet_wish_requires_avoid_main_roads() -> None:
    """Abenteuer 206: dreimal dieselbe Route trotz "so wenig wie möglich Hauptstraßen", dazu Rückfragen."""
    plan = AdventurePlan.model_validate(make_plan() | {"transport": "cycling", "route_ids": ["r-0"]})
    wish = [
        "Rundtour über Bad Zwischenahn, ca. 60 km",
        "Gerne die Route anpassen, ich möchte so wenig wie möglich entlang von befahrenen Hauptstraßen fahren.",
    ]
    normal = {"transport": "cycling", "waypoints": []}
    with pytest.raises(ModelRetry, match="avoid_main_roads"):
        check_quiet_wish(quiet_context(wish, normal), plan)
    # Rückfrage "soll ich noch strenger vorgehen?" statt neu zu routen
    question = ClarificationQuestion(question="Soll ich noch strenger vorgehen?")
    with pytest.raises(ModelRetry, match="avoid_main_roads"):
        check_quiet_wish(quiet_context(wish, normal), question)
    quiet = normal | {"avoid_main_roads": True}
    check_quiet_wish(quiet_context(wish, normal, quiet), plan.model_copy(update={"route_ids": ["r-1"]}))
    # Gilt weiter in späteren Nachrichten ("noch strenger"), ohne Wunsch keine Pflicht
    check_quiet_wish(quiet_context([*wish, "Gerne noch strenger vorgehen."], quiet), plan)
    check_quiet_wish(quiet_context(wish[:1], normal), plan)
    # Rückfrage vor der ersten Route (fehlende Angaben) ist in Ordnung
    check_quiet_wish(quiet_context(wish), question)
    # Nachgespielt: ruhige Route berechnet, dann per answer_user drei Varianten zur Auswahl statt des Plans
    menu = TextAnswer(message="Welche Variante möchtest Du: 1) möglichst ruhig, 2) ab Deiner Adresse?")
    with pytest.raises(ModelRetry, match="final_plan"):
        check_quiet_wish(quiet_context(wish, quiet), menu)
    check_quiet_wish(quiet_context(wish, normal), menu)


def test_no_duration_question_with_daily_distance() -> None:
    """Abenteuer 217: Tagesetappen 60-75 km vorgegeben, keine Dauer → "8 oder 9 Tage?" geht einmal zurück."""
    wish = "Malmö nach Kalmar. Tagesetappen sollten zwischen 60 und 75km liegen, ggf. bis 85km."
    route = {"waypoints": [], "transport": "cycling"}
    question = ClarificationQuestion(question="Lieber 8 Tage mit zwei langen Etappen oder 9 Tage im Wunschbereich?")
    ctx = quiet_context([wish], route)
    with pytest.raises(ModelRetry, match="Tage ergeben sich aus der Strecke") as hint:
        check_duration_question(ctx, question)
    ctx.messages.append(ModelRequest(parts=[
        ToolReturnPart(tool_name="ask_user", content=str(hint.value), tool_call_id="c9", outcome="retried"),
    ]))  # fmt: skip
    check_duration_question(ctx, question)  # einmal pro Lauf

    check_duration_question(quiet_context([wish]), question)  # vor der ersten Route (z. B. fehlendes Datum)
    check_duration_question(quiet_context([wish + " Ich habe 5 Tage Zeit."], route), question)  # Widerspruch
    check_duration_question(quiet_context(["Von Malmö nach Kalmar"], route), question)  # keine Tagesdistanz
    other = ClarificationQuestion(question="Ab welchem Datum möchtest du starten?")
    check_duration_question(quiet_context([wish], route), other)
    assert DAILY_DISTANCE.search("Täglich möchte ich zwischen 55 und 75km fahren")
    assert DAILY_DISTANCE.search("etwa 70 km pro Tag") and not FIXED_DURATION.search("ab 15.07.2027 mehrtägig")


def test_main_road_wish_is_no_route_wish() -> None:
    """ "entlang von Hauptstraßen" ist kein Verlauf, für den ein Fernweg gesucht werden müsste."""
    assert not ROUTE_WORDS.search("so wenig wie möglich entlang von befahrenen Hauptstraßen entlang fahren")
    assert not ROUTE_WORDS.search("nicht die Bundesstraße entlang")
    assert ROUTE_WORDS.search("entlang der Nordseeküste nach Wilhelmshaven")
    assert ROUTE_WORDS.search("an der Elbe entlang. Bitte ruhige Wege")
    assert QUIET_WORDS.search("bitte ruhige Wege") and QUIET_WORDS.search("möglichst verkehrsarm")


def test_two_routes_for_same_stretch_are_rejected() -> None:
    """Oldenburg → Wilhelmshaven über den Vareler Hafen plus die direkte Variante: auf der Karte eine Schleife."""
    oldenburg, varel, hafen = [53.148, 8.213], [53.373, 8.127], [53.408, 8.17]
    dangast, whv = [53.446, 8.119], [53.513, 8.141]
    routes = {
        "r-hafen": [oldenburg, varel, hafen, dangast, whv],
        "r-direkt": [oldenburg, varel, [53.41, 8.108], dangast, whv],
        "r-weiter": [whv, [53.6, 8.0]],  # Folgetag ab dem Ziel
        "r-runde-ost": [oldenburg, [53.2, 8.4], oldenburg],  # Standorttour: anderer Tag, gleicher Start und Ziel
        "r-runde-west": [oldenburg, [53.2, 8.0], oldenburg],
    }
    ctx = SimpleNamespace(deps=SimpleNamespace(routes=routes))
    plan = AdventurePlan.model_validate(make_plan() | {"transport": "cycling"})
    with pytest.raises(ModelRetry, match="r-hafen und r-direkt"):
        check_duplicate_routes(ctx, plan.model_copy(update={"route_ids": ["r-hafen", "r-direkt"]}))
    check_duplicate_routes(ctx, plan.model_copy(update={"route_ids": ["r-hafen", "r-weiter"]}))
    check_duplicate_routes(ctx, plan.model_copy(update={"route_ids": ["r-runde-ost", "r-runde-west"]}))


def test_round_trip_on_same_way_back_is_rejected_once() -> None:
    """Abenteuer 210: Oldenburg → Bad Zwischenahn → Apen → Oldenburg, Apen hinter Bad Zwischenahn: Rückweg über
    dieselbe Strecke. Einmal pro Lauf zurück; wünscht der Nutzer ausdrücklich denselben Weg, geht der Plan durch."""
    oldenburg, zwischenahn, apen = [53.139, 8.215], [53.185, 8.003], [53.221, 7.81]
    routes = {
        "r-210": [oldenburg, zwischenahn, apen, zwischenahn, oldenburg],
        "r-211": [oldenburg, zwischenahn, [53.258, 7.927], [53.256, 8.115], [53.247, 8.202], oldenburg],
        "r-hin": [oldenburg, zwischenahn, apen],
        "r-zurueck": [apen, zwischenahn, oldenburg],
    }
    plan = AdventurePlan.model_validate(make_plan() | {"transport": "cycling"})
    ctx = SimpleNamespace(
        deps=SimpleNamespace(routes=routes),
        messages=[ModelRequest(parts=[UserPromptPart(content="Rundtour über Bad Zwischenahn, anderer Rückweg")])],
    )
    with pytest.raises(ModelRetry, match="Schleife") as hint:
        check_round_trip(ctx, plan.model_copy(update={"route_ids": ["r-210"]}))
    # Auf zwei Etappen bzw. Routen verteilt ebenso
    with pytest.raises(ModelRetry):
        check_round_trip(ctx, plan.model_copy(update={"route_ids": ["r-hin", "r-zurueck"]}))
    check_round_trip(ctx, plan.model_copy(update={"route_ids": ["r-211"]}))
    check_round_trip(ctx, plan.model_copy(update={"route_ids": ["r-hin"]}))  # keine Rundtour

    ctx.messages.append(ModelRequest(parts=[
        ToolReturnPart(tool_name="final_plan", content=str(hint.value), tool_call_id="c1", outcome="retried"),
    ]))  # fmt: skip
    check_round_trip(ctx, plan.model_copy(update={"route_ids": ["r-210"]}))

    same_way = SimpleNamespace(deps=ctx.deps, messages=[
        ModelRequest(parts=[UserPromptPart(content="Nach Apen und auf demselben Weg zurück")]),
    ])  # fmt: skip
    check_round_trip(same_way, plan.model_copy(update={"route_ids": ["r-210"]}))


def test_route_wish_hint_comes_once_per_run() -> None:
    """Kein Fernweg für "entlang der Hunte": Nach einem Hinweis geht der Plan durch, statt dass das Modell nach drei
    Zurückweisungen auf eine Textantwort ausweicht. In der nächsten Nachricht kommt der Hinweis wieder."""
    plan = AdventurePlan.model_validate(make_plan() | {"transport": "hiking"})
    ctx = route_wish_context("Gerne auch entlang der Hunte", {"waypoints": [], "transport": "hiking"})
    with pytest.raises(ModelRetry, match="Namen des Flusses") as hint:
        check_route_wish(ctx, plan)
    ctx.messages.append(ModelRequest(parts=[
        ToolReturnPart(tool_name="final_plan", content=str(hint.value), tool_call_id="c1", outcome="retried"),
    ]))  # fmt: skip
    check_route_wish(ctx, plan)
    ctx.messages.append(ModelRequest(parts=[UserPromptPart(content="Und zurück auch entlang der Hunte")]))
    with pytest.raises(ModelRetry):
        check_route_wish(ctx, plan)


def test_stage_without_stay_is_moved_once_per_run() -> None:
    """Abenteuer 201: Etappenende "Checkpoint km 171" ohne Unterkunft. Einmal pro Lauf zurück mit dem Auftrag, das
    Etappenende zu verschieben; findet sich nirgends etwas, geht der Plan beim zweiten Mal durch."""
    data = make_plan(days=3, accommodation="camping") | {"transport": "cycling"}
    stay = {"name": "Kemping Bartek", "type": "camping", "lat": 54.04, "lon": 14.87}
    data["stages"][0] |= {"accommodation": stay}
    data["stages"][1] |= {"end": {"name": "Checkpoint km 171", "lat": 54.3, "lon": 16.3}}
    plan = AdventurePlan.model_validate(data)
    ctx = route_wish_context("Radtour an der Ostsee", {"transport": "cycling"})
    with pytest.raises(ModelRetry, match=r"Tag 2 \(Checkpoint km 171\)\. Verschiebe") as hint:
        check_stays(ctx, plan)
    ctx.messages.append(ModelRequest(parts=[
        ToolReturnPart(tool_name="final_plan", content=str(hint.value), tool_call_id="c1", outcome="retried"),
    ]))  # fmt: skip
    check_stays(ctx, plan)

    # Unterkunft vorhanden, aber Etappenende als Kilometerpunkt benannt
    data["stages"][1] |= {"accommodation": stay, "end": {"name": "km 171", "lat": 54.3, "lon": 16.3}}
    with pytest.raises(ModelRetry, match=r"Tag 2 \(km 171\)"):
        check_stays(route_wish_context("Radtour", {"transport": "cycling"}), AdventurePlan.model_validate(data))
    # Letzter Tag und Tagestouren brauchen keine Unterkunft
    data["stages"][1]["end"] = {"name": "Kołobrzeg", "lat": 54.18, "lon": 15.58}
    check_stays(route_wish_context("Radtour", {"transport": "cycling"}), AdventurePlan.model_validate(data))
    check_stays(route_wish_context("Radtour", {"transport": "cycling"}), AdventurePlan.model_validate(make_plan()))


def test_wild_camping_allowed_needs_no_stay_and_no_question() -> None:
    """Abenteuer 217: "Ist kein Campingplatz verfügbar, nutze ich das Jedermannsrecht": Etappenende ohne Platz ist in
    Ordnung (nur kein Checkpoint als Name), und die Rückfrage "Wildcampen oder Etappen verschieben?" geht zurück."""
    wish = "Malmö nach Kalmar mit Camping. Ist kein Campingplatz verfügbar, nutze ich das Jedermannsrecht."
    data = make_plan(days=3, accommodation="camping") | {"transport": "cycling"}
    data["stages"][1] |= {"end": {"name": "Ronneby", "lat": 56.21, "lon": 15.28}}
    check_stays(route_wish_context(wish, {"transport": "cycling"}), AdventurePlan.model_validate(data))
    data["stages"][1] |= {"end": {"name": "Checkpoint km 340", "lat": 56.21, "lon": 15.28}}
    with pytest.raises(ModelRetry, match="Checkpoint km 340"):
        check_stays(route_wish_context(wish, {"transport": "cycling"}), AdventurePlan.model_validate(data))

    question = ClarificationQuestion(question="Für Nacht 6 kein Campingplatz: A) Wildcampen, B) Etappen verschieben?")
    route = {"waypoints": [], "transport": "cycling"}
    with pytest.raises(ModelRetry, match="Wildcampen"):
        check_wild_camping_question(quiet_context([wish], route), question)
    check_wild_camping_question(quiet_context(["Malmö nach Kalmar mit Camping"], route), question)


def test_no_text_answer_after_rejected_plan() -> None:
    """Nach einem zurückgewiesenen final_plan darf das Modell nicht per answer_user "Route angepasst" melden."""
    ctx = route_wish_context("Gerne auch entlang der Hunte", {"transport": "hiking"})
    check_no_plan_evasion(ctx)
    ctx.messages.append(ModelRequest(parts=[
        ToolReturnPart(tool_name="final_plan", content="Final result processed.", tool_call_id="c0"),
    ]))  # fmt: skip
    check_no_plan_evasion(ctx)  # angenommener Plan, z. B. aus einer früheren Nachricht
    ctx.messages.append(ModelRequest(parts=[
        ToolReturnPart(tool_name="final_plan", content="Startdatum liegt in der Vergangenheit.", tool_call_id="c1", outcome="retried"),
    ]))  # fmt: skip
    with pytest.raises(ModelRetry, match="final_plan"):
        check_no_plan_evasion(ctx)
    # Gespeicherter Verlauf früherer Nachrichten zählt nicht
    ctx.messages.append(ModelRequest(parts=[UserPromptPart(content="Bitte den Bericht schreiben")]))
    check_no_plan_evasion(ctx)


def test_plan_must_admit_blocked_section_of_named_route() -> None:
    """Umfährt die Route einen gesperrten Abschnitt (Abschlussdeich als Baustelle in OSM), darf der Plan nicht so tun,
    als führe er darüber; das System ergänzt einen festen Sicherheitshinweis."""
    blocked = {"from": {"lat": 53.168, "lon": 5.416}, "to": {"lat": 52.95, "lon": 4.719}, "off_route_km": 190,
               "named_route_km": 72}  # fmt: skip
    result = {"route_id": "r-1", "distance_km": 700, "named_route": {"routes": ["Nordseeküsten-Radweg"],
              "blocked_sections": [blocked]}}  # fmt: skip
    ctx = SimpleNamespace(messages=[ModelRequest(parts=[ToolReturnPart(tool_name="plan_route", content=result)])])
    plan = make_plan(days=2, accommodation="camping") | {"transport": "cycling", "route_ids": ["r-1"]}
    plan["stages"][0]["end"] = {"name": "Harlingen", "lat": 53.17, "lon": 5.42}
    plan["stages"][1]["start"] = plan["stages"][0]["end"]
    plan["stages"][1]["end"] = {"name": "Den Helder", "lat": 52.96, "lon": 4.76}
    claim = AdventurePlan.model_validate(plan | {"summary": "Mit der gewünschten Alternative über den Afsluitdijk."})
    with pytest.raises(ModelRetry, match=r"zwischen Harlingen und Den Helder \(190 km Umfahrung statt 72 km"):
        check_blocked_sections(ctx, claim)
    honest = claim.model_copy(update={"summary": "Der Afsluitdijk ist laut Kartendaten nicht befahrbar."})
    notes = check_blocked_sections(ctx, honest).safety_notes
    assert notes[0].startswith("Laut Kartendaten (OpenStreetMap) ist der Fernweg zwischen Harlingen und Den Helder")
    # Andere Route ohne gesperrten Abschnitt: nichts zu tun
    assert check_blocked_sections(ctx, claim.model_copy(update={"route_ids": ["r-2"]})) == claim.model_copy(
        update={"route_ids": ["r-2"]}
    )


def test_plan_must_not_route_over_places_in_blocked_section() -> None:
    """Geocodierte Orte auf dem umfahrenen Abschnitt (Kornwerderzand auf dem Abschlussdeich) darf der Plan nur verneint
    nennen; "nicht befahrbar" an anderer Stelle der Zusammenfassung genügt nicht."""
    blocked = {"from": {"lat": 53.168, "lon": 5.416}, "to": {"lat": 52.95, "lon": 4.719}, "off_route_km": 190,
               "named_route_km": 72, "via": [[53.11, 5.37], [53.072, 5.335], [53.0, 5.17]]}  # fmt: skip
    result = {"route_id": "r-1", "distance_km": 700, "named_route": {"routes": ["Nordseeküsten-Radweg"],
              "blocked_sections": [blocked]}}  # fmt: skip
    geocoded = [
        {"name": "Kornwerderzand, Súdwest-Fryslân, Friesland", "lat": 53.0722, "lon": 5.3372},
        {"name": "Harlingen, Friesland", "lat": 53.1747, "lon": 5.4226},  # Ende des Abschnitts, dort fährt die Route
    ]
    ctx = SimpleNamespace(messages=[ModelRequest(parts=[
        ToolReturnPart(tool_name="geocode", content=geocoded), ToolReturnPart(tool_name="plan_route", content=result),
    ])])  # fmt: skip
    plan = AdventurePlan.model_validate(make_plan() | {"transport": "cycling", "route_ids": ["r-1"], "summary": (
        "Die Route führt über den Afsluitdijk (Kornwerderzand) als gewünschte Alternativroute; ein Abschnitt ist laut "
        "Kartendaten nicht befahrbar. Start in Harlingen."
    )})  # fmt: skip
    with pytest.raises(ModelRetry, match=r"führt nicht über Kornwerderzand.*„Die Route führt über den Afsluitdijk"):
        check_blocked_sections(ctx, plan)
    honest = "Die Route umfährt den Afsluitdijk (Kornwerderzand), er ist laut Kartendaten nicht befahrbar."
    check_blocked_sections(ctx, plan.model_copy(update={"summary": honest}))
