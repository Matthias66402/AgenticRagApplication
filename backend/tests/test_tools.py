import datetime as dt
import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AccommodationType, TransportMode
from app.agent.tools import accommodation, routing
from app.agent.tools import geocode as geocode_module
from app.agent.tools.accommodation import find_accommodation
from app.agent.tools.geocode import geocode
from app.agent.tools.routing import BROUTER_URL, ORS_URL, Coordinate, elevation_gain, estimate_hiking_time, plan_route
from app.agent.tools.sun import sun_times
from app.agent.tools.transit import find_train_connections
from app.agent.tools.weather import ARCHIVE_URL, FORECAST_URL, get_weather
from app.config import get_settings
from app.db import SessionLocal

TODAY = dt.date(2026, 7, 1)


@pytest.fixture(autouse=True)
def reset_overpass_state(monkeypatch) -> None:
    monkeypatch.setattr(accommodation, "_cache", {})
    monkeypatch.setattr(accommodation, "_unavailable_until", 0.0)
    monkeypatch.setattr(routing, "_slow_until", {})
    monkeypatch.setattr(geocode_module, "_last_request", 0.0)

    async def no_throttle() -> None:
        pass

    monkeypatch.setattr(geocode_module, "_throttle", no_throttle)


@pytest.fixture
async def ctx():
    settings = get_settings().model_copy(update={"ors_api_key": "test-key"})
    async with httpx.AsyncClient() as http:
        yield SimpleNamespace(
            deps=AgentDeps(settings=settings, http=http, session_factory=SessionLocal, user_name="T", today=TODAY)
        )


@respx.mock
async def test_geocode(ctx) -> None:
    geocode_module._cache.clear()
    respx.get(geocode_module.NOMINATIM_URL).respond(
        json=[{"display_name": "Garmisch", "lat": "47.49", "lon": "11.09", "category": "boundary", "type": "x"}]
    )
    result = await geocode(ctx, "Garmisch")
    assert result[0]["lat"] == 47.49


@respx.mock
async def test_geocode_not_found(ctx) -> None:
    geocode_module._cache.clear()
    respx.get(geocode_module.NOMINATIM_URL).respond(json=[])
    assert "error" in await geocode(ctx, "Nirgendwo-XYZ")


@respx.mock
async def test_route_with_ors(ctx) -> None:
    respx.post(ORS_URL.format(profile="foot-hiking")).respond(
        json={
            "features": [
                {
                    "geometry": {"coordinates": [[11.0, 47.0, 700], [11.1, 47.1, 900]]},
                    "properties": {
                        "summary": {"distance": 21000, "duration": 18000},
                        "ascent": 900,
                        "descent": 400,
                        "segments": [{"distance": 21000, "duration": 18000, "ascent": 900, "descent": 400}],
                    },
                }
            ]
        }
    )
    result = await plan_route(ctx, [Coordinate(lat=47, lon=11), Coordinate(lat=47.1, lon=11.1)], TransportMode.hiking)
    assert result["distance_km"] == 21.0
    assert result["ascent_m"] == 900
    # Geometrie landet in den Deps (für die Karte), nicht im LLM-Ergebnis
    assert ctx.deps.routes[result["route_id"]] == [[47.0, 11.0, 700.0], [47.1, 11.1, 900.0]]
    assert "geometry" not in result


ORS_OK = {
    "features": [
        {
            "geometry": {"coordinates": [[8.2, 53.1, 5], [8.4, 52.9, 20]]},
            "properties": {"summary": {"distance": 36000, "duration": 7200}, "ascent": 40, "descent": 25, "segments": []},
        }
    ]
}


# Gerade Strecke 53.10 → 52.90 entlang 8.2 mit Zwischenpunkt bei 53.00; Höhe steigt erst, fällt dann
BROUTER_OK = {
    "features": [
        {
            "geometry": {"coordinates": [
                [8.2, 53.10, 10.0], [8.2, 53.05, 40.0], [8.2, 53.00, 60.0], [8.2, 52.95, 30.0], [8.2, 52.90, 15.0],
            ]},
            "properties": {"track-length": "22300", "total-time": "4000"},
        }
    ]
}  # fmt: skip
START, VIA, END = Coordinate(lat=53.1, lon=8.2), Coordinate(lat=53.0, lon=8.2), Coordinate(lat=52.9, lon=8.2)


@respx.mock
async def test_route_falls_back_to_brouter(ctx) -> None:
    respx.post(ORS_URL.format(profile="cycling-regular")).mock(side_effect=httpx.ReadTimeout("langsam"))
    brouter = respx.get(BROUTER_URL).respond(json=BROUTER_OK)
    result = await plan_route(ctx, [START, VIA, END], TransportMode.cycling)

    assert result["estimated"] is False
    assert result["source"] == "BRouter trekking"
    assert brouter.calls[0].request.url.params["lonlats"] == "8.200000,53.100000|8.200000,53.000000|8.200000,52.900000"
    assert result["distance_km"] == 22.3
    assert (result["ascent_m"], result["descent_m"]) == (50, 45)
    # Segmente werden am Zwischenpunkt geteilt, Dauer anteilig nach Strecke
    first, second = result["segments"]
    assert (first["ascent_m"], first["descent_m"], second["ascent_m"], second["descent_m"]) == (50, 0, 0, 45)
    assert first["distance_km"] == second["distance_km"] == 11.1
    assert first["duration_hours"] == second["duration_hours"] == 0.6

    # Das hängende ORS-Profil wird beim nächsten Aufruf übersprungen → nur noch BRouter
    calls_before = respx.calls.call_count
    await plan_route(ctx, [START, END], TransportMode.cycling)
    assert respx.calls.call_count == calls_before + 1


@respx.mock
async def test_route_uses_ors_cycling_electric_when_brouter_fails(ctx) -> None:
    respx.post(ORS_URL.format(profile="cycling-regular")).respond(status_code=503)
    respx.get(BROUTER_URL).respond(status_code=500)
    respx.post(ORS_URL.format(profile="cycling-electric")).respond(json=ORS_OK)
    result = await plan_route(ctx, [START, END], TransportMode.cycling)
    assert result["source"] == "OpenRouteService cycling-electric"
    assert result["distance_km"] == 36.0


@respx.mock
async def test_route_retries_same_profile_after_sporadic_timeout(ctx) -> None:
    route = respx.post(ORS_URL.format(profile="cycling-regular")).mock(
        side_effect=[httpx.ReadTimeout("hängt"), httpx.Response(200, json=ORS_OK)]
    )
    result = await plan_route(ctx, [START, END], TransportMode.cycling)
    assert result["source"] == "OpenRouteService cycling-regular"
    assert route.call_count == 2
    assert json.loads(route.calls[0].request.content)["instructions"] is True  # sonst fehlen die segments
    assert routing._slow_until == {}  # ein einzelner Aussetzer sperrt das Profil nicht


@respx.mock
async def test_route_without_ors_key_uses_brouter(ctx) -> None:
    ctx.deps.settings = ctx.deps.settings.model_copy(update={"ors_api_key": ""})
    respx.get(BROUTER_URL).respond(json=BROUTER_OK)
    result = await plan_route(ctx, [START, END], TransportMode.cycling)
    assert result["source"] == "BRouter trekking"


@respx.mock
async def test_route_estimates_when_all_providers_fail(ctx) -> None:
    respx.post(url__startswith="https://api.openrouteservice.org/").mock(side_effect=httpx.ReadTimeout("langsam"))
    respx.get(BROUTER_URL).respond(status_code=503)
    result = await plan_route(ctx, [START, END], TransportMode.cycling)
    assert result["estimated"] is True
    assert "cycling-regular" in result["note"] and "BRouter" in result["note"]


def test_elevation_gain_ignores_noise() -> None:
    noisy = [[0, 0, e] for e in (100, 104, 99, 103, 100, 130, 126, 131, 100)]
    # Rauschen um 100 m und die Delle 130 → 126 zählen nicht, der Gipfel bei 131 m schon
    assert elevation_gain(noisy) == (31.0, 31.0)


async def test_route_estimate_for_canoe(ctx) -> None:
    result = await plan_route(ctx, [Coordinate(lat=53.0, lon=13.0), Coordinate(lat=53.1, lon=13.0)], TransportMode.canoe)
    assert result["estimated"] is True
    assert result["distance_km"] > 11


@respx.mock
async def test_weather_forecast(ctx) -> None:
    respx.get(FORECAST_URL).respond(
        json={
            "daily": {
                "time": ["2026-07-03"],
                "weather_code": [95],
                "temperature_2m_max": [27.0],
                "temperature_2m_min": [14.0],
                "precipitation_sum": [12.0],
                "precipitation_probability_max": [80],
                "wind_speed_10m_max": [20.0],
            }
        }
    )
    result = await get_weather(ctx, 47.0, 11.0, dt.date(2026, 7, 3), 1)
    assert result["kind"] == "forecast"
    assert result["days"][0]["conditions"] == "Gewitter"


@respx.mock
async def test_weather_climate_far_future(ctx) -> None:
    respx.get(ARCHIVE_URL).respond(
        json={"daily": {"temperature_2m_max": [20.0], "temperature_2m_min": [10.0], "precipitation_sum": [2.0]}}
    )
    result = await get_weather(ctx, 47.0, 11.0, dt.date(2026, 9, 1), 1)
    assert result["kind"] == "climate"
    assert result["days"][0]["temp_max_c"] == 20.0


async def test_weather_rejects_past(ctx) -> None:
    assert "error" in await get_weather(ctx, 47.0, 11.0, dt.date(2026, 6, 1), 1)


@respx.mock
async def test_accommodation_via_nominatim(ctx) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        query = request.url.params["q"]
        assert request.url.params["bounded"] == "1" and request.url.params["extratags"] == "1"
        if query == "[tourism=alpine_hut]":
            return httpx.Response(200, json=[
                {"osm_type": "way", "osm_id": 1, "name": "Ferne Hütte", "lat": "47.04", "lon": "11.0", "extratags": None},
                {"osm_type": "node", "osm_id": 2, "name": "Nahe Hütte", "lat": "47.01", "lon": "11.0",
                 "extratags": {"contact:website": "https://huette.example", "capacity": "40"}},
                {"osm_type": "node", "osm_id": 3, "name": "", "lat": "47.0", "lon": "11.005", "extratags": {}},
            ])  # fmt: skip
        # Selbstversorgerhütte taucht doppelt auf (beide Filter) → nur einmal im Ergebnis
        return httpx.Response(200, json=[{"osm_type": "node", "osm_id": 2, "name": "Nahe Hütte", "lat": "47.01", "lon": "11.0"}])

    route = respx.get(geocode_module.NOMINATIM_URL).mock(side_effect=respond)
    result = await find_accommodation(ctx, 47.0, 11.0, AccommodationType.hut, radius_km=5)

    assert route.call_count == 2  # alpine_hut + wilderness_hut
    assert result["source"] == "Nominatim"
    assert [r["name"] for r in result["results"]] == ["Nahe Hütte", "Ferne Hütte", "Unbenannt"]
    assert result["results"][0]["capacity"] == "40"


@respx.mock
async def test_bivouac_filters_shelter_type(ctx) -> None:
    shelters = [
        {"osm_type": "node", "osm_id": 1, "name": "Biwakschachtel", "lat": "47.0", "lon": "11.0", "extratags": {"shelter_type": "basic_hut"}},
        {"osm_type": "node", "osm_id": 2, "name": "Bushaltestelle", "lat": "47.0", "lon": "11.0", "extratags": {"shelter_type": "public_transport"}},
    ]  # fmt: skip
    respx.get(geocode_module.NOMINATIM_URL).mock(
        side_effect=lambda request: httpx.Response(200, json=shelters if "shelter" in request.url.params["q"] else [])
    )
    result = await find_accommodation(ctx, 47.0, 11.0, AccommodationType.bivouac)
    assert [r["name"] for r in result["results"]] == ["Biwakschachtel"]


@respx.mock
async def test_accommodation_overpass_fallback_and_filter(ctx) -> None:
    respx.get(geocode_module.NOMINATIM_URL).respond(status_code=503)
    urls = ctx.deps.settings.overpass_url_list
    respx.post(urls[0]).respond(status_code=504)
    respx.post(urls[1]).respond(
        json={
            "elements": [
                {"type": "node", "lat": 47.01, "lon": 11.0, "tags": {"name": "Nahe Hütte", "website": "https://x"}},
                {"type": "way", "center": {"lat": 47.0, "lon": 11.005}, "tags": {}},
                {"type": "node", "lat": 47.5, "lon": 11.0, "tags": {"name": "Zu weit weg"}},
            ]
        }
    )
    result = await find_accommodation(ctx, 47.0, 11.0, AccommodationType.hut, radius_km=5)
    names = [r["name"] for r in result["results"]]
    assert names == ["Nahe Hütte", "Unbenannt"]
    assert result["results"][0]["website"] == "https://x"
    assert result["source"] == "Overpass"


@respx.mock
async def test_accommodation_circuit_breaker(ctx) -> None:
    respx.get(geocode_module.NOMINATIM_URL).respond(status_code=503)
    routes = [respx.post(url).respond(status_code=504) for url in ctx.deps.settings.overpass_url_list]
    assert "error" in await find_accommodation(ctx, 47.0, 11.0, AccommodationType.hut)
    calls = sum(r.call_count for r in routes)
    # Zweite Anfrage (anderer Ort) wird ohne HTTP-Aufruf sofort abgelehnt
    result = await find_accommodation(ctx, 47.2, 11.2, AccommodationType.hut)
    assert "überlastet" in result["error"]
    assert sum(r.call_count for r in routes) == calls


def test_hiking_time_dav_formula() -> None:
    # 12 km → 3 h horizontal; 900 Hm auf → 3 h vertikal → 3 + 1.5 = 4.5 h
    assert estimate_hiking_time(12, 900, 0)["walking_hours"] == 4.5


def test_sun_times() -> None:
    result = sun_times(47.5, 11.1, dt.date(2026, 6, 21))
    assert result["daylight_hours"] > 15


def transit_place(name: str, tz: str | None = "Europe/Berlin", track: str | None = None) -> dict:
    return {"name": name, "lat": 48.0, "lon": 11.0, "tz": tz} | ({"track": track} if track else {})


def transit_leg(mode: str, origin: dict, destination: dict, start: str, end: str, **extra) -> dict:
    duration = (dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)).seconds
    return {"mode": mode, "from": origin, "to": destination, "startTime": start, "endTime": end, "duration": duration} | extra


# Zeiten in UTC wie bei der MOTIS-API; Ende Oktober gilt in Berlin noch Sommerzeit (UTC+2)
TRANSIT_OK = {
    "itineraries": [
        {
            "startTime": "2026-10-24T06:20:00Z",
            "endTime": "2026-10-24T08:28:00Z",
            "duration": 7680,
            "transfers": 0,
            "legs": [
                transit_leg("WALK", transit_place("START"), transit_place("München Hbf", track="28"),
                            "2026-10-24T06:20:00Z", "2026-10-24T06:28:00Z"),
                transit_leg("REGIONAL_RAIL", transit_place("München Hbf", track="28"),
                            transit_place("Garmisch-Partenkirchen"), "2026-10-24T06:28:00Z", "2026-10-24T07:52:00Z",
                            displayName="RB6", headsign="Mittenwald"),
                transit_leg("WALK", transit_place("Garmisch-Partenkirchen"), transit_place("END"),
                            "2026-10-24T07:52:00Z", "2026-10-24T07:53:00Z"),
            ],
        },
        {
            "startTime": "2026-10-24T05:12:00Z",
            "endTime": "2026-10-24T07:28:00Z",
            "duration": 8160,
            "transfers": 1,
            "legs": [
                transit_leg("REGIONAL_RAIL", transit_place("München Hbf", track="27"), transit_place("Weilheim"),
                            "2026-10-24T05:12:00Z", "2026-10-24T06:42:00Z", routeShortName="RB74"),
                transit_leg("WALK", transit_place("Weilheim"), transit_place("weilheim "),
                            "2026-10-24T06:42:00Z", "2026-10-24T06:44:00Z"),
                transit_leg("REGIONAL_RAIL", transit_place("Weilheim", track="Gl3"),
                            transit_place("Garmisch-Partenkirchen"), "2026-10-24T06:49:00Z", "2026-10-24T07:28:00Z",
                            displayName="RE61", cancelled=True),
            ],
        },
    ]
}  # fmt: skip

TRANSIT_ARGS = {
    "from_name": "München", "from_lat": 48.14, "from_lon": 11.56,
    "to_name": "Garmisch-Partenkirchen", "to_lat": 47.49, "to_lon": 11.1,
    "date": dt.date(2026, 10, 24),
}  # fmt: skip


@respx.mock
async def test_train_connections(ctx) -> None:
    route = respx.get(get_settings().transit_api_url).respond(json=TRANSIT_OK)
    result = await find_train_connections(ctx, **TRANSIT_ARGS, time=dt.time(10, 0), arrive_by=True)

    params = route.calls.last.request.url.params
    assert params["fromPlace"] == "48.14,11.56"
    assert params["time"] == "2026-10-24T10:00:00+02:00"
    assert params["arriveBy"] == "true"
    assert "COACH" not in params["transitModes"]
    assert result["source"].startswith("Fahrplandaten: Transitous")
    # chronologisch, Ortszeit statt UTC
    early, late = result["connections"]
    assert (early["departure"], early["arrival"], early["transfers"]) == ("2026-10-24T07:12", "2026-10-24T09:28", 1)
    assert early["legs"][0] == {
        "line": "RB74", "from_stop": "München Hbf", "to_stop": "Weilheim",
        "departure": "2026-10-24T07:12", "arrival": "2026-10-24T08:42", "track": "27",
    }  # fmt: skip
    assert early["legs"][1]["cancelled"] is True
    assert early["legs"][1]["track"] == "3"
    # Bahnsteigwechsel entfällt
    assert len(early["legs"]) == 2
    # Fußweg zum Bahnhof mit Abfahrtsort statt START; 1 min vom Bahnhof zum gleichnamigen Ziel entfällt
    assert late["duration_minutes"] == 128
    assert [leg["line"] for leg in late["legs"]] == ["Fußweg (8 min)", "RB6"]
    assert late["legs"][0]["from_stop"] == "München"
    assert late["legs"][1]["direction"] == "Mittenwald"


@respx.mock
async def test_train_connections_none_found(ctx) -> None:
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": []})
    result = await find_train_connections(ctx, **TRANSIT_ARGS)
    assert "Keine Verbindung München → Garmisch-Partenkirchen" in result["error"]


@respx.mock
async def test_train_connections_service_down(ctx) -> None:
    respx.get(get_settings().transit_api_url).respond(status_code=503)
    result = await find_train_connections(ctx, **TRANSIT_ARGS)
    assert "Transitous meldet HTTP 503" in result["error"]


async def test_train_connections_disabled_or_past(ctx) -> None:
    assert "Vergangenheit" in (await find_train_connections(ctx, **TRANSIT_ARGS | {"date": TODAY - dt.timedelta(days=1)}))["error"]
    ctx.deps.settings = ctx.deps.settings.model_copy(update={"transit_api_url": ""})
    assert "nicht konfiguriert" in (await find_train_connections(ctx, **TRANSIT_ARGS))["error"]


@respx.mock
async def test_train_connections_repeated_query_is_not_sent_again(ctx) -> None:
    route = respx.get(get_settings().transit_api_url).respond(json=TRANSIT_OK)
    first = await find_train_connections(ctx, **TRANSIT_ARGS)
    again = await find_train_connections(ctx, **TRANSIT_ARGS)
    assert route.call_count == 1
    assert again["connections"] == first["connections"]
    assert "bereits ausgeführt" in again["hint"]
    assert "hint" not in first


@respx.mock
async def test_train_connections_merge_variants_with_same_trains(ctx) -> None:
    direct, with_bus = TRANSIT_OK["itineraries"][1], dict(TRANSIT_OK["itineraries"][1])
    bus = transit_leg("BUS", transit_place("Stachus"), transit_place("München Hbf"),
                      "2026-10-24T05:00:00Z", "2026-10-24T05:08:00Z", displayName="100")  # fmt: skip
    with_bus |= {"legs": [bus, *direct["legs"]], "transfers": 2, "duration": direct["duration"] + 720}
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": [with_bus, *TRANSIT_OK["itineraries"]]})
    result = await find_train_connections(ctx, **TRANSIT_ARGS)
    assert [c["transfers"] for c in result["connections"]] == [1, 0]
