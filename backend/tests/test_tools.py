import datetime as dt
import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AccommodationType, TransportMode
from app.agent.tools import accommodation, named_route, routing
from app.agent.tools import geocode as geocode_module
from app.agent.tools.accommodation import find_accommodation
from app.agent.tools.common import haversine_km
from app.agent.tools.geocode import geocode
from app.agent.tools.routing import BROUTER_URL, ORS_URL, Coordinate, elevation_gain, estimate_hiking_time, plan_route
from app.agent.tools.sun import sun_times
from app.agent.tools.transit import (
    _line,
    bahn_search_url,
    find_train_connections,
    format_train_queries,
    next_timetable_change,
    timetable_change,
)
from app.agent.tools.weather import ARCHIVE_URL, FORECAST_URL, get_weather
from app.config import get_settings
from app.db import SessionLocal
from app.models import LongDistanceRoute

TODAY = dt.date(2026, 7, 1)


@pytest.fixture(autouse=True)
def reset_overpass_state(monkeypatch) -> None:
    monkeypatch.setattr(accommodation, "_cache", {})
    monkeypatch.setattr(accommodation, "_unavailable_until", 0.0)
    monkeypatch.setattr(routing, "_slow_until", {})
    monkeypatch.setattr(named_route, "_graphs", {})
    monkeypatch.setattr(named_route, "_catalog_graphs", {})
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
async def test_geocode_prefers_locality_over_municipality_area(ctx) -> None:
    """Gemeinde ohne Ortspunkt (Flächenmitte im Moor) und gleichnamige Ortschaft darin: die Ortschaft zuerst."""
    geocode_module._cache.clear()
    municipality = {
        "display_name": "Worpswede, Landkreis Osterholz", "name": "Worpswede", "lat": "53.2616751", "lon": "8.9758702",
        "category": "boundary", "type": "administrative", "place_rank": 16,
        "boundingbox": ["53.1844", "53.3420", "8.8510", "9.1210"],
    }  # fmt: skip
    locality = municipality | {
        "lat": "53.2230018", "lon": "8.9283482", "place_rank": 18, "boundingbox": ["53.2000", "53.2400", "8.9000", "8.9600"],
    }  # fmt: skip
    respx.get(geocode_module.NOMINATIM_URL).respond(json=[municipality, locality])
    result = await geocode(ctx, "Worpswede")
    assert [(r["lat"], r["lon"]) for r in result] == [(53.223002, 8.928348), (53.261675, 8.97587)]

    # Ortspunkt statt Ortschaftsgrenze (z. B. Bad Zwischenahn)
    geocode_module._cache.clear()
    respx.get(geocode_module.NOMINATIM_URL).respond(json=[municipality, locality | {"category": "place", "type": "town"}])
    assert (await geocode(ctx, "Worpswede"))[0]["lat"] == 53.223002

    # Gleichnamiger Ort außerhalb der Gemeinde (anderswo in Deutschland) ändert die Reihenfolge nicht
    geocode_module._cache.clear()
    elsewhere = locality | {"lat": "50.1", "lon": "10.1"}
    respx.get(geocode_module.NOMINATIM_URL).respond(json=[municipality, elsewhere])
    assert (await geocode(ctx, "Worpswede"))[0]["lat"] == 53.261675


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


# Fernweg als Bogen nach Norden: zwei Wege mit ~45 m Lücke zwischen 8.30 und der Fortsetzung
COAST_START, COAST_VIA, COAST_END = (
    Coordinate(lat=53.10, lon=8.20), Coordinate(lat=53.15, lon=8.30), Coordinate(lat=53.10, lon=8.40)
)
COAST_RELATIONS = {"elements": [
    {"type": "relation", "id": 1, "tags": {"name": "D2 Testküsten-Radweg - Abschnitt Nord", "network": "ncn"}},
    {"type": "relation", "id": 2, "tags": {"name": "Nord-Testküsten-Weg", "network": "rcn"}},
    {"type": "relation", "id": 3, "tags": {"name": "Moortour", "network": "lcn"}},
]}
COAST_WAYS = {"elements": [
    {"type": "way", "id": 10, "nodes": [1, 2, 3],
     "geometry": [{"lat": 53.10, "lon": 8.20}, {"lat": 53.15, "lon": 8.25}, {"lat": 53.15, "lon": 8.30}]},
    {"type": "way", "id": 11, "nodes": [4, 5, 6],
     "geometry": [{"lat": 53.1504, "lon": 8.30}, {"lat": 53.15, "lon": 8.35}, {"lat": 53.10, "lon": 8.40}]},
]}  # fmt: skip


def overpass_coast(request: httpx.Request) -> httpx.Response:
    query = request.content.decode()
    return httpx.Response(200, json=COAST_RELATIONS if "rel%28bw%29" in query else COAST_WAYS)


def brouter_echo(request: httpx.Request) -> httpx.Response:
    """BRouter-Antwort mit geraden Linien zwischen den übergebenen Punkten."""
    points = [[float(v) for v in p.split(",")] for p in request.url.params["lonlats"].split("|")]
    meters = sum(haversine_km(a[1], a[0], b[1], b[0]) for a, b in zip(points, points[1:])) * 1000
    return httpx.Response(200, json={"features": [{
        "geometry": {"coordinates": [[lon, lat, 10.0] for lon, lat in points]},
        "properties": {"track-length": str(meters), "total-time": str(meters / 5)},
    }]})  # fmt: skip


def test_named_route_matching() -> None:
    assert named_route.matches("Ostseeküsten-Radweg", {"name": "D2 Ostseeküsten-Radweg - Abschnitt SH-Ost"})
    assert named_route.matches("ostseekuesten radweg", {"name": "EV10 Ostseeküsten-Radweg"})
    assert named_route.matches("EV10", {"name": "EuroVelo", "ref": "EV10"})
    # Kein beliebiger Teilstring: "Ostseeradweg" ist nicht der Nord-Ostsee-Radweg
    assert not named_route.matches("Ostseeradweg", {"name": "Nord-Ostsee-Radweg"})


@respx.mock
async def test_route_follows_named_route(ctx) -> None:
    overpass = respx.post(get_settings().overpass_url_list[0]).mock(side_effect=overpass_coast)
    brouter = respx.get(BROUTER_URL).mock(side_effect=brouter_echo)
    result = await plan_route(ctx, [COAST_START, COAST_VIA, COAST_END], TransportMode.cycling, "Testküsten-Radweg")

    assert overpass.call_count == 2  # Relationen nahe den Wegpunkten, dann deren Wege
    assert "rel(id:1)" in httpx.QueryParams(overpass.calls[1].request.content.decode())["data"]
    # BRouter zuerst (viele Teilanfragen); der Bogen nach Norden kommt als Stützpunkt mit
    assert "8.250000,53.150000" in brouter.calls[0].request.url.params["lonlats"]
    assert result["source"].startswith("BRouter trekking entlang D2 Testküsten-Radweg")
    assert "OpenStreetMap" in result["source"]
    assert result["named_route"]["routes"] == ["D2 Testküsten-Radweg"]
    assert result["named_route"]["off_route_legs"] == []
    # Lücke zwischen den Wegen überbrückt: beide Teilstücke liegen auf dem Fernweg
    assert result["named_route"]["on_route_km"] == pytest.approx(19.8, abs=0.5)
    assert len(result["segments"]) == 2
    assert result["distance_km"] == pytest.approx(19.8, abs=0.5)

    # Weitere Aufrufe derselben Tour nutzen den gespeicherten Graphen
    await plan_route(ctx, [COAST_START, COAST_END], TransportMode.cycling, "Testküsten-Radweg")
    assert overpass.call_count == 2


async def add_test_route() -> None:
    """Katalogeintrag "Testweg" als leichter Bogen nach Osten zwischen START und END."""
    async with SessionLocal() as session:
        session.add(LongDistanceRoute(
            name="Testweg", aliases=["Testradweg", "T1"], route_type="bicycle", relation_ids=[99],
            ways=[[[1, 2, 3], [[53.10, 8.20], [53.00, 8.25], [52.90, 8.20]]]], length_km=23.2,
        ))  # fmt: skip
        await session.commit()


def test_catalog_matching() -> None:
    names = ["Ostseeküsten-Radweg", "Ostseeradweg", "D2", "EV10"]
    assert named_route.catalog_matches("Ostseeradweg", names)
    assert named_route.catalog_matches("EV10 Ostseeküsten-Radweg", names)  # Kürzel davor
    assert named_route.catalog_matches("Ostseeküsten", names)  # Namensanfang
    assert not named_route.catalog_matches("D", names)
    assert not named_route.catalog_matches("D1", ["Elberadweg", "D10"])  # Kürzel nur exakt


@respx.mock
async def test_named_route_from_catalog_without_overpass(ctx) -> None:
    await add_test_route()
    overpass = respx.post(url__startswith="https://overpass").respond(status_code=503)
    brouter = respx.get(BROUTER_URL).mock(side_effect=brouter_echo)
    result = await plan_route(ctx, [START, END], TransportMode.cycling, "Testradweg")

    assert overpass.call_count == 0
    assert "8.250000,53.000000" in brouter.calls[0].request.url.params["lonlats"]
    assert result["named_route"] == {"routes": ["Testweg"], "on_route_km": 23.2, "off_route_legs": []}
    assert result["source"] == "BRouter trekking entlang Testweg (Verlauf: © OpenStreetMap-Mitwirkende)"
    # Wandern sucht nur Wanderwege im Katalog → live über Overpass (hier nicht erreichbar)
    respx.post(url__startswith="https://api.openrouteservice.org/").respond(status_code=503)
    result = await plan_route(ctx, [START, END], TransportMode.hiking, "Testradweg")
    assert "named_route" not in result and overpass.call_count > 0


@respx.mock
async def test_named_route_unknown_name_lists_routes_nearby(ctx) -> None:
    respx.post(get_settings().overpass_url_list[0]).mock(side_effect=overpass_coast)
    result = await plan_route(ctx, [COAST_START, COAST_END], TransportMode.cycling, "Küstenradweg")
    # Fernwege zuerst, Abschnittsangaben gekürzt
    assert "„D2 Testküsten-Radweg“, „Nord-Testküsten-Weg“, „Moortour“" in result["error"]


@respx.mock
async def test_named_route_far_waypoint_is_routed_normally(ctx) -> None:
    respx.post(get_settings().overpass_url_list[0]).mock(side_effect=overpass_coast)
    respx.get(BROUTER_URL).mock(side_effect=brouter_echo)
    inland = Coordinate(lat=53.00, lon=8.40)  # ~11 km vom Fernweg
    result = await plan_route(ctx, [COAST_START, COAST_END, inland], TransportMode.cycling, "Testküsten-Radweg")
    assert result["named_route"]["off_route_legs"] == [2]
    assert "Teilstück(e) 2 folgen nicht dem Fernweg" in result["note"]


@respx.mock
async def test_named_route_without_overpass_routes_normally(ctx) -> None:
    ctx.deps.settings = ctx.deps.settings.model_copy(update={"ors_api_key": ""})
    respx.post(url__startswith="https://overpass").respond(status_code=400)
    respx.get(BROUTER_URL).respond(json=BROUTER_OK)
    result = await plan_route(ctx, [START, END], TransportMode.cycling, "Testküsten-Radweg")
    assert result["source"] == "BRouter trekking"
    assert "named_route" not in result
    assert "nicht abrufbar" in result["note"]


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


def hourly_rs30(hours: range) -> list[dict]:
    """Stündliche RS30-Fahrten Oldenburg → Bremen (UTC), alle mit gleicher Dauer und gleichem Liniennamen."""
    return [
        {
            "startTime": f"2026-10-24T{h:02d}:02:00Z", "endTime": f"2026-10-24T{h:02d}:29:00Z", "duration": 1620,
            "transfers": 0,
            "legs": [transit_leg("METRO", transit_place("Oldenburg(Oldb) Hbf"), transit_place("Bremen Hbf"),
                                 f"2026-10-24T{h:02d}:02:00Z", f"2026-10-24T{h:02d}:29:00Z", displayName="RS30")],
        }
        for h in hours
    ]  # fmt: skip


@respx.mock
async def test_train_connections_closest_to_requested_time(ctx) -> None:
    """Fahrten derselben Linie bleiben getrennt; bei arrive_by die spätesten Ankünfte, sonst die frühesten Abfahrten."""
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": hourly_rs30(range(3, 9))})
    arrive = await find_train_connections(ctx, **TRANSIT_ARGS, time=dt.time(11, 0), arrive_by=True)
    assert [c["arrival"][11:] for c in arrive["connections"]] == ["08:29", "09:29", "10:29"]
    depart = await find_train_connections(ctx, **TRANSIT_ARGS, time=dt.time(5, 0))
    assert [c["departure"][11:] for c in depart["connections"]] == ["05:02", "06:02", "07:02"]


def test_local_lines_name_their_mode() -> None:
    assert _line({"mode": "BUS", "displayName": "24"}) == "Bus 24"
    assert _line({"mode": "TRAM", "displayName": "1"}) == "Tram 1"
    assert _line({"mode": "BUS", "routeShortName": "303E"}) == "Bus 303E"
    assert _line({"mode": "SUBWAY", "displayName": "U3"}) == "U3"
    assert _line({"mode": "TRAM", "displayName": "STR 4"}) == "STR 4"
    assert _line({"mode": "LONG_DISTANCE", "displayName": "IC 2035"}) == "IC 2035"
    assert _line({"mode": "BUS"}) == "Bus"


@respx.mock
async def test_train_connections_merge_variants_with_same_trains(ctx) -> None:
    direct, with_bus = TRANSIT_OK["itineraries"][1], dict(TRANSIT_OK["itineraries"][1])
    bus = transit_leg("BUS", transit_place("Stachus"), transit_place("München Hbf"),
                      "2026-10-24T05:00:00Z", "2026-10-24T05:08:00Z", displayName="100")  # fmt: skip
    with_bus |= {"legs": [bus, *direct["legs"]], "transfers": 2, "duration": direct["duration"] + 720}
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": [with_bus, *TRANSIT_OK["itineraries"]]})
    result = await find_train_connections(ctx, **TRANSIT_ARGS)
    assert [c["transfers"] for c in result["connections"]] == [1, 0]


def test_timetable_change_is_second_sunday_in_december() -> None:
    assert [timetable_change(year) for year in (2025, 2026, 2027)] == [
        dt.date(2025, 12, 14), dt.date(2026, 12, 13), dt.date(2027, 12, 12),
    ]  # fmt: skip
    assert next_timetable_change(dt.date(2026, 12, 12)) == dt.date(2026, 12, 13)
    assert next_timetable_change(dt.date(2026, 12, 13)) == dt.date(2027, 12, 12)


def test_bahn_search_url() -> None:
    moment = dt.datetime(2026, 12, 20, 12, 0)
    url = bahn_search_url("Oldenburg", 53.1435, 8.2146, "Hannover", 52.3767, 9.741, moment, arrive_by=True)
    assert url.startswith("https://www.bahn.de/buchung/fahrplan/suche#sts=true&so=Oldenburg&zo=Hannover&")
    assert "soid=A%3D2%40O%3DOldenburg%40X%3D8214600%40Y%3D53143500%40" in url
    assert "hd=2026-12-20T12%3A00%3A00&hza=A" in url


def one_trip(start: str, end: str, mode: str = "REGIONAL_RAIL") -> dict:
    """Eine Verbindung ohne Umstieg (Zeiten in UTC)."""
    leg = transit_leg(mode, transit_place("München Hbf"), transit_place("Garmisch-Partenkirchen"), start, end,
                      displayName="RB6")  # fmt: skip
    return {"startTime": start, "endTime": end, "duration": leg["duration"], "transfers": 0, "legs": [leg]}


AFTER_CHANGE = TRANSIT_ARGS | {"date": dt.date(2026, 12, 20), "time": dt.time(9, 0)}


@respx.mock
async def test_train_connections_missing_timetable_after_change(ctx) -> None:
    """Nach dem Fahrplanwechsel liefert Transitous nur Verbindungen weit ab der Wunschzeit: Daten fehlen noch."""
    trip = one_trip("2026-12-20T19:31:00Z", "2026-12-20T21:00:00Z")
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": [trip]})
    result = await find_train_connections(ctx, **AFTER_CHANGE)

    assert result["connections"] == []
    assert "Fahrplanwechsel am 13.12.2026" in result["error"]
    assert "hd=2026-12-20T09%3A00%3A00&hza=D" in result["bahn_url"]
    shown = format_train_queries(list(ctx.deps.train_queries.values()))
    assert f"[Verbindungen bei bahn.de]({result['bahn_url']})" in shown


@respx.mock
async def test_train_connections_after_change_with_data(ctx) -> None:
    trip = one_trip("2026-12-20T08:28:00Z", "2026-12-20T09:52:00Z")
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": [trip]})
    result = await find_train_connections(ctx, **AFTER_CHANGE)

    assert [c["departure"] for c in result["connections"]] == ["2026-12-20T09:28"]
    assert "unvollständig" in result["note"]
    shown = format_train_queries(list(ctx.deps.train_queries.values()))
    assert "09:28 RB6" in shown
    assert f"[bei bahn.de prüfen]({result['bahn_url']})" in shown


@respx.mock
async def test_train_connections_before_change_far_from_time_is_kept(ctx) -> None:
    """Vor dem Fahrplanwechsel sind große Lücken echt (z. B. nachts, auf dem Land) und kein Datenproblem."""
    trip = one_trip("2026-10-24T17:00:00Z", "2026-10-24T18:00:00Z")
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": [trip]})
    result = await find_train_connections(ctx, **TRANSIT_ARGS)
    assert [c["departure"] for c in result["connections"]] == ["2026-10-24T19:00"]
    assert "bahn_url" not in result


@respx.mock
async def test_train_connections_none_found_links_bahn(ctx) -> None:
    respx.get(get_settings().transit_api_url).respond(json={"itineraries": []})
    result = await find_train_connections(ctx, **TRANSIT_ARGS)
    assert "https://www.bahn.de/buchung/fahrplan/suche#" in result["error"]
    assert not ctx.deps.train_queries
