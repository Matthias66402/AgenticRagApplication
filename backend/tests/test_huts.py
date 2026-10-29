from types import SimpleNamespace

import httpx
import pytest
import respx
from pydantic_ai import ModelResponse, ModelRetry, ToolCallPart

from app.agent.plan_schema import AccommodationType, AdventurePlan
from app.agent.planner import check_hut_tour
from app.agent.tools import geocode as geocode_module
from app.agent.tools.common import haversine_km
from app.agent.tools.huts import choose_chain, overnight_likely, plan_hut_tour, problem_legs, valley_drops
from app.agent.tools.routing import BROUTER_URL, Coordinate
from tests.test_agent import make_plan
from tests.test_tools import brouter_echo, ctx, reset_overpass_state  # noqa: F401 (Fixtures)

START = Coordinate(lat=47.0, lon=11.0)
END = Coordinate(lat=47.0, lon=11.4)  # ca. 30 km östlich


def osm_hut(osm_id: int, name: str, lat: float, lon: float, **tags: str) -> dict:
    return {"osm_type": "node", "osm_id": osm_id, "name": name, "lat": str(lat), "lon": str(lon), "extratags": tags}


HUTS = [
    osm_hut(1, "Westhütte", 47.0, 11.13, operator="DAV Sektion Test", ele="1900"),
    osm_hut(2, "Hochalm", 47.035, 11.14),  # seitlich und ohne Hinweis auf Übernachtung
    osm_hut(3, "Osthaus", 47.0, 11.27, capacity="40", ele="2000"),
    osm_hut(4, "Osthaus Nebengebäude", 47.001, 11.271),  # derselbe Standort
    osm_hut(5, "Rückwärtshütte", 47.0, 10.85),  # 11 km hinter dem Start
    osm_hut(6, "Umweghütte", 47.0, 11.2, operator="ÖAV"),
]


def nominatim_huts(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=HUTS if request.url.params["q"] == "[tourism=alpine_hut]" else [])


def corridor_hut(name: str, along_km: float, offset_km: float = 0.0, likely: bool = True) -> dict:
    lon = START.lon + along_km / (111.32 * 0.682)
    lat = START.lat + offset_km / 110.57
    return {"name": name, "lat": lat, "lon": lon, "km_from_start": along_km, "offset_km": offset_km, "overnight_likely": likely}


def test_overnight_likely() -> None:
    assert overnight_likely({"name": "Kemptner Hütte"})
    assert overnight_likely({"name": "Sonnalm", "operator": "DAV Sektion Allgäu"})
    assert overnight_likely({"name": "Gasthof", "capacity": "30"})
    assert not overnight_likely({"name": "Hocheggalm", "operator": "Bergbahnen Seefeld"})
    assert not overnight_likely({"name": "Alte Meilerhütte (Winterraum)"})


def test_choose_chain_spaces_stages_evenly_and_prefers_overnight() -> None:
    huts = [
        corridor_hut("Zu früh", 3),
        corridor_hut("Erste", 10),
        corridor_hut("Alm", 11, likely=False),
        corridor_hut("Zweite", 20),
        corridor_hut("Seitlich", 21, offset_km=7),
    ]
    chain = choose_chain(huts, START, END, nights=2)
    assert [h["name"] for h in chain or []] == ["Erste", "Zweite"]
    # Zu viele Nächte für die Hütten im Korridor
    assert choose_chain(huts[:1], START, END, nights=2) is None


@respx.mock
async def test_plan_hut_tour_routes_over_huts_on_the_way(ctx) -> None:  # noqa: F811
    respx.get(geocode_module.NOMINATIM_URL).mock(side_effect=nominatim_huts)
    respx.post(url__startswith="https://api.openrouteservice.org/").respond(status_code=503)
    respx.get(BROUTER_URL).mock(side_effect=brouter_echo)

    result = await plan_hut_tour(ctx, START, END, nights=2)

    assert [h["name"] for h in result["huts"]] == ["Westhütte", "Osthaus"]
    assert result["huts"][1]["capacity"] == "40"
    route = result["route"]
    assert route["route_id"] in ctx.deps.routes
    assert len(route["segments"]) == 3 and all(s["lowest_m"] == 10 for s in route["segments"])
    others = [h["name"] for h in result["other_huts"]]
    assert "Osthaus Nebengebäude" not in others and "Rückwärtshütte" not in others
    assert {"Hochalm", "Umweghütte"} <= set(others)


@respx.mock
async def test_plan_hut_tour_replaces_hut_with_long_detour(ctx) -> None:  # noqa: F811
    """Führt der Weg zu einer Hütte weit außen herum, liegt sie nicht auf dem Weg und wird ersetzt."""
    respx.get(geocode_module.NOMINATIM_URL).mock(side_effect=nominatim_huts)
    respx.post(url__startswith="https://api.openrouteservice.org/").respond(status_code=503)

    def detour_to_osthaus(request: httpx.Request) -> httpx.Response:
        points = [[float(v) for v in p.split(",")] for p in request.url.params["lonlats"].split("|")]
        if [11.27, 47.0] in points:
            index = points.index([11.27, 47.0])
            points.insert(index, [11.2, 47.15])  # Umweg über einen Punkt 17 km nördlich
        lonlats = "|".join(f"{lon},{lat}" for lon, lat in points)
        return brouter_echo(httpx.Request("GET", BROUTER_URL, params={"lonlats": lonlats}))

    respx.get(BROUTER_URL).mock(side_effect=detour_to_osthaus)
    result = await plan_hut_tour(ctx, START, END, nights=2)

    assert "Osthaus" not in [h["name"] for h in result["huts"]]
    assert max(s["distance_km"] for s in result["route"]["segments"]) < 26


def test_deep_valley_between_huts_is_a_problem_leg() -> None:
    """Zwischen zwei Hütten zurück ins Tal (Wank-Haus → Meilerhütte über Garmisch) taugt nicht als Etappe."""
    waypoints = [START, Coordinate(lat=47.0, lon=11.1), Coordinate(lat=47.0, lon=11.2), END]
    geometry = [
        [47.0, 11.0, 700.0], [47.0, 11.1, 1800.0],  # Aufstieg zur ersten Hütte
        [47.0, 11.15, 750.0], [47.0, 11.2, 2300.0],  # ins Tal und wieder hinauf
        [47.0, 11.3, 1700.0], [47.0, 11.4, 1500.0],
    ]
    drops = valley_drops(geometry, waypoints)
    assert drops == [0.0, 1050.0, 0.0]
    segments = [{"distance_km": 8.0}, {"distance_km": 10.0}, {"distance_km": 15.0}]
    assert problem_legs(waypoints, {"segments": segments}, drops) == [1]
    # Schätzung ohne Wegverlauf: keine Aussage
    assert problem_legs(waypoints, {"segments": segments, "estimated": True}, drops) == []


async def test_plan_hut_tour_rejects_round_trip(ctx) -> None:  # noqa: F811
    assert "error" in await plan_hut_tour(ctx, START, START, nights=1)


def test_hut_tour_must_use_plan_hut_tour() -> None:
    plan = AdventurePlan.model_validate(
        make_plan(days=2, accommodation="hut") | {"destination": {"name": "Innsbruck", "lat": 47.27, "lon": 11.39}}
    )
    assert haversine_km(plan.start.lat, plan.start.lon, plan.destination.lat, plan.destination.lon) > 3
    with pytest.raises(ModelRetry, match="plan_hut_tour"):
        check_hut_tour(SimpleNamespace(messages=[]), plan)

    used = [ModelResponse(parts=[ToolCallPart(tool_name="plan_hut_tour", args={})])]
    check_hut_tour(SimpleNamespace(messages=used), plan)
    # Rundtour und Hotel-Übernachtung brauchen das Tool nicht
    check_hut_tour(SimpleNamespace(messages=[]), AdventurePlan.model_validate(make_plan(days=2, accommodation="hut")))
    check_hut_tour(SimpleNamespace(messages=[]), plan.model_copy(update={"accommodation_type": AccommodationType.hotel}))
