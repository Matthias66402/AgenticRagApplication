import datetime as dt
from types import SimpleNamespace

import httpx
import pytest
import respx

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AccommodationType, TransportMode
from app.agent.tools import accommodation
from app.agent.tools import geocode as geocode_module
from app.agent.tools.accommodation import find_accommodation
from app.agent.tools.geocode import geocode
from app.agent.tools.routing import ORS_URL, Coordinate, estimate_hiking_time, plan_route
from app.agent.tools.sun import sun_times
from app.agent.tools.weather import ARCHIVE_URL, FORECAST_URL, get_weather
from app.config import get_settings
from app.db import SessionLocal

TODAY = dt.date(2026, 7, 1)


@pytest.fixture(autouse=True)
def reset_overpass_state(monkeypatch) -> None:
    monkeypatch.setattr(accommodation, "_cache", {})
    monkeypatch.setattr(accommodation, "_unavailable_until", 0.0)


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
    assert ctx.deps.routes[result["route_id"]] == [[47.0, 11.0], [47.1, 11.1]]
    assert "geometry" not in result


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
async def test_accommodation_fallback_and_filter(ctx) -> None:
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


@respx.mock
async def test_accommodation_circuit_breaker(ctx) -> None:
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
