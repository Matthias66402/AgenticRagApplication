"""Unterkunftssuche auf OpenStreetMap-Daten.

Primär über die Nominatim-Kategoriesuche (schnell, mit Kontaktdaten über `extratags`);
die öffentlichen Overpass-Server sind oft überlastet und dienen nur noch als Ersatz.
"""

import asyncio
import math
import time
from typing import Any

import httpx
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AccommodationType
from app.agent.tools.common import haversine_km, http_error_message, tool_error
from app.agent.tools.geocode import nominatim_search

# (OSM-Key, OSM-Wert, zusätzliche Tag-Bedingungen)
OsmFilter = tuple[str, str, dict[str, str]]

OSM_FILTERS: dict[AccommodationType, list[OsmFilter]] = {
    AccommodationType.hut: [("tourism", "alpine_hut", {}), ("tourism", "wilderness_hut", {})],
    AccommodationType.camping: [("tourism", "camp_site", {})],
    AccommodationType.hotel: [("tourism", "hotel", {}), ("tourism", "guest_house", {})],
    AccommodationType.hostel: [("tourism", "hostel", {})],
    AccommodationType.bivouac: [("tourism", "wilderness_hut", {}), ("amenity", "shelter", {"shelter_type": "basic_hut"})],
}

# Ausgewählte OSM-Tags → Feldnamen im Tool-Ergebnis
DETAIL_TAGS = [
    ("website", "website"), ("contact:website", "website"), ("phone", "phone"), ("contact:phone", "phone"),
    ("opening_hours", "opening_hours"), ("beds", "beds"), ("capacity", "capacity"),
    ("ele", "elevation_m"), ("operator", "operator"), ("fee", "fee"),
]  # fmt: skip

MAX_RESULTS = 8
NOMINATIM_LIMIT = 40  # Nominatim sortiert nicht nach Entfernung → großzügig abfragen, selbst sortieren

# Overpass (Ersatzquelle)
RETRYABLE_STATUS = {429, 502, 503, 504}
REQUEST_TIMEOUT = 25
COOLDOWN_SECONDS = 120

# Öffentliche Overpass-Server erlauben nur wenige parallele Abfragen pro IP → serialisieren und cachen
_overpass_lock = asyncio.Lock()
_cache: dict[str, dict[str, Any]] = {}
_unavailable_until = 0.0


class OverpassUnavailable(httpx.HTTPError):
    pass


def _bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """(Süd, West, Nord, Ost)"""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _matches(tags: dict[str, str], conditions: dict[str, str]) -> bool:
    return all(tags.get(k) == v for k, v in conditions.items())


def _entry(
    name: str | None, el_lat: float, el_lon: float, tags: dict[str, str], center: tuple[float, float],
    accommodation_type: AccommodationType,
) -> dict[str, Any]:  # fmt: skip
    entry: dict[str, Any] = {
        "name": name or tags.get("name") or "Unbenannt",
        "type": accommodation_type.value,
        "lat": round(el_lat, 6),
        "lon": round(el_lon, 6),
        "distance_km": round(haversine_km(center[0], center[1], el_lat, el_lon), 1),
    }
    for key, out in DETAIL_TAGS:
        if key in tags and out not in entry:
            entry[out] = tags[key]
    return entry


async def search_nominatim(
    ctx: RunContext[AgentDeps], accommodation_type: AccommodationType, lat: float, lon: float, radius_km: float
) -> list[dict[str, Any]]:
    south, west, north, east = _bbox(lat, lon, radius_km)
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for key, value, conditions in OSM_FILTERS[accommodation_type]:
        items = await nominatim_search(ctx, {
            "q": f"[{key}={value}]",
            "viewbox": f"{west:.5f},{south:.5f},{east:.5f},{north:.5f}",
            "bounded": 1,
            "limit": NOMINATIM_LIMIT,
            "extratags": 1,
        })  # fmt: skip
        for item in items:
            tags = item.get("extratags") or {}
            if not _matches(tags, conditions):
                continue
            entry = _entry(item.get("name"), float(item["lat"]), float(item["lon"]), tags, (lat, lon), accommodation_type)
            # Erster Treffer gewinnt (behält seine Detail-Tags, falls ein Objekt mehrfach auftaucht)
            found.setdefault((item.get("osm_type", ""), item.get("osm_id", 0)), entry)
    return list(found.values())


def build_query(accommodation_type: AccommodationType, lat: float, lon: float, radius_km: float) -> str:
    # Bounding Box statt around: auf den öffentlichen Servern schneller; die Distanz wird danach gefiltert
    south, west, north, east = _bbox(lat, lon, radius_km)
    selectors = "\n".join(
        f'  nwr["{key}"="{value}"]' + "".join(f'["{k}"="{v}"]' for k, v in conditions.items()) + ";"
        for key, value, conditions in OSM_FILTERS[accommodation_type]
    )
    return f"[out:json][timeout:25][bbox:{south:.5f},{west:.5f},{north:.5f},{east:.5f}];\n(\n{selectors}\n);\nout center tags 80;"


async def query_overpass(
    ctx: RunContext[AgentDeps], query: str, timeout: float = REQUEST_TIMEOUT, cache: bool = True
) -> dict[str, Any]:
    """Probiert die konfigurierten Overpass-Instanzen der Reihe nach durch.

    `cache=False` für große Antworten, die der Aufrufer selbst (aufbereitet) zwischenspeichert.
    """
    global _unavailable_until
    async with _overpass_lock:
        if query in _cache:
            return _cache[query]
        if time.monotonic() < _unavailable_until:
            raise OverpassUnavailable("Overpass ist gerade überlastet (vor Kurzem keine Antwort).")
        last_error: httpx.HTTPError | None = None
        for url in ctx.deps.settings.overpass_url_list:
            try:
                response = await ctx.deps.http.post(
                    url,
                    data={"data": query},
                    headers={"User-Agent": ctx.deps.settings.nominatim_user_agent},
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                if cache:
                    _cache[query] = data
                return data
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in RETRYABLE_STATUS:
                    break
            except httpx.HTTPError as exc:
                last_error = exc
        if last_error is None:
            raise httpx.HTTPError("Keine Overpass-URL konfiguriert")
        if isinstance(last_error, httpx.TimeoutException) or (
            isinstance(last_error, httpx.HTTPStatusError) and last_error.response.status_code in RETRYABLE_STATUS
        ):
            # Circuit Breaker: weitere Anfragen eine Weile sofort ablehnen, statt erneut lange zu warten
            _unavailable_until = time.monotonic() + COOLDOWN_SECONDS
        raise last_error


async def search_overpass(
    ctx: RunContext[AgentDeps], accommodation_type: AccommodationType, lat: float, lon: float, radius_km: float
) -> list[dict[str, Any]]:
    data = await query_overpass(ctx, build_query(accommodation_type, lat, lon, radius_km))
    found = []
    for element in data.get("elements", []):
        el_lat = element.get("lat") or element.get("center", {}).get("lat")
        el_lon = element.get("lon") or element.get("center", {}).get("lon")
        if el_lat is not None and el_lon is not None:
            found.append(_entry(None, el_lat, el_lon, element.get("tags", {}), (lat, lon), accommodation_type))
    return found


async def find_accommodation(
    ctx: RunContext[AgentDeps],
    lat: float,
    lon: float,
    accommodation_type: AccommodationType,
    radius_km: float = 5.0,
) -> dict[str, Any]:
    """Sucht Unterkünfte in der Nähe eines Punktes (OpenStreetMap).

    Typischerweise nahe dem Ende einer Tagesetappe aufrufen. Findet sich nichts,
    den Radius vergrößern oder einen anderen Etappenort wählen.

    Args:
        lat: Breitengrad des Suchmittelpunkts.
        lon: Längengrad des Suchmittelpunkts.
        accommodation_type: hut, camping, hotel, hostel oder bivouac.
        radius_km: Suchradius in km (0.5-30).
    """
    radius_km = max(0.5, min(radius_km, 30))
    try:
        found, source = await search_nominatim(ctx, accommodation_type, lat, lon, radius_km), "Nominatim"
    except httpx.HTTPError as nominatim_exc:
        try:
            found, source = await search_overpass(ctx, accommodation_type, lat, lon, radius_km), "Overpass"
        except httpx.HTTPError as overpass_exc:
            return tool_error(
                f"{http_error_message('Nominatim', nominatim_exc)} Ersatz: {http_error_message('Overpass API', overpass_exc)}"
            )

    # Ecken der Bounding Box abschneiden; benannte Einträge zuerst, dann nach Entfernung
    found = [e for e in found if e["distance_km"] <= radius_km]
    found.sort(key=lambda e: (e["name"] == "Unbenannt", e["distance_km"]))
    if not found:
        return {
            "results": [],
            "source": source,
            "note": f"Keine Unterkünfte vom Typ {accommodation_type.value} im Umkreis von {radius_km:g} km.",
        }
    return {"results": found[:MAX_RESULTS], "total_found": len(found), "source": source}
