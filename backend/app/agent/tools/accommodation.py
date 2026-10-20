import asyncio
import math
import time
from typing import Any

import httpx
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AccommodationType
from app.agent.tools.common import haversine_km, http_error_message, tool_error

# Öffentliche Instanzen sind zeitweise überlastet (HTTP 429/502/504) → der Reihe nach probieren
RETRYABLE_STATUS = {429, 502, 503, 504}

OSM_FILTERS: dict[AccommodationType, list[str]] = {
    AccommodationType.hut: ['["tourism"="alpine_hut"]', '["tourism"="wilderness_hut"]'],
    AccommodationType.camping: ['["tourism"="camp_site"]'],
    AccommodationType.hotel: ['["tourism"="hotel"]', '["tourism"="guest_house"]'],
    AccommodationType.hostel: ['["tourism"="hostel"]'],
    AccommodationType.bivouac: ['["tourism"="wilderness_hut"]', '["amenity"="shelter"]["shelter_type"="basic_hut"]'],
}

MAX_RESULTS = 8
REQUEST_TIMEOUT = 25
COOLDOWN_SECONDS = 120

# Öffentliche Overpass-Server erlauben nur wenige parallele Abfragen pro IP → serialisieren und cachen
_overpass_lock = asyncio.Lock()
_cache: dict[str, dict[str, Any]] = {}
_unavailable_until = 0.0


class OverpassUnavailable(httpx.HTTPError):
    pass


def build_query(accommodation_type: AccommodationType, lat: float, lon: float, radius_km: float) -> str:
    # Bounding Box statt around: auf den öffentlichen Servern schneller; die Distanz wird danach gefiltert
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    bbox = f"{lat - dlat:.5f},{lon - dlon:.5f},{lat + dlat:.5f},{lon + dlon:.5f}"
    selectors = "\n".join(f"  nwr{f};" for f in OSM_FILTERS[accommodation_type])
    return f"[out:json][timeout:25][bbox:{bbox}];\n(\n{selectors}\n);\nout center tags 80;"


async def query_overpass(ctx: RunContext[AgentDeps], query: str) -> dict[str, Any]:
    """Probiert die konfigurierten Overpass-Instanzen der Reihe nach durch."""
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
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                _cache[query] = response.json()
                return _cache[query]
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


async def find_accommodation(
    ctx: RunContext[AgentDeps],
    lat: float,
    lon: float,
    accommodation_type: AccommodationType,
    radius_km: float = 5.0,
) -> dict[str, Any]:
    """Sucht Unterkünfte in der Nähe eines Punktes (OpenStreetMap/Overpass).

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
        data = await query_overpass(ctx, build_query(accommodation_type, lat, lon, radius_km))
    except httpx.HTTPError as exc:
        return tool_error(http_error_message("Overpass API", exc))

    found = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        el_lat = element.get("lat") or element.get("center", {}).get("lat")
        el_lon = element.get("lon") or element.get("center", {}).get("lon")
        if el_lat is None or el_lon is None:
            continue
        entry = {
            "name": tags.get("name", "Unbenannt"),
            "type": accommodation_type.value,
            "lat": round(el_lat, 6),
            "lon": round(el_lon, 6),
            "distance_km": round(haversine_km(lat, lon, el_lat, el_lon), 1),
        }
        if entry["distance_km"] > radius_km:
            continue
        for key, out in [
            ("website", "website"), ("contact:website", "website"), ("phone", "phone"),
            ("opening_hours", "opening_hours"), ("beds", "beds"), ("capacity", "capacity"),
            ("ele", "elevation_m"), ("operator", "operator"), ("fee", "fee"),
        ]:  # fmt: skip
            if key in tags and out not in entry:
                entry[out] = tags[key]
        found.append(entry)

    # Benannte Einträge zuerst, dann nach Entfernung
    found.sort(key=lambda e: (e["name"] == "Unbenannt", e["distance_km"]))
    if not found:
        return {"results": [], "note": f"Keine Unterkünfte vom Typ {accommodation_type.value} im Umkreis von {radius_km:g} km."}
    return {"results": found[:MAX_RESULTS], "total_found": len(found)}
