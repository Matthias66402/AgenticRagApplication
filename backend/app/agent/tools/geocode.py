import asyncio
import time
from typing import Any

import httpx
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.tools.common import http_error_message, tool_error

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim-Nutzungsrichtlinie: max. 1 Anfrage pro Sekunde
_rate_lock = asyncio.Lock()
_last_request = 0.0
_cache: dict[tuple[str, str | None], list[dict[str, Any]]] = {}


async def _throttle() -> None:
    global _last_request
    async with _rate_lock:
        wait = 1.0 - (time.monotonic() - _last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request = time.monotonic()


async def nominatim_search(ctx: RunContext[AgentDeps], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Gedrosselte Nominatim-Suche (gemeinsam für Geocoding und Unterkunftssuche)."""
    await _throttle()
    response = await ctx.deps.http.get(
        NOMINATIM_URL,
        params={"format": "jsonv2", "accept-language": "de"} | params,
        headers={"User-Agent": ctx.deps.settings.nominatim_user_agent},
    )
    response.raise_for_status()
    return response.json()


async def geocode(
    ctx: RunContext[AgentDeps], place_name: str, country_codes: str | None = None
) -> list[dict[str, Any]] | dict[str, Any]:
    """Ermittelt Koordinaten für einen Ort, Berg, eine Hütte oder Adresse (OpenStreetMap/Nominatim).

    Args:
        place_name: Name des Ortes, z. B. "Garmisch-Partenkirchen" oder "Zugspitze".
        country_codes: Optional kommagetrennte ISO-Ländercodes zur Eingrenzung, z. B. "de,at,ch".

    Returns:
        Bis zu 3 Treffer mit name, lat, lon und type, der beste zuerst.
    """
    key = (place_name.strip().lower(), country_codes)
    if key in _cache:
        return _cache[key]

    params: dict[str, Any] = {"q": place_name, "limit": 3}
    if country_codes:
        params["countrycodes"] = country_codes
    try:
        items = await nominatim_search(ctx, params)
    except httpx.HTTPError as exc:
        return tool_error(http_error_message("Nominatim", exc))

    results = [
        {
            "name": item.get("display_name", place_name),
            "lat": round(float(item["lat"]), 6),
            "lon": round(float(item["lon"]), 6),
            "type": f"{item.get('category', '')}/{item.get('type', '')}",
        }
        for item in items
    ]
    if not results:
        return tool_error(f"Kein Ort zu '{place_name}' gefunden. Schreibweise prüfen oder den Nutzer fragen.")
    _cache[key] = results
    return results
