import math
import time
import uuid
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.plan_schema import TransportMode
from app.agent.tools.common import haversine_km, http_error_message, tool_error

ORS_URL = "https://api.openrouteservice.org/v2/directions/{profile}/geojson"

# Profile in Reihenfolge der Präferenz. cycling-regular ist bei ORS zeitweise sehr langsam;
# cycling-electric routet fast identisch (Tourenrad), cycling-road als letzte Alternative.
ORS_PROFILES: dict[TransportMode, list[str]] = {
    TransportMode.hiking: ["foot-hiking"],
    TransportMode.cycling: ["cycling-regular", "cycling-electric", "cycling-road"],
    TransportMode.mtb: ["cycling-mountain"],
    TransportMode.car: ["driving-car"],
}
ORS_TIMEOUT_SECONDS = 12
# Bei diesen Status ein anderes Profil versuchen (Überlast; 404 = Punkt im Profil nicht routbar)
ORS_RETRY_STATUS = {404, 429, 500, 502, 503, 504}
SLOW_PROFILE_COOLDOWN_SECONDS = 600

# Profil -> Zeitpunkt, bis zu dem es nach einem Timeout übersprungen wird (sofern es Alternativen gibt)
_slow_until: dict[str, float] = {}

# Für die Luftlinien-Schätzung: Umwegfaktor und Durchschnittsgeschwindigkeit (km/h)
ESTIMATE_FACTORS: dict[TransportMode, tuple[float, float]] = {
    TransportMode.hiking: (1.3, 4.0),
    TransportMode.cycling: (1.25, 18.0),
    TransportMode.mtb: (1.35, 12.0),
    TransportMode.car: (1.3, 70.0),
    TransportMode.canoe: (1.4, 5.0),
    TransportMode.public_transport: (1.3, 50.0),
}

# Vereinfachung der Geometrie: genau genug für GPX-Navigation, klein genug für die Karte
SIMPLIFY_TOLERANCE_M = 3.0
MAX_GEOMETRY_POINTS = 4000
METERS_PER_DEGREE = 111_320.0


class Coordinate(BaseModel):
    lat: float
    lon: float


def simplify_track(points: list[list[float]], tolerance_m: float = SIMPLIFY_TOLERANCE_M) -> list[list[float]]:
    """Douglas-Peucker auf [lat, lon, (ele)]-Punkten; behält Kurven, entfernt Punkte auf Geraden.

    Wird die Obergrenze überschritten, wird die Toleranz schrittweise verdoppelt.
    """
    if len(points) <= 2:
        return points
    cos_lat = math.cos(math.radians(points[0][0]))

    def xy(p: list[float]) -> tuple[float, float]:
        return p[1] * cos_lat * METERS_PER_DEGREE, p[0] * METERS_PER_DEGREE

    coords = [xy(p) for p in points]
    while True:
        keep = [False] * len(points)
        keep[0] = keep[-1] = True
        stack = [(0, len(points) - 1)]
        while stack:
            start, end = stack.pop()
            (x1, y1), (x2, y2) = coords[start], coords[end]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            max_dist, index = 0.0, -1
            for i in range(start + 1, end):
                px, py = coords[i]
                if length == 0:
                    dist = math.hypot(px - x1, py - y1)
                else:
                    dist = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length
                if dist > max_dist:
                    max_dist, index = dist, i
            if max_dist > tolerance_m:
                keep[index] = True
                stack.extend([(start, index), (index, end)])
        result = [p for p, k in zip(points, keep) if k]
        if len(result) <= MAX_GEOMETRY_POINTS:
            return result
        tolerance_m *= 2


def _estimate(waypoints: list[Coordinate], transport: TransportMode, reason: str) -> dict[str, Any]:
    factor, speed = ESTIMATE_FACTORS[transport]
    segments = []
    for a, b in zip(waypoints, waypoints[1:]):
        km = haversine_km(a.lat, a.lon, b.lat, b.lon) * factor
        segments.append({"distance_km": round(km, 1), "duration_hours": round(km / speed, 1)})
    return {
        "estimated": True,
        "note": f"{reason} Werte sind eine Luftlinien-Schätzung (Faktor {factor}) ohne Höhenmeter.",
        "distance_km": round(sum(s["distance_km"] for s in segments), 1),
        "duration_hours": round(sum(s["duration_hours"] for s in segments), 1),
        "segments": segments,
    }


async def plan_route(
    ctx: RunContext[AgentDeps], waypoints: list[Coordinate], transport: TransportMode
) -> dict[str, Any]:
    """Berechnet eine Route über die angegebenen Wegpunkte (OpenRouteService).

    Für mehrtägige Touren die Etappenorte bzw. Unterkünfte als Zwischenpunkte übergeben:
    `segments` enthält dann Distanz, Dauer und Höhenmeter je Teilstück (= je Etappe).

    Args:
        waypoints: Mindestens 2 Koordinaten in Reihenfolge: Start, optionale Zwischenpunkte, Ziel.
        transport: Fortbewegungsart.

    Returns:
        route_id (für AdventurePlan.route_ids), distance_km, duration_hours, ascent_m, descent_m, segments.
    """
    if len(waypoints) < 2:
        return tool_error("Mindestens Start und Ziel als Wegpunkte angeben.")
    if len(waypoints) > 25:
        return tool_error("Maximal 25 Wegpunkte pro Anfrage.")

    profiles = ORS_PROFILES.get(transport)
    if profiles is None:
        return _estimate(waypoints, transport, f"Für '{transport.value}' gibt es kein Straßen-/Wegerouting.")
    api_key = ctx.deps.settings.ors_api_key
    if not api_key:
        return _estimate(waypoints, transport, "Kein ORS_API_KEY konfiguriert.")

    body = {
        "coordinates": [[w.lon, w.lat] for w in waypoints],
        "elevation": True,
        "instructions": False,
    }
    response: httpx.Response | None = None
    problems: list[str] = []
    available = [p for p in profiles if _slow_until.get(p, 0) < time.monotonic()] or profiles
    for profile in available:
        try:
            response = await ctx.deps.http.post(
                ORS_URL.format(profile=profile),
                json=body,
                headers={"Authorization": api_key},
                timeout=ORS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            problems.append(f"{profile}: {http_error_message('OpenRouteService', exc)}")
            response = None
            if exc.response.status_code not in ORS_RETRY_STATUS:
                break  # z. B. Distanzlimit überschritten – ein anderes Profil hilft nicht
        except httpx.HTTPError as exc:
            problems.append(f"{profile}: {http_error_message('OpenRouteService', exc)}")
            response = None
            if isinstance(exc, httpx.TimeoutException):
                _slow_until[profile] = time.monotonic() + SLOW_PROFILE_COOLDOWN_SECONDS
    if response is None:
        # Schätzung statt Abbruch, damit der Agent weiterplanen kann
        return _estimate(waypoints, transport, " ".join(problems))

    feature = response.json()["features"][0]
    props = feature["properties"]
    summary = props.get("summary", {})
    route_id = f"r-{uuid.uuid4().hex[:8]}"
    # GeoJSON liefert [lon, lat, ele]; gespeichert wird [lat, lon, ele] (Leaflet- und GPX-Reihenfolge)
    ctx.deps.routes[route_id] = simplify_track(
        [[round(c[1], 6), round(c[0], 6), *([round(c[2], 1)] if len(c) > 2 else [])] for c in feature["geometry"]["coordinates"]]
    )

    return {
        "route_id": route_id,
        "estimated": False,
        "profile": profile,
        "distance_km": round(summary.get("distance", 0) / 1000, 1),
        "duration_hours": round(summary.get("duration", 0) / 3600, 1),
        "ascent_m": round(props.get("ascent", 0)),
        "descent_m": round(props.get("descent", 0)),
        "segments": [
            {
                "distance_km": round(s.get("distance", 0) / 1000, 1),
                "duration_hours": round(s.get("duration", 0) / 3600, 1),
                "ascent_m": round(s.get("ascent", 0)),
                "descent_m": round(s.get("descent", 0)),
            }
            for s in props.get("segments", [])
        ],
    }


def estimate_hiking_time(distance_km: float, ascent_m: float = 0, descent_m: float = 0) -> dict[str, Any]:
    """Berechnet die Gehzeit einer Wanderetappe nach der DAV-/SAC-Faustformel.

    Horizontal 4 km/h, Aufstieg 300 Hm/h, Abstieg 500 Hm/h; die kleinere Zeit zählt zur Hälfte.

    Args:
        distance_km: Strecke in km.
        ascent_m: Aufstieg in Höhenmetern.
        descent_m: Abstieg in Höhenmetern.
    """
    horizontal = distance_km / 4
    vertical = ascent_m / 300 + descent_m / 500
    hours = max(horizontal, vertical) + min(horizontal, vertical) / 2
    return {
        "walking_hours": round(hours, 1),
        "hint": "Reine Gehzeit ohne Pausen; mit Pausen etwa 20-30 % aufschlagen.",
    }
