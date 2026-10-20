"""Routing über OpenRouteService (mit API-Key) und BRouter (ohne Key) mit gegenseitigem Fallback."""

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.plan_schema import TransportMode
from app.agent.tools.common import haversine_km, http_error_message, tool_error

ORS_URL = "https://api.openrouteservice.org/v2/directions/{profile}/geojson"
BROUTER_URL = "https://brouter.de/brouter"

# (Anbieter, Profil) in Reihenfolge der Präferenz. ORS hängt zeitweise (cycling-regular teils stundenlang),
# BRouter ist schnell und braucht keinen Key; cycling-electric/-road routen ähnlich wie cycling-regular.
ROUTING_CHAIN: dict[TransportMode, list[tuple[str, str]]] = {
    TransportMode.hiking: [("ors", "foot-hiking"), ("brouter", "hiking-mountain")],
    TransportMode.cycling: [
        ("ors", "cycling-regular"), ("brouter", "trekking"), ("ors", "cycling-electric"), ("ors", "cycling-road"),
    ],
    TransportMode.mtb: [("ors", "cycling-mountain"), ("brouter", "mtb")],
    TransportMode.car: [("ors", "driving-car"), ("brouter", "car-fast")],
}  # fmt: skip
PROVIDER_NAMES = {"ors": "OpenRouteService", "brouter": "BRouter"}

# ORS hängt sporadisch bei einzelnen Anfragen; eine Wiederholung antwortet meist sofort
ORS_TIMEOUT_SECONDS = 8
ORS_ATTEMPTS_PER_PROFILE = 2
BROUTER_TIMEOUT_SECONDS = 20
SLOW_PROFILE_COOLDOWN_SECONDS = 600
# Höhen-Hysterese für selbst berechnete Höhenmeter (filtert Messrauschen der Höhendaten)
ELEVATION_HYSTERESIS_M = 10.0

# "anbieter:profil" -> Zeitpunkt, bis zu dem es nach wiederholtem Timeout übersprungen wird
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


@dataclass
class RouteData:
    """Einheitliches Ergebnis beider Anbieter. Geometrie als [lat, lon, ele]."""

    geometry: list[list[float]]
    distance_m: float
    duration_s: float
    ascent_m: float
    descent_m: float
    segments: list[dict[str, float]] = field(default_factory=list)


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


def elevation_gain(points: list[list[float]], hysteresis_m: float = ELEVATION_HYSTERESIS_M) -> tuple[float, float]:
    """Auf- und Abstieg mit Hysterese.

    Ein Richtungswechsel zählt erst ab `hysteresis_m`; in der bestätigten Richtung zählt jeder weitere Meter.
    """
    elevations = [p[2] for p in points if len(p) > 2]
    if len(elevations) < 2:
        return 0.0, 0.0
    ascent = descent = 0.0
    reference, direction = elevations[0], 0
    for ele in elevations[1:]:
        diff = ele - reference
        if diff >= hysteresis_m or (direction > 0 and diff > 0):
            ascent += diff
            reference, direction = ele, 1
        elif diff <= -hysteresis_m or (direction < 0 and diff < 0):
            descent -= diff
            reference, direction = ele, -1
    return ascent, descent


def split_segments(
    geometry: list[list[float]], waypoints: list[Coordinate], duration_s: float
) -> list[dict[str, float]]:
    """Teilt eine Gesamtroute an den Zwischen-Wegpunkten und berechnet Werte je Teilstück."""
    cumulative = [0.0]
    for a, b in zip(geometry, geometry[1:]):
        cumulative.append(cumulative[-1] + haversine_km(a[0], a[1], b[0], b[1]) * 1000)
    total = cumulative[-1] or 1.0

    cuts = [0]
    for waypoint in waypoints[1:-1]:
        start = cuts[-1]
        nearest = min(
            range(start, len(geometry)),
            key=lambda i: haversine_km(waypoint.lat, waypoint.lon, geometry[i][0], geometry[i][1]),
        )
        cuts.append(nearest)
    cuts.append(len(geometry) - 1)

    segments = []
    for start, end in zip(cuts, cuts[1:]):
        distance = cumulative[end] - cumulative[start]
        ascent, descent = elevation_gain(geometry[start : end + 1])
        segments.append({
            "distance": distance, "duration": duration_s * distance / total, "ascent": ascent, "descent": descent,
        })  # fmt: skip
    return segments


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


async def _route_ors(ctx: RunContext[AgentDeps], profile: str, waypoints: list[Coordinate]) -> RouteData:
    body = {
        "coordinates": [[w.lon, w.lat] for w in waypoints],
        "elevation": True,
        "instructions": True,  # nötig, damit ORS `segments` (Werte je Teilstück = je Etappe) mitliefert
    }
    for attempt in range(1, ORS_ATTEMPTS_PER_PROFILE + 1):
        try:
            response = await ctx.deps.http.post(
                ORS_URL.format(profile=profile),
                json=body,
                headers={"Authorization": ctx.deps.settings.ors_api_key},
                timeout=ORS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            break
        except httpx.TimeoutException:
            if attempt == ORS_ATTEMPTS_PER_PROFILE:
                raise

    feature = response.json()["features"][0]
    props = feature["properties"]
    summary = props.get("summary", {})
    return RouteData(
        # GeoJSON liefert [lon, lat, ele]; gespeichert wird [lat, lon, ele] (Leaflet- und GPX-Reihenfolge)
        geometry=[[c[1], c[0], *c[2:3]] for c in feature["geometry"]["coordinates"]],
        distance_m=summary.get("distance", 0),
        duration_s=summary.get("duration", 0),
        ascent_m=props.get("ascent", 0),
        descent_m=props.get("descent", 0),
        segments=props.get("segments", []),
    )


async def _route_brouter(ctx: RunContext[AgentDeps], profile: str, waypoints: list[Coordinate]) -> RouteData:
    response = await ctx.deps.http.get(
        BROUTER_URL,
        params={
            "lonlats": "|".join(f"{w.lon:.6f},{w.lat:.6f}" for w in waypoints),
            "profile": profile,
            "alternativeidx": 0,
            "format": "geojson",
        },
        headers={"User-Agent": ctx.deps.settings.nominatim_user_agent},
        timeout=BROUTER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    feature = response.json()["features"][0]
    props = feature["properties"]
    geometry = [[c[1], c[0], *c[2:3]] for c in feature["geometry"]["coordinates"]]
    duration = float(props.get("total-time", 0))
    ascent, descent = elevation_gain(geometry)
    return RouteData(
        geometry=geometry,
        distance_m=float(props.get("track-length", 0)),
        duration_s=duration,
        ascent_m=ascent,
        descent_m=descent,
        segments=split_segments(geometry, waypoints, duration),
    )


ROUTERS = {"ors": _route_ors, "brouter": _route_brouter}


async def plan_route(
    ctx: RunContext[AgentDeps], waypoints: list[Coordinate], transport: TransportMode
) -> dict[str, Any]:
    """Berechnet eine Route über die angegebenen Wegpunkte (OpenRouteService, Ersatz: BRouter).

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

    chain = ROUTING_CHAIN.get(transport)
    if chain is None:
        return _estimate(waypoints, transport, f"Für '{transport.value}' gibt es kein Straßen-/Wegerouting.")
    if not ctx.deps.settings.ors_api_key:
        chain = [(provider, profile) for provider, profile in chain if provider != "ors"]

    now = time.monotonic()
    available = [entry for entry in chain if _slow_until.get(":".join(entry), 0) < now] or chain
    problems: list[str] = []
    route: RouteData | None = None
    for provider, profile in available:
        label = f"{PROVIDER_NAMES[provider]} {profile}"
        try:
            route = await ROUTERS[provider](ctx, profile, waypoints)
            break
        except httpx.HTTPError as exc:
            # Jeder Fehler (Überlast, ungültiger Key, Distanzlimit, nicht routbarer Punkt) → nächster Eintrag
            problems.append(f"{label}: {http_error_message(PROVIDER_NAMES[provider], exc)}")
            if isinstance(exc, httpx.TimeoutException):
                # Hing bei jedem Versuch → eine Weile überspringen
                _slow_until[f"{provider}:{profile}"] = time.monotonic() + SLOW_PROFILE_COOLDOWN_SECONDS
        except (KeyError, IndexError, ValueError) as exc:  # unerwartetes Antwortformat
            problems.append(f"{label}: unerwartete Antwort ({exc})")
    if route is None:
        # Schätzung statt Abbruch, damit der Agent weiterplanen kann
        return _estimate(waypoints, transport, " ".join(problems))

    route_id = f"r-{uuid.uuid4().hex[:8]}"
    ctx.deps.routes[route_id] = simplify_track(
        [[round(p[0], 6), round(p[1], 6), *[round(e, 1) for e in p[2:3]]] for p in route.geometry]
    )
    return {
        "route_id": route_id,
        "estimated": False,
        "source": label,
        "distance_km": round(route.distance_m / 1000, 1),
        "duration_hours": round(route.duration_s / 3600, 1),
        "ascent_m": round(route.ascent_m),
        "descent_m": round(route.descent_m),
        "segments": [
            {
                "distance_km": round(s.get("distance", 0) / 1000, 1),
                "duration_hours": round(s.get("duration", 0) / 3600, 1),
                "ascent_m": round(s.get("ascent", 0)),
                "descent_m": round(s.get("descent", 0)),
            }
            for s in route.segments
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
