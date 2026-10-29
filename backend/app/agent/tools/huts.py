"""Hüttentour planen: eine Kette von Hütten, die auf dem Weg zwischen Start und Ziel liegen.

Sucht der Agent Hütten rund um Talorte (`find_accommodation`), liegen sie oft abseits: als Sackgasse über dem Tal,
quer zur Laufrichtung oder als Almwirtschaft ohne Übernachtung. Hier werden stattdessen alle Hütten in einem
Korridor zwischen Start und Ziel gesucht und so ausgewählt, dass sie in Laufrichtung liegen und die Tagesetappen
etwa gleich lang werden.
"""

import math
import re
from typing import Any

import httpx
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AccommodationType, TransportMode
from app.agent.tools.accommodation import search_nominatim
from app.agent.tools.common import haversine_km, http_error_message, tool_error
from app.agent.tools.routing import Coordinate, cut_indices, plan_route

# Wanderwege im Gebirge sind deutlich länger als die Luftlinie
DETOUR_FACTOR = 1.4
MAX_LEG_KM = 22.0
# Kürzere Tagesetappen als Ziel nur, wenn die Strecke selbst so kurz ist (sonst landen alle Hütten am Start)
MIN_TARGET_LEG_KM = 10.0
# Aufschlag je Hütte ohne Hinweis auf Übernachtung, in Einheiten der Etappenabweichung
NO_OVERNIGHT_PENALTY = 3.0
# Nach dem Routing: Etappen, die viel länger als geschätzt sind, liegen nicht auf dem Weg → Hütte austauschen
MAX_ROUTED_LEG_KM = 26.0
MAX_ROUTED_DETOUR = 2.2
CHAIN_ATTEMPTS = 3
# Einträge so nah beieinander gelten als dieselbe Hütte (z. B. Hütte und Nebengebäude)
SAME_HUT_KM = 0.4
# Korridor: Höchstabstand zur Luftlinie Start–Ziel (km), auch rund um Start und Ziel. Zwischen nahen Talorten
# liegen oft keine Hütten auf der Achse (Garmisch–Mittenwald: Meilerhütte 7 km seitlich, Mittenwalder Hütte
# oberhalb des Ziels), daher auch bei kurzen Strecken mindestens 8 km.
MIN_CORRIDOR_KM = 8.0
MAX_CORRIDOR_KM = 10.0
CORRIDOR_SHARE = 0.25
# Etappen, deren tiefster Punkt so weit unter beiden Enden liegt, führen durchs Tal (Hinweis im Ergebnis). Ab
# DEEP_VALLEY_DROP_M wird die Verbindung verworfen und eine andere Kette gesucht (Wank-Haus → Meilerhütte über
# Garmisch); kürzere Talquerungen zwischen zwei Gebirgsgruppen sind oft unvermeidbar.
VALLEY_DROP_M = 500
DEEP_VALLEY_DROP_M = 800
MAX_OTHER_HUTS = 12
# Darunter ist es eher eine Rundtour bzw. Start und Ziel am selben Ort
MIN_HUT_TOUR_KM = 3.0

# Hinweise auf Übernachtung: Alpenvereine und Hüttennamen; Almen und Berggasthöfe sind oft nur Einkehr
CLUB_RE = re.compile(
    r"alpenverein|\b(dav|öav|oeav|avs|sac|cai|caf|pzs)\b|sektion|section|sezione|naturfreunde|club alpin",
    re.IGNORECASE,
)
HUT_NAME_RE = re.compile(r"hütte|haus|huette|rifugio|refuge|cabane|chamanna|biwak|bivacco|koča|hut\b", re.IGNORECASE)
# Nur Notunterkunft bzw. außer Betrieb
CLOSED_RE = re.compile(r"winterraum|winter room|\balte\b|ehemalig|ruine", re.IGNORECASE)


def overnight_likely(hut: dict[str, Any]) -> bool:
    if CLOSED_RE.search(hut["name"]):
        return False
    if hut.get("beds") or hut.get("capacity") or CLUB_RE.search(hut.get("operator", "")):
        return True
    return bool(HUT_NAME_RE.search(hut["name"])) and "alm" not in hut["name"].lower()


def _local_km(lat: float, lon: float, origin: tuple[float, float]) -> tuple[float, float]:
    """Ebene Näherung in km um den Ursprung (genügt für einige Dutzend km)."""
    return (lon - origin[1]) * 111.32 * math.cos(math.radians(origin[0])), (lat - origin[0]) * 110.57


def _axis_position(lat: float, lon: float, start: Coordinate, end: Coordinate) -> tuple[float, float]:
    """(km entlang der Luftlinie ab Start, seitlicher Abstand in km)"""
    origin = (start.lat, start.lon)
    ex, ey = _local_km(end.lat, end.lon, origin)
    px, py = _local_km(lat, lon, origin)
    length = math.hypot(ex, ey) or 1.0
    ux, uy = ex / length, ey / length
    return px * ux + py * uy, px * uy - py * ux


async def _huts_in_corridor(
    ctx: RunContext[AgentDeps], start: Coordinate, end: Coordinate, accommodation_type: AccommodationType, width: float
) -> list[dict[str, Any]]:
    """Kreissuchen entlang der Luftlinie; Nominatim liefert je Suche höchstens 40 Treffer, daher kleine Kreise."""
    distance = haversine_km(start.lat, start.lon, end.lat, end.lon)
    steps = max(1, math.ceil(distance / width))
    found: dict[tuple[str, float, float], dict[str, Any]] = {}
    for i in range(steps + 1):
        lat = start.lat + (end.lat - start.lat) * i / steps
        lon = start.lon + (end.lon - start.lon) * i / steps
        for hut in await search_nominatim(ctx, accommodation_type, lat, lon, width):
            if hut["name"] == "Unbenannt":
                continue
            found.setdefault((hut["name"], round(hut["lat"], 3), round(hut["lon"], 3)), hut)

    huts: list[dict[str, Any]] = []
    # Aussagekräftigster Eintrag zuerst, damit er bei Doppeleinträgen derselben Hütte bleibt
    for hut in sorted(found.values(), key=lambda h: (not overnight_likely(h), -len(h))):
        along, offset = _axis_position(hut["lat"], hut["lon"], start, end)
        if math.hypot(along - min(max(along, 0.0), distance), offset) > width:
            continue  # weiter als `width` von der Strecke Start–Ziel
        if any(haversine_km(hut["lat"], hut["lon"], h["lat"], h["lon"]) < SAME_HUT_KM for h in huts):
            continue
        hut = {k: v for k, v in hut.items() if k != "distance_km"}
        hut |= {"km_from_start": round(along, 1), "offset_km": round(offset, 1), "overnight_likely": overnight_likely(hut)}
        huts.append(hut)
    return sorted(huts, key=lambda h: h["km_from_start"])


def choose_chain(
    huts: list[dict[str, Any]],
    start: Coordinate,
    end: Coordinate,
    nights: int,
    forbidden: frozenset[tuple[int, int]] = frozenset(),
) -> list[dict[str, Any]] | None:
    """Wählt `nights` Hütten in Laufrichtung, sodass die Tagesetappen möglichst gleich lang und nicht zu lang sind.

    Dynamische Programmierung über die nach Fortschritt sortierten Hütten; Hütten ohne Hinweis auf Übernachtung
    kosten einen Aufschlag. `forbidden` enthält Verbindungen (i, j), die nach dem Routing nicht taugen; Indizes wie
    `points`: 0 = Start, 1..n = Hütten, n + 1 = Ziel.
    """
    points = [(start.lat, start.lon)] + [(h["lat"], h["lon"]) for h in huts] + [(end.lat, end.lon)]
    progress = [0.0] + [h["km_from_start"] for h in huts] + [haversine_km(start.lat, start.lon, end.lat, end.lon)]
    target = max(progress[-1] * DETOUR_FACTOR / (nights + 1), MIN_TARGET_LEG_KM)

    last = len(points) - 1

    def leg_cost(i: int, j: int) -> float:
        # Kein Rückschritt, keine zwei Hütten nebeneinander; nur zum Ziel darf es zurückgehen (letzte Hütte
        # oberhalb des Zielorts)
        if (j != last and progress[j] - progress[i] < 1.0) or (i, j) in forbidden:
            return math.inf
        km =haversine_km(*points[i], *points[j]) * DETOUR_FACTOR
        return ((km - target) / target) ** 2 + 5 * max(0.0, km - MAX_LEG_KM) / MAX_LEG_KM

    def node_cost(i: int) -> float:
        return 0.0 if i in (0, last) or huts[i - 1]["overnight_likely"] else NO_OVERNIGHT_PENALTY

    # best[k][j]: minimale Kosten bis Punkt j mit k Hütten (j selbst eingeschlossen), prev für den Rückweg
    best: list[dict[int, float]] = [{0: 0.0}]
    prev: list[dict[int, int]] = [{}]
    for k in range(1, nights + 2):
        targets = [last] if k == nights + 1 else range(1, last)
        layer: dict[int, float] = {}
        back: dict[int, int] = {}
        for j in targets:
            for i, cost in best[k - 1].items():
                total = cost + leg_cost(i, j) + node_cost(j)
                if total < layer.get(j, math.inf):
                    layer[j], back[j] = total, i
        best.append(layer)
        prev.append(back)
    if not math.isfinite(best[-1].get(last, math.inf)):
        return None

    chain, j = [], last
    for k in range(nights + 1, 1, -1):
        j = prev[k][j]
        chain.append(huts[j - 1])
    return chain[::-1]


def valley_drops(geometry: list[list[float]], waypoints: list[Coordinate]) -> list[float | None]:
    """Je Etappe: wie weit der tiefste Punkt unter dem tieferen der beiden Enden liegt (m; Höhen aus der Route)."""
    if not geometry or len(geometry[0]) < 3:
        return [None] * (len(waypoints) - 1)
    cuts = cut_indices(geometry, waypoints)
    return [
        min(geometry[a][2], geometry[b][2]) - min(p[2] for p in geometry[a : b + 1])
        for a, b in zip(cuts, cuts[1:])
    ]


def problem_legs(waypoints: list[Coordinate], route: dict[str, Any], drops: list[float | None]) -> list[int]:
    """Etappen (0-basiert), deren Wanderweg viel länger als Luftlinie bzw. Tagesetappe ist oder tief ins Tal führt."""
    segments = route.get("segments", [])
    if route.get("estimated") or len(segments) != len(waypoints) - 1:
        return []  # Schätzung ohne echten Wegverlauf sagt über Umwege nichts aus
    problems = []
    for day, (segment, drop) in enumerate(zip(segments, drops)):
        a, b = waypoints[day], waypoints[day + 1]
        straight = haversine_km(a.lat, a.lon, b.lat, b.lon)
        km = segment["distance_km"]
        if km > MAX_ROUTED_LEG_KM or km > straight * MAX_ROUTED_DETOUR + 3 or (drop or 0) > DEEP_VALLEY_DROP_M:
            problems.append(day)
    return problems


async def plan_hut_tour(
    ctx: RunContext[AgentDeps],
    start: Coordinate,
    end: Coordinate,
    nights: int,
    accommodation_type: AccommodationType = AccommodationType.hut,
) -> dict[str, Any]:
    """Plant eine Hütten-Wanderung von Start nach Ziel mit Hütten, die auf dem Weg liegen (OpenStreetMap).

    Sucht Hütten im Korridor zwischen Start und Ziel, wählt je Nacht eine Hütte in Laufrichtung mit möglichst
    gleich langen Tagesetappen und berechnet die Wanderroute über diese Hütten.

    Args:
        start: Startort der Wanderung.
        end: Ziel der Wanderung (bei einer Rundtour Start = Ziel nicht geeignet, dann find_accommodation).
        nights: Anzahl der Übernachtungen (= Tage - 1).
        accommodation_type: hut (bewirtschaftete Hütten) oder bivouac.

    Returns:
        huts (gewählte Hütten in Reihenfolge), route (wie plan_route, segments je Etappe mit lowest_m),
        other_huts (weitere Hütten im Korridor zum Austauschen) und ggf. note.
    """
    if nights < 1:
        return tool_error("Mindestens eine Übernachtung angeben; für Tagestouren plan_route nutzen.")
    if accommodation_type not in (AccommodationType.hut, AccommodationType.bivouac):
        return tool_error("Nur für hut oder bivouac; andere Unterkünfte mit find_accommodation suchen.")
    distance = haversine_km(start.lat, start.lon, end.lat, end.lon)
    if distance < MIN_HUT_TOUR_KM:
        return tool_error("Start und Ziel liegen zu nah beieinander (Rundtour?). Hütten mit find_accommodation suchen.")

    width = min(MAX_CORRIDOR_KM, max(MIN_CORRIDOR_KM, distance * CORRIDOR_SHARE))
    try:
        huts = await _huts_in_corridor(ctx, start, end, accommodation_type, width)
    except httpx.HTTPError as exc:
        return tool_error(http_error_message("Nominatim", exc) + " Ersatzweise find_accommodation nutzen.")

    # Kette wählen und routen; Verbindungen mit großem Umweg oder tiefem Talabstieg sperren und neu wählen.
    # Von den Versuchen gewinnt der mit den wenigsten solchen Etappen.
    index = {id(h): i + 1 for i, h in enumerate(huts)}
    forbidden: set[tuple[int, int]] = set()
    attempts: list[tuple[int, int, list[dict[str, Any]], dict[str, Any], list[float | None]]] = []
    for attempt in range(CHAIN_ATTEMPTS):
        candidate = choose_chain(huts, start, end, nights, frozenset(forbidden))
        if candidate is None:
            break
        waypoints = [start, *(Coordinate(lat=h["lat"], lon=h["lon"]) for h in candidate), end]
        candidate_route = await plan_route(ctx, waypoints, TransportMode.hiking)
        drops = valley_drops(ctx.deps.routes.get(candidate_route.get("route_id", ""), []), waypoints)
        problems = problem_legs(waypoints, candidate_route, drops)
        attempts.append((len(problems), attempt, candidate, candidate_route, drops))
        if not problems:
            break
        stops = [0, *(index[id(h)] for h in candidate), len(huts) + 1]
        forbidden |= {(stops[day], stops[day + 1]) for day in problems}
    if not attempts:
        return {
            "huts": [],
            "other_huts": huts[:MAX_OTHER_HUTS],
            "note": f"Keine {nights} Hütten in Laufrichtung im Korridor von {width:g} km gefunden. Weniger "
            "Übernachtungen, Talorte mit find_accommodation oder ein anderes Ziel vorschlagen.",
        }

    _, _, chain, route, drops = min(attempts, key=lambda a: a[:2])
    notes = []
    unlikely = [h["name"] for h in chain if not h["overnight_likely"]]
    if unlikely:
        notes.append(f"Ohne Hinweis auf Übernachtung (evtl. nur Einkehr): {', '.join(unlikely)}.")
    for day, (segment, drop) in enumerate(zip(route.get("segments", []), drops), start=1):
        if "lowest_m" in segment and (drop or 0) > VALLEY_DROP_M:
            notes.append(f"Etappe {day} steigt bis auf {segment['lowest_m']} m ins Tal ab.")
    chosen = {id(h) for h in chain}
    others = sorted(
        (h for h in huts if id(h) not in chosen), key=lambda h: (not h["overnight_likely"], abs(h["offset_km"]))
    )
    result: dict[str, Any] = {
        "huts": chain,
        "route": route,
        "other_huts": sorted(others[:MAX_OTHER_HUTS], key=lambda h: h["km_from_start"]),
    }
    if notes:
        result["note"] = " ".join(notes)
    return result
