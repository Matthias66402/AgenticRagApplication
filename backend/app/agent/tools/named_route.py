"""Verlauf ausgeschilderter Fernwege (OSM-Routenrelationen, z. B. Ostseeküsten-Radweg) als Leitlinie fürs Routing.

Quelle ist zuerst der vorab importierte Katalog (Tabelle long_distance_routes, app.seed.long_distance_routes): sofort
verfügbar und mit weiteren Namen wie "Ostseeradweg". Fehlt der Fernweg dort, sucht Overpass live die passenden
Relationen nahe den Wegpunkten, die übrigen Abschnitte derselben Superroute und deren Wege (oft ein bis zwei Minuten).
Aus den Wegen entsteht ein Graph; zwischen den Wegpunkten wird der kürzeste Pfad innerhalb des Fernwegs gesucht.
`plan_route` gibt diesen Pfad als dichte Stützpunkte an den Router, der Höhen und Fahrzeiten liefert. Die Wege einer
Relation sind in OSM ungeordnet und enthalten Varianten; der Graph braucht keine Reihenfolge.
"""

import heapq
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.agent.deps import AgentDeps
from app.agent.plan_schema import TransportMode
from app.agent.tools.accommodation import query_overpass
from app.agent.tools.common import haversine_km
from app.models import LongDistanceRoute

Point = tuple[float, float]  # (lat, lon)
Way = tuple[list[int], list[Point]]  # Knoten-IDs und Koordinaten

ROUTE_TYPES: dict[TransportMode, str] = {
    TransportMode.cycling: "bicycle|mtb",
    TransportMode.mtb: "bicycle|mtb",
    TransportMode.hiking: "hiking|foot",
}
NAME_KEYS = ("name", "name:de", "official_name", "alt_name", "short_name")
NETWORK_RANK = {"icn": 0, "iwn": 0, "ncn": 1, "nwn": 1, "rcn": 2, "rwn": 2}

# Relationen werden über Wege im Umkreis der Wegpunkte gefunden (schnell, liefert passende Namensvorschläge);
# weitere Abschnitte desselben Fernwegs kommen über die übergeordnete Superroute dazu.
SEARCH_RADIUS_M = 3000
# Wege nur im Bereich der Wegpunkte plus Rand, auf ein Raster gerundet (gleiche Abfrage bei neuen Unterkünften)
BBOX_MARGIN_DEG = 0.3
BBOX_GRID_DEG = 0.5
OVERPASS_TIMEOUT_SECONDS = 90

GRID_DEG = 0.01  # Zellgröße des Suchrasters (~1 km)
MAX_SNAP_KM = 5.0  # weiter entfernte Wegpunkte (z. B. Unterkunft im Hinterland) werden normal angebunden
GAP_KM = 0.1  # Lücken zwischen Wegen der Relation bis zu dieser Länge werden überbrückt
# Pfad länger als 6 × Luftlinie (+2 km) → Fernweg verbindet die Punkte nicht sinnvoll (z. B. nur über das andere
# Flussufer weit entfernt). Küstenwege um Halbinseln und Buchten sind oft 3-4 × so lang wie die Luftlinie
# (Nordseeküsten-Radweg Emden–Wilhelmshaven: 201 km bei 62 km Luftlinie); genau diesen Verlauf will der Nutzer.
DETOUR_LIMIT = 6.0
SAMPLE_SPACING_KM = 2.0  # Abstand der Stützpunkte für den Router
SNAP_POINT_MIN_KM = 0.2  # Einstiegspunkt auf dem Fernweg nur bei spürbarem Abstand als eigener Stützpunkt

MAX_CANDIDATES = 15
CACHE_SECONDS = 6 * 3600
MAX_CACHED_GRAPHS = 8


def normalize(text: str) -> str:
    """Kleinbuchstaben ohne Akzente, Leer- und Sonderzeichen: "Ostseeküsten-Radweg" → "ostseekuestenradweg"."""
    text = text.lower()
    for umlaut, replacement in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(umlaut, replacement)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if c.isalnum() and not unicodedata.combining(c))


def short_name(tags: dict[str, str]) -> str:
    """Name ohne Abschnittsangabe: "D2 Ostseeküsten-Radweg - Abschnitt SH-Ost" → "D2 Ostseeküsten-Radweg"."""
    name = tags.get("name") or tags.get("ref") or ""
    return re.split(r"\s+[-–:]\s+", name)[0]


def without_ref(text: str) -> str:
    """Name ohne vorangestelltes Kürzel: "EV10 Ostseeküsten-Radweg" → "Ostseeküsten-Radweg"."""
    return re.sub(r"^(\S*\d\S*\s+)+", "", text)


def catalog_matches(query: str, names: list[str]) -> bool:
    """Katalogeintrag passt: gleicher Name bzw. weiterer Name (auch ohne Kürzel davor) oder längerer Namensanfang.

    Kürzel wie "D1" müssen exakt passen, sonst träfe "D1" auch "D10".
    """
    wanted = {normalize(query), normalize(without_ref(query))} - {""}
    known = [normalize(n) for n in names]
    return any(w == n or (len(w) >= 6 and n.startswith(w)) for w in wanted for n in known)


def matches(query: str, tags: dict[str, str]) -> bool:
    """Name beginnt mit der Suche (ein Kürzel wie "D2" oder "EV10" davor darf fehlen) oder `ref` ist gleich.

    Bewusst kein beliebiger Teilstring: "Ostseeradweg" soll nicht den "Nord-Ostsee-Radweg" treffen.
    """
    wanted = normalize(query)
    if len(wanted) < 2:
        return False
    if normalize(tags.get("ref", "")) == wanted:
        return True
    for key in NAME_KEYS:
        text = tags.get(key, "")
        if text and (normalize(text).startswith(wanted) or normalize(without_ref(text)).startswith(wanted)):
            return True
    return False


def search_bbox(points: list[Point]) -> tuple[float, float, float, float]:
    """(Süd, West, Nord, Ost) um alle Punkte, nach außen auf das Raster gerundet."""
    grid = BBOX_GRID_DEG
    return (
        round(math.floor((min(p[0] for p in points) - BBOX_MARGIN_DEG) / grid) * grid, 2),
        round(math.floor((min(p[1] for p in points) - BBOX_MARGIN_DEG) / grid) * grid, 2),
        round(math.ceil((max(p[0] for p in points) + BBOX_MARGIN_DEG) / grid) * grid, 2),
        round(math.ceil((max(p[1] for p in points) + BBOX_MARGIN_DEG) / grid) * grid, 2),
    )


def relation_query(transport: TransportMode, points: list[Point]) -> str:
    around = "".join(f"way(around:{SEARCH_RADIUS_M},{lat:.4f},{lon:.4f});" for lat, lon in points)
    return f'[out:json][timeout:60];\n({around});\nrel(bw)["route"~"^({ROUTE_TYPES[transport]})$"];\nout tags;'


def way_query(relation_ids: list[int], bbox: tuple[float, float, float, float]) -> str:
    # Lokale statt globaler Bounding Box: schneidet die Geometrie der Wege nicht ab
    s, w, n, e = bbox
    ids = ",".join(str(i) for i in relation_ids)
    return (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_SECONDS}];\n"
        f"rel(id:{ids})->.found;\n"
        f'rel(br.found)["type"="superroute"]->.parents;\n'
        f"(.found; rel(r.parents););\n"  # gefundene Abschnitte und ihre Geschwister
        f"way(r)({s},{w},{n},{e});\n"
        f"out skel geom;"
    )


def _cell(lat: float, lon: float) -> tuple[int, int]:
    return math.floor(lat / GRID_DEG), math.floor(lon / GRID_DEG)


@dataclass
class RouteGraph:
    names: list[str]
    coords: dict[int, Point] = field(default_factory=dict, repr=False)
    edges: dict[int, dict[int, float]] = field(default_factory=dict, repr=False)
    grid: dict[tuple[int, int], list[int]] = field(default_factory=dict, repr=False)

    def _connect(self, a: int, b: int) -> None:
        km = haversine_km(*self.coords[a], *self.coords[b])
        self.edges.setdefault(a, {})[b] = km
        self.edges.setdefault(b, {})[a] = km

    def nearby(self, lat: float, lon: float, radius_km: float) -> list[tuple[float, int]]:
        """(Entfernung, Knoten) aller Knoten im Umkreis."""
        dlat = math.ceil(radius_km / 111.0 / GRID_DEG)
        dlon = math.ceil(radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.1)) / GRID_DEG)
        row, col = _cell(lat, lon)
        found = []
        for r in range(row - dlat, row + dlat + 1):
            for c in range(col - dlon, col + dlon + 1):
                for node in self.grid.get((r, c), []):
                    km = haversine_km(lat, lon, *self.coords[node])
                    if km <= radius_km:
                        found.append((km, node))
        return found

    def snap(self, point: Point) -> tuple[int, float] | None:
        """Nächster Knoten des Fernwegs (Knoten, Entfernung in km) oder None, wenn weiter als MAX_SNAP_KM."""
        found = self.nearby(*point, MAX_SNAP_KM)
        if not found:
            return None
        km, node = min(found)
        return node, km

    def shortest_path(self, start: int, goal: int) -> tuple[list[Point], float] | None:
        """Dijkstra innerhalb des Fernwegs; (Punkte, Länge in km) oder None ohne Verbindung."""
        dist = {start: 0.0}
        previous: dict[int, int] = {}
        queue = [(0.0, start)]
        while queue:
            km, node = heapq.heappop(queue)
            if node == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                return [self.coords[n] for n in reversed(path)], km
            if km > dist[node]:
                continue
            for neighbour, length in self.edges.get(node, {}).items():
                candidate = km + length
                if candidate < dist.get(neighbour, math.inf):
                    dist[neighbour] = candidate
                    previous[neighbour] = node
                    heapq.heappush(queue, (candidate, neighbour))
        return None


def overpass_ways(data: dict[str, Any]) -> list[Way]:
    ways = []
    for element in data.get("elements", []):
        if element.get("type") == "way":
            # Overpass liefert für fehlende Knoten null
            nodes = zip(element.get("nodes", []), element.get("geometry", []))
            pairs = [(n, (p["lat"], p["lon"])) for n, p in nodes if p]
            ways.append(([n for n, _ in pairs], [p for _, p in pairs]))
    return ways


def build_graph(names: list[str], ways: list[Way]) -> RouteGraph:
    graph = RouteGraph(names=names)
    node_way: dict[int, int] = {}
    ends: list[int] = []
    for index, (way_nodes, way_coords) in enumerate(ways):
        nodes = [(node, (point[0], point[1])) for node, point in zip(way_nodes, way_coords)]
        for node, point in nodes:
            if node not in graph.coords:
                graph.coords[node] = point
                graph.grid.setdefault(_cell(*point), []).append(node)
                node_way[node] = index
        for (a, _), (b, _) in zip(nodes, nodes[1:]):
            graph._connect(a, b)
        if nodes:
            ends += [nodes[0][0], nodes[-1][0]]

    # Lücken schließen: Sackgassen-Enden mit dem nächsten Knoten eines anderen Wegs verbinden
    for end in ends:
        if len(graph.edges.get(end, {})) != 1:
            continue
        nearby = graph.nearby(*graph.coords[end], GAP_KM)
        others = [(km, node) for km, node in nearby if node_way[node] != node_way[end]]
        if others:
            graph._connect(end, min(others)[1])
    return graph


def sample(points: list[Point], spacing_km: float) -> list[Point]:
    """Punkte im Abstand `spacing_km` entlang des Pfads, ohne Anfang und Ende."""
    samples = []
    travelled = 0.0
    for a, b in zip(points, points[1:]):
        travelled += haversine_km(*a, *b)
        if travelled >= spacing_km:
            samples.append(b)
            travelled = 0.0
    if samples and samples[-1] == points[-1]:
        samples.pop()
    return samples


@dataclass
class Guide:
    """Dichte Wegpunktliste für den Router und wie viel davon auf dem Fernweg liegt."""

    waypoints: list[Point]
    on_route_km: float
    off_route_legs: list[int]  # 1-basierte Nummern der Teilstücke, die nicht dem Fernweg folgen
    off_route_reasons: list[str] = field(default_factory=list)  # je Teilstück abseits ein Grund, für den Agenten


def guide(graph: RouteGraph, waypoints: list[Point]) -> Guide:
    snaps = [graph.snap(p) for p in waypoints]
    dense = [waypoints[0]]
    on_route_km = 0.0
    off_route_legs = []
    reasons = []
    for leg, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
        path = None
        start, goal = snaps[leg], snaps[leg + 1]
        far = [str(i + 1) for i in (leg, leg + 1) if snaps[i] is None]
        if far:
            reasons.append(f"Teilstück {leg + 1}: Wegpunkt {' und '.join(far)} weiter als {MAX_SNAP_KM:g} km vom Fernweg")
        else:
            path = graph.shortest_path(start[0], goal[0])
            straight = haversine_km(*a, *b)
            if path is None:
                reasons.append(f"Teilstück {leg + 1}: keine Verbindung auf dem Fernweg")
            elif path[1] > DETOUR_LIMIT * straight + 2:
                reasons.append(
                    f"Teilstück {leg + 1}: auf dem Fernweg {path[1]:.0f} km bei {straight:.0f} km Luftlinie (Umweg)"
                )
                path = None
        if path:
            points, km = path
            if start[1] > SNAP_POINT_MIN_KM:
                dense.append(points[0])
            dense += sample(points, SAMPLE_SPACING_KM)
            if goal[1] > SNAP_POINT_MIN_KM:
                dense.append(points[-1])
            on_route_km += km
        else:
            off_route_legs.append(leg + 1)
        dense.append(b)
    return Guide(dense, on_route_km, off_route_legs, reasons)


# Katalog-ID -> (Importzeitpunkt, Graph); ein neuer Import ersetzt den Graphen
_catalog_graphs: dict[int, tuple[Any, RouteGraph]] = {}


async def catalog_graph(ctx: RunContext[AgentDeps], name: str, transport: TransportMode) -> RouteGraph | None:
    """Graph eines importierten Fernwegs oder None, wenn der Katalog keinen passenden enthält."""
    route_types = ROUTE_TYPES[transport].split("|")
    columns = (LongDistanceRoute.id, LongDistanceRoute.name, LongDistanceRoute.aliases, LongDistanceRoute.imported_at)
    try:
        async with ctx.deps.session_factory() as session:
            rows = (await session.execute(select(*columns).where(LongDistanceRoute.route_type.in_(route_types)))).all()
            match = next((r for r in rows if catalog_matches(name, [r.name, *r.aliases])), None)
            if match is None:
                return None
            cached = _catalog_graphs.get(match.id)
            if cached and cached[0] == match.imported_at:
                return cached[1]
            ways = await session.scalar(select(LongDistanceRoute.ways).where(LongDistanceRoute.id == match.id))
    except SQLAlchemyError:
        return None  # z. B. Tabelle fehlt (Migration nicht ausgeführt) → live über Overpass
    graph = build_graph([match.name], [(nodes, [(lat, lon) for lat, lon in coords]) for nodes, coords in ways or []])
    _catalog_graphs[match.id] = (match.imported_at, graph)
    return graph


# (Name normalisiert, Fortbewegungsart, Bounding Box) -> (Ablaufzeitpunkt, Graph). Spätere Aufrufe derselben Tour
# (z. B. mit den gewählten Unterkünften als Wegpunkte) brauchen so keine Overpass-Abfrage mehr.
_graphs: dict[tuple[str, TransportMode, tuple[float, ...]], tuple[float, RouteGraph]] = {}


async def load_route_graph(
    ctx: RunContext[AgentDeps], name: str, transport: TransportMode, waypoints: list[Point]
) -> RouteGraph | str:
    """Graph des Fernwegs aus dem Katalog, sonst live aus Overpass im Bereich der Wegpunkte; Text mit
    Namensvorschlägen, wenn keine Relation passt.

    Raises:
        httpx.HTTPError: Overpass nicht erreichbar.
    """
    if graph := await catalog_graph(ctx, name, transport):
        return graph

    bbox = search_bbox(waypoints)
    key = (normalize(name), transport, bbox)
    cached = _graphs.get(key)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    data = await query_overpass(ctx, relation_query(transport, waypoints), timeout=OVERPASS_TIMEOUT_SECONDS)
    relations = [r for r in data.get("elements", []) if r.get("type") == "relation"]
    found = [r for r in relations if matches(name, r.get("tags", {}))]
    if not found:
        # Fernwege zuerst (international, national, regional, lokal), je Name einmal
        tags = [r.get("tags", {}) for r in relations]
        ranked = sorted((NETWORK_RANK.get(t.get("network", ""), 3), short_name(t)) for t in tags)
        candidates = list(dict.fromkeys(n for _, n in ranked if n))[:MAX_CANDIDATES]
        names = ", ".join(f"„{n}“" for n in candidates)
        return (
            f"Keine ausgeschilderte Route „{name}“ nahe den Wegpunkten gefunden (OpenStreetMap). "
            + (f"Routen dort: {names}. Erneut mit einem dieser Namen bzw. einem Teil davon aufrufen" if names else "")
            + " oder ohne route_name routen."
        )

    query = way_query(sorted(r["id"] for r in found), bbox)
    ways = await query_overpass(ctx, query, timeout=OVERPASS_TIMEOUT_SECONDS + 10, cache=False)
    names = sorted({short_name(r.get("tags", {})) for r in found} - {""})
    graph = build_graph(names, overpass_ways(ways))
    if len(_graphs) >= MAX_CACHED_GRAPHS:
        _graphs.pop(next(iter(_graphs)))
    _graphs[key] = (time.monotonic() + CACHE_SECONDS, graph)
    return graph
