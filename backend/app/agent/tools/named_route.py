"""Verlauf ausgeschilderter Fernwege (OSM-Routenrelationen, z. B. Ostseeküsten-Radweg) als Leitlinie fürs Routing.

Quelle ist zuerst der vorab importierte Katalog (Tabelle long_distance_routes, app.seed.long_distance_routes): sofort
verfügbar und mit weiteren Namen wie "Ostseeradweg". Fehlt der Fernweg dort, liefert Waymarked Trails die Routen nahe
den Wegpunkten und den Verlauf der passenden (meist in unter einer Sekunde, aufbereitete OSM-Daten). Ist Waymarked
Trails nicht erreichbar, sucht Overpass live die passenden Relationen, die übrigen Abschnitte derselben Superroute
und deren Wege (oft ein bis zwei Minuten, die öffentlichen Server sind häufig überlastet).
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

import httpx
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

# Waymarked Trails (https://waymarkedtrails.org): je Fortbewegungsart eine Instanz, MTB-Touren auch auf Radrouten
WAYMARKED_URLS: dict[TransportMode, list[str]] = {
    TransportMode.cycling: ["https://cycling.waymarkedtrails.org/api/v1"],
    TransportMode.mtb: ["https://mtb.waymarkedtrails.org/api/v1", "https://cycling.waymarkedtrails.org/api/v1"],
    TransportMode.hiking: ["https://hiking.waymarkedtrails.org/api/v1"],
}
WAYMARKED_GROUP_RANK = {"INT": 0, "NAT": 1, "REG": 2, "LOC": 3}
WAYMARKED_TIMEOUT_SECONDS = 60  # Verlauf sehr langer Routen (EuroVelo 12: 12 MB) braucht einige Sekunden
MAX_AREA_QUERIES = 6  # Gebietssuchen je Fernweg: Start, Ziel und gleichmäßig verteilte Zwischenpunkte
EARTH_RADIUS_M = 6378137.0  # Web-Mercator (EPSG:3857), Koordinatensystem der Waymarked-Trails-API

GRID_DEG = 0.01  # Zellgröße des Suchrasters (~1 km)
# Wegpunkte bis zu dieser Entfernung werden normal zum Fernweg hin angebunden, das Teilstück folgt dann dem Fernweg
# (Tønder und Hurup liegen 8-9 km von der Vestkystruten); weiter entfernte Teilstücke werden direkt geroutet
MAX_SNAP_KM = 15.0
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


def to_mercator(lat: float, lon: float) -> tuple[float, float]:
    return math.radians(lon) * EARTH_RADIUS_M, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * EARTH_RADIUS_M


def from_mercator(x: float, y: float) -> Point:
    lat = math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2)
    return round(lat, 6), round(math.degrees(x / EARTH_RADIUS_M), 6)


def area_bbox(point: Point, radius_km: float) -> str:
    """Quadrat um den Punkt in Web-Mercator (West, Süd, Ost, Nord) für `list/by_area`."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(point[0])), 0.1))
    west, south = to_mercator(point[0] - dlat, point[1] - dlon)
    east, north = to_mercator(point[0] + dlat, point[1] + dlon)
    return f"{west:.0f},{south:.0f},{east:.0f},{north:.0f}"


def spread(points: list[Point], count: int) -> list[Point]:
    """Start, Ziel und gleichmäßig verteilte Zwischenpunkte, höchstens `count`."""
    if len(points) <= count:
        return points
    return list(dict.fromkeys(points[round(i * (len(points) - 1) / (count - 1))] for i in range(count)))


def waymarked_ways(details: dict[str, Any], ways: dict[int, Way]) -> None:
    """Wege aus `details/relation/{id}` in `ways` (nach Weg-ID, Abschnitte mehrerer Relationen nur einmal).

    Die API liefert keine Knoten-IDs; gemeinsame Knoten benachbarter Wege haben aber dieselben Koordinaten.
    """

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            geometry = node.get("geometry")
            if isinstance(geometry, dict) and geometry.get("type") == "LineString" and "id" in node:
                coords = [from_mercator(x, y) for x, y in geometry["coordinates"]]
                ways.setdefault(node["id"], ([hash(p) for p in coords], coords))
                return
            for key, value in node.items():
                if key != "tags":
                    walk(value)

    walk(details.get("route"))


async def waymarked_get(
    http: httpx.AsyncClient, user_agent: str, url: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    response = await http.get(url, params=params, headers={"User-Agent": user_agent}, timeout=WAYMARKED_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


async def waymarked_relation_ways(
    http: httpx.AsyncClient, user_agent: str, base: str, relation_ids: list[int]
) -> list[Way]:
    """Wege der Relationen (samt Abschnitten) aus Waymarked Trails; auch für den Import des Katalogs.

    Raises:
        httpx.HTTPError: Waymarked Trails nicht erreichbar.
    """
    ways: dict[int, Way] = {}
    for relation_id in relation_ids:
        waymarked_ways(await waymarked_get(http, user_agent, f"{base}/details/relation/{relation_id}"), ways)
    return list(ways.values())


def not_found_message(name: str, ranked: list[tuple[int, str]]) -> str:
    """Hinweis mit den Routen nahe den Wegpunkten, Fernwege zuerst (international, national, regional, lokal)."""
    candidates = list(dict.fromkeys(n for _, n in sorted(ranked) if n))[:MAX_CANDIDATES]
    names = ", ".join(f"„{n}“" for n in candidates)
    return (
        f"Keine ausgeschilderte Route „{name}“ nahe den Wegpunkten gefunden (OpenStreetMap). "
        + (f"Routen dort: {names}. Erneut mit einem dieser Namen bzw. einem Teil davon aufrufen" if names else "")
        + " oder ohne route_name routen."
    )


async def waymarked_graph(
    ctx: RunContext[AgentDeps], name: str, transport: TransportMode, waypoints: list[Point]
) -> RouteGraph | str:
    """Graph aus Waymarked Trails: Routen im Umkreis der Wegpunkte, davon die passenden mit ihrem Verlauf.

    `list/by_area` liefert nur die obersten Relationen (Superrouten mit allen Abschnitten).

    Raises:
        httpx.HTTPError: Waymarked Trails nicht erreichbar.
    """
    http, user_agent = ctx.deps.http, ctx.deps.settings.nominatim_user_agent
    near: dict[tuple[str, int], dict[str, Any]] = {}
    for base in WAYMARKED_URLS[transport]:
        for point in spread(waypoints, MAX_AREA_QUERIES):
            params = {"bbox": area_bbox(point, MAX_SNAP_KM), "limit": 100}
            for result in (await waymarked_get(http, user_agent, f"{base}/list/by_area", params)).get("results", []):
                near.setdefault((base, result["id"]), result)
    found = {key: r for key, r in near.items() if matches(name, {"name": r.get("name", ""), "ref": r.get("ref", "")})}
    if not found:
        return not_found_message(name, [(WAYMARKED_GROUP_RANK.get(r.get("group"), 4), short_name(r)) for r in near.values()])

    ways: list[Way] = []
    for base in dict.fromkeys(base for base, _ in found):
        ways += await waymarked_relation_ways(http, user_agent, base, [rid for b, rid in found if b == base])
    return build_graph(sorted({short_name(r) for r in found.values()} - {""}), ways)


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


async def catalog_overview(deps: AgentDeps) -> str:
    """Importierte Fernwege mit weiteren Namen für die Instructions; leer ohne Katalog.

    Ohne diese Liste rät das Modell Namen wie "North Sea Cycle Route", die erst eine langsame Live-Suche auslösen.
    """
    columns = (LongDistanceRoute.name, LongDistanceRoute.aliases, LongDistanceRoute.route_type)
    try:
        async with deps.session_factory() as session:
            rows = (await session.execute(select(*columns).order_by(LongDistanceRoute.name))).all()
    except SQLAlchemyError:
        return ""
    return "\n".join(
        f"- {r.name} ({r.route_type})" + (f", auch: {', '.join(r.aliases)}" if r.aliases else "") for r in rows
    )


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
        return None  # z. B. Tabelle fehlt (Migration nicht ausgeführt) → live über Waymarked Trails bzw. Overpass
    graph = build_graph([match.name], [(nodes, [(lat, lon) for lat, lon in coords]) for nodes, coords in ways or []])
    _catalog_graphs[match.id] = (match.imported_at, graph)
    return graph


async def overpass_graph(
    ctx: RunContext[AgentDeps], name: str, transport: TransportMode, waypoints: list[Point],
    bbox: tuple[float, float, float, float],
) -> RouteGraph | str:  # fmt: skip
    """Graph aus Overpass (Ersatz für Waymarked Trails): Relationen nahe den Wegpunkten, Wege in der Bounding Box.

    Raises:
        httpx.HTTPError: Overpass nicht erreichbar.
    """
    data = await query_overpass(ctx, relation_query(transport, waypoints), timeout=OVERPASS_TIMEOUT_SECONDS)
    relations = [r for r in data.get("elements", []) if r.get("type") == "relation"]
    found = [r for r in relations if matches(name, r.get("tags", {}))]
    if not found:
        tags = [r.get("tags", {}) for r in relations]
        return not_found_message(name, [(NETWORK_RANK.get(t.get("network", ""), 3), short_name(t)) for t in tags])

    query = way_query(sorted(r["id"] for r in found), bbox)
    ways = await query_overpass(ctx, query, timeout=OVERPASS_TIMEOUT_SECONDS + 10, cache=False)
    names = sorted({short_name(r.get("tags", {})) for r in found} - {""})
    return build_graph(names, overpass_ways(ways))


# (Name normalisiert, Fortbewegungsart, Bounding Box) -> (Ablaufzeitpunkt, Graph). Spätere Aufrufe derselben Tour
# (z. B. mit den gewählten Unterkünften als Wegpunkte) brauchen so keine erneute Abfrage mehr.
_graphs: dict[tuple[str, TransportMode, tuple[float, ...]], tuple[float, RouteGraph]] = {}


async def load_route_graph(
    ctx: RunContext[AgentDeps], name: str, transport: TransportMode, waypoints: list[Point]
) -> RouteGraph | str:
    """Graph des Fernwegs aus dem Katalog, sonst live aus Waymarked Trails bzw. Overpass im Bereich der Wegpunkte;
    Text mit Namensvorschlägen, wenn keine Relation passt.

    Raises:
        httpx.HTTPError: Weder Waymarked Trails noch Overpass erreichbar.
    """
    if graph := await catalog_graph(ctx, name, transport):
        return graph

    bbox = search_bbox(waypoints)
    key = (normalize(name), transport, bbox)
    cached = _graphs.get(key)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    try:
        graph = await waymarked_graph(ctx, name, transport, waypoints)
    except (httpx.HTTPError, ValueError, KeyError):  # nicht erreichbar oder unerwartete Antwort
        graph = await overpass_graph(ctx, name, transport, waypoints, bbox)
    if isinstance(graph, str):
        return graph
    if len(_graphs) >= MAX_CACHED_GRAPHS:
        _graphs.pop(next(iter(_graphs)))
    _graphs[key] = (time.monotonic() + CACHE_SECONDS, graph)
    return graph
