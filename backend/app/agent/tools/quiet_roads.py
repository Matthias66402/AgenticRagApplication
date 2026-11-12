"""Hauptstraßen erkennen und meiden: Abschnitte einer Route auf oder neben Landes-, Kreis- und Bundesstraßen.

Die Rad-Profile der Router werten straßenbegleitende Radwege als ruhig. In Deutschland sind sie meist eigene Wege
(highway=path/cycleway) ohne Hinweis auf die Straße daneben: Die Rundtour Oldenburg – Bad Zwischenahn – Westerstede –
Rastede führte mit BRouter trekking auf 31 von 62 km neben L 815, K 131 & Co., mit "fastbike-verylowtraffic" sogar auf
55 km. Erkennbar ist das nur geometrisch: Hauptstraßen aus OpenStreetMap (Overpass) laden und messen, wie viel der
Route in ihrer Nähe verläuft. Zum Meiden bekommt BRouter diese Abschnitte als gewichtete Sperrflächen (`polygons`)
und routet neu; Querungen bleiben möglich, nur das lange Entlangfahren wird teuer.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.tools.accommodation import query_overpass

MAIN_ROAD_PATTERN = "^(trunk|primary|secondary|tertiary)(_link)?$"
NEAR_M = 30.0  # Abstand Route–Straßenachse, ab dem ein Abschnitt als "an der Hauptstraße" gilt
MIN_RUN_M = 300.0  # kürzere Stücke (Querungen, Ortsdurchfahrten bis zur nächsten Abzweigung) nicht meiden
CORRIDOR_HALF_WIDTH_M = 35.0
CORRIDOR_SIMPLIFY_M = 25.0
# Kosten je Meter in der Sperrfläche: 10 bzw. 100 änderten die Testroute kaum, ab ~1000 weicht BRouter aus
PENALTY_WEIGHT = 100000
MAX_POLYGON_CHARS = 15000  # BRouter (nginx) lehnt URLs ab ~38 kB ab
TILE_DEG = 0.1
TILE_MARGIN_DEG = 0.001  # ~100 m, damit Straßen knapp jenseits der Kachelgrenze mitkommen
MAX_TILES = 80  # Route samt Ausweichrouten; ~ eine Tagesetappe bis 150 km, längere Routen etappenweise
MAX_CACHED_TILES = 2000
OVERPASS_TIMEOUT = 60
GRID_M = 200.0
METERS_PER_DEGREE = 111_320.0

Road = tuple[str, list[tuple[float, float]]]  # (Bezeichnung, [(lat, lon), ...])

# Kachel (lat_index, lon_index) -> Hauptstraßen; Straßennetz ändert sich kaum, daher ohne Ablauf
_tiles: dict[tuple[int, int], list[Road]] = {}


class TooLong(Exception):
    pass


def _tile(lat: float, lon: float) -> tuple[int, int]:
    return math.floor(lat / TILE_DEG), math.floor(lon / TILE_DEG)


def _tiles_for(geometry: list[list[float]]) -> set[tuple[int, int]]:
    return {_tile(p[0], p[1]) for p in geometry}


def _label(tags: dict[str, str]) -> str:
    return tags.get("ref") or tags.get("name") or ""


def _query(tiles: list[tuple[int, int]]) -> str:
    boxes = []
    for lat_i, lon_i in tiles:
        south, west = lat_i * TILE_DEG - TILE_MARGIN_DEG, lon_i * TILE_DEG - TILE_MARGIN_DEG
        north, east = (lat_i + 1) * TILE_DEG + TILE_MARGIN_DEG, (lon_i + 1) * TILE_DEG + TILE_MARGIN_DEG
        boxes.append(f'way[highway~"{MAIN_ROAD_PATTERN}"]({south:.3f},{west:.3f},{north:.3f},{east:.3f});')
    return f"[out:json][timeout:{OVERPASS_TIMEOUT - 5}];\n(\n{chr(10).join(boxes)}\n);\nout geom qt;"


async def _load_tiles(ctx: RunContext[AgentDeps], tiles: set[tuple[int, int]]) -> None:
    missing = sorted(tiles - _tiles.keys())
    if not missing:
        return
    if len(_tiles) + len(missing) > MAX_CACHED_TILES:
        _tiles.clear()
        missing = sorted(tiles)
    data = await query_overpass(ctx, _query(missing), timeout=OVERPASS_TIMEOUT, cache=False)
    found: dict[tuple[int, int], list[Road]] = {tile: [] for tile in missing}
    for element in data.get("elements", []):
        points = [(p["lat"], p["lon"]) for p in element.get("geometry", []) if p]
        if len(points) < 2:
            continue
        road = (_label(element.get("tags", {})), points)
        for tile in {_tile(lat, lon) for lat, lon in points} & found.keys():
            found[tile].append(road)
    _tiles.update(found)


@dataclass
class Run:
    """Zusammenhängender Abschnitt der Route an einer Hauptstraße."""

    points: list[tuple[float, float]]  # lokale Meter (x, y)
    length_m: float
    roads: dict[str, float] = field(default_factory=dict)  # Bezeichnung -> Meter


@dataclass
class Measurement:
    distance_m: float
    along_m: float
    runs: list[Run]

    def sections(self, limit: int = 6) -> list[dict[str, Any]]:
        """Längste Abschnitte nach Straße, z. B. [{"road": "L 815", "km": 2.1}]."""
        per_road: dict[str, float] = defaultdict(float)
        for run in self.runs:
            for road, meters in run.roads.items():
                per_road[road or "ohne Nummer"] += meters
        ranked = sorted(per_road.items(), key=lambda item: -item[1])[:limit]
        return [{"road": road, "km": round(meters / 1000, 1)} for road, meters in ranked if meters >= 100]


class MainRoads:
    """Hauptstraßen um eine Route mit Raster-Index in lokalen Metern (äquirektangulär um die Startbreite)."""

    def __init__(self, lat0: float) -> None:
        self.cos = math.cos(math.radians(lat0))
        self.grid: dict[tuple[int, int], list[tuple[tuple[float, float], tuple[float, float], str]]] = defaultdict(list)
        self.loaded: set[tuple[int, int]] = set()
        self._seen: set[int] = set()

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        return lon * self.cos * METERS_PER_DEGREE, lat * METERS_PER_DEGREE

    def latlon(self, x: float, y: float) -> tuple[float, float]:
        return y / METERS_PER_DEGREE, x / self.cos / METERS_PER_DEGREE

    async def cover(self, ctx: RunContext[AgentDeps], geometry: list[list[float]], ring: bool = False) -> None:
        """Lädt die Hauptstraßen aller Kacheln, durch die die Route führt.

        `ring`: auch die Nachbarkacheln, in die Ausweichrouten meist führen. Eine Overpass-Abfrage dauerte im Test
        ~40 s, fast unabhängig von der Zahl der Kacheln (Warteschlange), eine zweite für die Ausweichroute ebenso.
        """
        tiles = _tiles_for(geometry)
        if len(tiles | self.loaded) > MAX_TILES:
            raise TooLong
        if ring:
            ringed = {(lat + dlat, lon + dlon) for lat, lon in tiles for dlat in (-1, 0, 1) for dlon in (-1, 0, 1)}
            if len(ringed | self.loaded) <= MAX_TILES:
                tiles = ringed
        await _load_tiles(ctx, tiles)
        for tile in tiles - self.loaded:
            for road in _tiles[tile]:
                if id(road) not in self._seen:
                    self._seen.add(id(road))
                    self._add(road)
        self.loaded |= tiles

    def _add(self, road: Road) -> None:
        label, points = road
        coords = [self.xy(lat, lon) for lat, lon in points]
        for a, b in zip(coords, coords[1:]):
            for gx in range(int(min(a[0], b[0]) // GRID_M) - 1, int(max(a[0], b[0]) // GRID_M) + 2):
                for gy in range(int(min(a[1], b[1]) // GRID_M) - 1, int(max(a[1], b[1]) // GRID_M) + 2):
                    self.grid[(gx, gy)].append((a, b, label))

    def nearest(self, point: tuple[float, float]) -> str | None:
        """Bezeichnung der Hauptstraße innerhalb NEAR_M (leer ohne ref/name) oder None."""
        best, label = NEAR_M, None
        for a, b, road in self.grid.get((int(point[0] // GRID_M), int(point[1] // GRID_M)), []):
            distance = _segment_distance(point, a, b)
            if distance < best:
                best, label = distance, road
        return label

    def measure(self, geometry: list[list[float]]) -> Measurement:
        runs: list[Run] = []
        current: Run | None = None
        total = along = 0.0
        coords = [self.xy(p[0], p[1]) for p in geometry]
        for a, b in zip(coords, coords[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            total += length
            road = self.nearest(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            if road is None:
                if current is not None and current.length_m >= MIN_RUN_M:
                    runs.append(current)
                current = None
                continue
            along += length
            if current is None:
                current = Run(points=[a], length_m=0.0)
            current.points.append(b)
            current.length_m += length
            current.roads[road] = current.roads.get(road, 0.0) + length
        if current is not None and current.length_m >= MIN_RUN_M:
            runs.append(current)
        return Measurement(distance_m=total, along_m=along, runs=runs)

    def corridor(self, run: Run) -> list[tuple[float, float]]:
        """Schmales Polygon um einen Abschnitt als (lat, lon), vereinfacht für eine kurze BRouter-URL."""
        points = _simplify(run.points, CORRIDOR_SIMPLIFY_M)
        left, right = [], []
        for i, (x, y) in enumerate(points):
            ax, ay = points[max(i - 1, 0)]
            bx, by = points[min(i + 1, len(points) - 1)]
            dx, dy = bx - ax, by - ay
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm * CORRIDOR_HALF_WIDTH_M, dx / norm * CORRIDOR_HALF_WIDTH_M
            left.append((x + nx, y + ny))
            right.append((x - nx, y - ny))
        return [self.latlon(x, y) for x, y in left + right[::-1]]


def _segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _simplify(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Douglas-Peucker (iterativ) auf lokalen Metern."""
    if len(points) <= 2:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        best, index = 0.0, -1
        for i in range(start + 1, end):
            distance = _segment_distance(points[i], points[start], points[end])
            if distance > best:
                best, index = distance, i
        if best > tolerance:
            keep[index] = True
            stack.extend([(start, index), (index, end)])
    return [p for p, k in zip(points, keep) if k]


def polygons_param(polygons: list[list[tuple[float, float]]]) -> str:
    """BRouter-Parameter `polygons`: lon,lat,...,Gewicht je Polygon, mit | getrennt."""
    return "|".join(
        ",".join(f"{lon:.5f},{lat:.5f}" for lat, lon in polygon) + f",{PENALTY_WEIGHT}" for polygon in polygons
    )
