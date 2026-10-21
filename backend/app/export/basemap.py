"""Hintergrundkarte für den PDF-Export: OpenStreetMap-Kacheln (Web-Mercator) zu einem Bild zusammengesetzt."""

import asyncio
import io
import logging
import math
from dataclasses import dataclass

import httpx
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)

TILE_SIZE = 256
MAX_ZOOM = 16
# Höchstbreite des Ausschnitts in Pixeln; bei 174 mm Kartenbreite sind das 80–160 dpi, Beschriftungen bleiben lesbar
TARGET_WIDTH_PX = 1100
PADDING = 0.08  # Rand um Route und Marker, Anteil der Kartenbreite
MAX_PARALLEL = 2  # OSM-Tile-Policy: höchstens 2 gleichzeitige Verbindungen
TILE_TIMEOUT_SECONDS = 10.0
MAX_LAT = 85.05112878
EARTH_CIRCUMFERENCE_M = 40_075_016.686
BACKGROUND = (242, 239, 233)  # Hintergrundfarbe des OSM-Kartenstils für fehlende Kacheln
LIGHTEN = 0.25  # Karte aufhellen, damit Route und Marker hervortreten
ATTRIBUTION = "© OpenStreetMap-Mitwirkende"

# Kachel-Cache über Anfragen hinweg, damit wiederholte Exporte die Tile-Server nicht erneut belasten
MAX_CACHED_TILES = 500
_cache: dict[tuple[int, int, int], bytes] = {}


def _world(lat: float, lon: float) -> tuple[float, float]:
    """Web-Mercator-Koordinaten bei Zoom 0 (0..TILE_SIZE)."""
    lat_rad = math.radians(max(-MAX_LAT, min(MAX_LAT, lat)))
    x = (lon + 180) / 360 * TILE_SIZE
    y = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * TILE_SIZE
    return x, y


@dataclass(frozen=True)
class Viewport:
    """Kartenausschnitt in Weltpixeln der Zoomstufe `zoom`."""

    zoom: int
    left: float
    top: float
    width: float
    height: float

    def project(self, lat: float, lon: float) -> tuple[float, float]:
        """Position im Ausschnitt als Anteil von Breite und Höhe (0..1)."""
        x, y = _world(lat, lon)
        scale = 2**self.zoom
        return (x * scale - self.left) / self.width, (y * scale - self.top) / self.height

    def meters_per_pixel(self) -> float:
        center_y = (self.top + self.height / 2) / 2**self.zoom
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * center_y / TILE_SIZE))))
        return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (TILE_SIZE * 2**self.zoom)


def fit_viewport(points: list[tuple[float, float]], aspect: float) -> Viewport:
    """Ausschnitt mit Seitenverhältnis `aspect` (Breite/Höhe), der alle Punkte samt Rand enthält.

    Die Zoomstufe ist die größte, bei der der Ausschnitt höchstens TARGET_WIDTH_PX breit ist.
    """
    world = [_world(lat, lon) for lat, lon in points]
    xs, ys = [p[0] for p in world], [p[1] for p in world]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    width = max(span_x, span_y * aspect) / (1 - 2 * PADDING)
    width = max(width, TARGET_WIDTH_PX / 2**MAX_ZOOM)  # einzelner Punkt: nicht unendlich weit hineinzoomen
    zoom = max(0, min(MAX_ZOOM, math.floor(math.log2(TARGET_WIDTH_PX / width))))
    scale = 2**zoom
    center_x, center_y = (min(xs) + max(xs)) / 2 * scale, (min(ys) + max(ys)) / 2 * scale
    width_px = width * scale
    height_px = width_px / aspect
    return Viewport(zoom, center_x - width_px / 2, center_y - height_px / 2, width_px, height_px)


async def _fetch_tile(
    http: httpx.AsyncClient, semaphore: asyncio.Semaphore, url: str, zoom: int, x: int, y: int
) -> bytes | None:
    key = (zoom, x, y)
    if key in _cache:
        return _cache[key]
    async with semaphore:
        try:
            response = await http.get(url.format(z=zoom, x=x, y=y))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Kartenkachel %s nicht geladen: %s", key, exc)
            return None
    if len(_cache) >= MAX_CACHED_TILES:
        _cache.pop(next(iter(_cache)))
    _cache[key] = response.content
    return response.content


async def render_basemap(viewport: Viewport) -> Image.Image | None:
    """Kacheln laden und auf den Ausschnitt zuschneiden. None, wenn keine Kachel geladen werden konnte."""
    settings = get_settings()
    if not settings.map_tile_url:
        return None

    count = 2**viewport.zoom
    columns = range(math.floor(viewport.left / TILE_SIZE), math.floor((viewport.left + viewport.width) / TILE_SIZE) + 1)
    rows = range(
        max(0, math.floor(viewport.top / TILE_SIZE)),
        min(count - 1, math.floor((viewport.top + viewport.height) / TILE_SIZE)) + 1,
    )
    tiles = [(x, y) for y in rows for x in columns]

    semaphore = asyncio.Semaphore(MAX_PARALLEL)
    headers = {"User-Agent": settings.nominatim_user_agent}
    async with httpx.AsyncClient(headers=headers, timeout=TILE_TIMEOUT_SECONDS) as http:
        # x modulo count: Ausschnitte über die Datumsgrenze hinweg
        images = await asyncio.gather(
            *(_fetch_tile(http, semaphore, settings.map_tile_url, viewport.zoom, x % count, y) for x, y in tiles)
        )
    if not any(images):
        return None

    canvas = Image.new("RGB", (math.ceil(viewport.width), math.ceil(viewport.height)), BACKGROUND)
    for (x, y), data in zip(tiles, images):
        if data:
            tile = Image.open(io.BytesIO(data)).convert("RGB")
            canvas.paste(tile, (round(x * TILE_SIZE - viewport.left), round(y * TILE_SIZE - viewport.top)))
    return Image.blend(canvas, Image.new("RGB", canvas.size, "white"), LIGHTEN)
