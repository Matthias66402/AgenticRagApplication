"""Importiert den Verlauf von Fernwegen aus OpenStreetMap in die Tabelle long_distance_routes.

Quelle ist Waymarked Trails (aufbereitete OSM-Routen, schnell); nur wenn es nicht erreichbar ist, Overpass.
Der Katalog steht in long_distance_routes.json (Name, weitere Namen, OSM-Relationen). Beim Planen mit `route_name`
nutzt `plan_route` diese Daten sofort, ohne Live-Abfrage.

Aufruf:
    python -m app.seed.long_distance_routes                       # alle noch fehlenden Fernwege
    python -m app.seed.long_distance_routes "Ostseeküsten-Radweg" # nur diese (Name oder weiterer Name)
    python -m app.seed.long_distance_routes --force               # vorhandene neu importieren (aktueller OSM-Stand)
    python -m app.seed.long_distance_routes --list                # Katalog und Importstand anzeigen
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from app.agent.plan_schema import TransportMode
from app.agent.tools.common import haversine_km
from app.agent.tools.named_route import WAYMARKED_URLS, normalize, waymarked_relation_ways
from app.config import get_settings
from app.db import SessionLocal, engine
from app.models import LongDistanceRoute

CATALOG = Path(__file__).with_name("long_distance_routes.json")
OVERPASS_TIMEOUT_SECONDS = 300
ATTEMPTS = 5
RETRY_WAIT_SECONDS = 30
RETRYABLE_STATUS = {429, 502, 503, 504}
# Instanz von Waymarked Trails je route_type des Katalogs
WAYMARKED_BASE = {
    "bicycle": WAYMARKED_URLS[TransportMode.cycling][0],
    "mtb": WAYMARKED_URLS[TransportMode.mtb][0],
    "hiking": WAYMARKED_URLS[TransportMode.hiking][0],
    "foot": WAYMARKED_URLS[TransportMode.hiking][0],
}


def load_catalog() -> list[dict[str, Any]]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def ways_query(relation_ids: list[int]) -> str:
    # Superroute → Abschnitte → ggf. Unterabschnitte; dann alle Wege dieser Relationen
    ids = ",".join(map(str, relation_ids))
    return (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_SECONDS}];\n"
        f"rel(id:{ids});\n(._; rel(r););\n(._; rel(r););\n(._; rel(r););\nway(r);\nout skel geom;"
    )


async def fetch_ways(http: httpx.AsyncClient, route_type: str, relation_ids: list[int]) -> tuple[list[list[Any]], str]:
    """Wege als [[Knoten-IDs], [[lat, lon], ...]] und Quelle; aus Waymarked Trails, sonst Overpass."""
    try:
        ways = await waymarked_relation_ways(
            http, get_settings().nominatim_user_agent, WAYMARKED_BASE[route_type], relation_ids
        )
        return [[nodes, [list(p) for p in coords]] for nodes, coords in ways if len(nodes) >= 2], "Waymarked Trails"
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        print(f"    Waymarked Trails: {type(exc).__name__}, weiter mit Overpass")
    return await fetch_overpass_ways(http, relation_ids), "Overpass"


async def fetch_overpass_ways(http: httpx.AsyncClient, relation_ids: list[int]) -> list[list[Any]]:
    """Wege aus Overpass; wartet bei überlasteten Servern und versucht es erneut."""
    settings = get_settings()
    urls = settings.overpass_url_list
    for attempt in range(ATTEMPTS):
        url = urls[attempt % len(urls)]
        try:
            response = await http.post(
                url,
                data={"data": ways_query(relation_ids)},
                headers={"User-Agent": settings.nominatim_user_agent},
                timeout=OVERPASS_TIMEOUT_SECONDS + 30,
            )
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRYABLE_STATUS or attempt == ATTEMPTS - 1:
                raise
            print(f"    {url}: HTTP {exc.response.status_code}, neuer Versuch in {RETRY_WAIT_SECONDS} s")
        except httpx.TransportError as exc:
            if attempt == ATTEMPTS - 1:
                raise
            print(f"    {url}: {type(exc).__name__}, neuer Versuch in {RETRY_WAIT_SECONDS} s")
        await asyncio.sleep(RETRY_WAIT_SECONDS)

    ways = []
    for element in response.json().get("elements", []):
        if element.get("type") != "way":
            continue
        nodes, coords = [], []
        for node, point in zip(element.get("nodes", []), element.get("geometry", [])):
            if point:
                nodes.append(node)
                coords.append([round(point["lat"], 6), round(point["lon"], 6)])
        if len(nodes) >= 2:
            ways.append([nodes, coords])
    return ways


def ways_length_km(ways: list[list[Any]]) -> float:
    """Summe aller Wege (inkl. Varianten und beider Richtungen, daher meist länger als der Fernweg selbst)."""
    return sum(haversine_km(*a, *b) for _, coords in ways for a, b in zip(coords, coords[1:]))


def select_entries(catalog: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    if not names:
        return catalog
    wanted = {normalize(n) for n in names}
    selected = [e for e in catalog if wanted & {normalize(n) for n in [e["name"], *e["aliases"]]}]
    unknown = wanted - {normalize(n) for e in selected for n in [e["name"], *e["aliases"]]}
    if unknown:
        raise SystemExit(f"Nicht im Katalog ({CATALOG.name}): {', '.join(sorted(unknown))}")
    return selected


async def import_routes(names: list[str], force: bool) -> None:
    entries = select_entries(load_catalog(), names)
    async with httpx.AsyncClient() as http:
        for entry in entries:
            async with SessionLocal() as session:
                route = await session.scalar(select(LongDistanceRoute).where(LongDistanceRoute.name == entry["name"]))
                if route and not force:
                    print(f"  übersprungen: {entry['name']} (bereits importiert, --force importiert neu)")
                    continue
                print(f"  lade {entry['name']} (Relation {', '.join(map(str, entry['relation_ids']))}) …")
                try:
                    ways, source = await fetch_ways(http, entry["route_type"], entry["relation_ids"])
                except httpx.HTTPError as exc:
                    print(f"  FEHLER {entry['name']}: {exc!r}")
                    continue
                if not ways:
                    print(f"  FEHLER {entry['name']}: keine Wege gefunden (Relation-IDs prüfen)")
                    continue
                route = route or LongDistanceRoute(name=entry["name"])
                route.aliases = entry["aliases"]
                route.route_type = entry["route_type"]
                route.relation_ids = entry["relation_ids"]
                route.ways = ways
                route.length_km = round(ways_length_km(ways), 1)
                session.add(route)
                await session.commit()
                print(f"  importiert: {entry['name']} ({len(ways)} Wege, {route.length_km:g} km Wege, {source})")
    await engine.dispose()


async def list_routes() -> None:
    async with SessionLocal() as session:
        imported = {r.name: r for r in await session.scalars(select(LongDistanceRoute))}
    for entry in load_catalog():
        route = imported.get(entry["name"])
        state = f"importiert {route.imported_at:%d.%m.%Y}, {route.length_km:g} km Wege" if route else "nicht importiert"
        print(f"  {entry['name']} ({', '.join(entry['aliases'])}): {state}")
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="Fernwege aus dem Katalog (Standard: alle)")
    parser.add_argument("--force", action="store_true", help="bereits importierte Fernwege neu importieren")
    parser.add_argument("--list", action="store_true", help="Katalog und Importstand anzeigen")
    args = parser.parse_args()
    asyncio.run(list_routes() if args.list else import_routes(args.names, args.force))


if __name__ == "__main__":
    main()
