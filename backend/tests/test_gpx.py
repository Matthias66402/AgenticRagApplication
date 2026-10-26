import xml.etree.ElementTree as ET

import httpx
from sqlalchemy import select

from app.agent.tools.routing import simplify_track
from app.db import SessionLocal
from app.export.filename import export_filename
from app.export.gpx import build_gpx
from app.models import Adventure, AdventureStatus, User
from tests.conftest import auth_headers, create_user

NS = {"g": "http://www.topografix.com/GPX/1/1"}

PLAN = {
    "title": "Hüttentour Garmisch → Innsbruck",
    "summary": "Zwei Tage über die Berge.",
    "transport": "hiking",
    "start_date": "2026-11-06",
    "duration_days": 2,
    "start": {"name": "Garmisch-Partenkirchen, Bayern", "lat": 47.49, "lon": 11.09},
    "destination": {"name": "Innsbruck", "lat": 47.26, "lon": 11.39},
    "accommodation_type": "hut",
    "stages": [
        {
            "day": 1, "date": "2026-11-06",
            "start": {"name": "Garmisch", "lat": 47.49, "lon": 11.09},
            "end": {"name": "Mittenwald", "lat": 47.44, "lon": 11.26},
            "distance_km": 20.5, "ascent_m": 900, "descent_m": 700,
            "accommodation": {"name": "Testhütte", "type": "hut", "lat": 47.45, "lon": 11.25, "website": "https://huette.example"},
        },
        {
            "day": 2, "date": "2026-11-07",
            "start": {"name": "Mittenwald", "lat": 47.44, "lon": 11.26},
            "end": {"name": "Innsbruck", "lat": 47.26, "lon": 11.39},
            "distance_km": 25.0,
        },
    ],
    "route_ids": ["r-1"],
    "routes": {"r-1": [[47.49, 11.09, 708.0], [47.44, 11.26, 912.5], [47.26, 11.39, 574.0]]},
    "packing_tips": [], "safety_notes": [], "sources": [],
}  # fmt: skip


async def store_adventure(owner_email: str, plan: dict | None) -> int:
    async with SessionLocal() as session:
        owner = await session.scalar(select(User).where(User.email == owner_email))
        adventure = Adventure(
            owner_id=owner.id, title=PLAN["title"], plan=plan, agent_history=[],
            status=AdventureStatus.planned if plan else AdventureStatus.draft,
        )  # fmt: skip
        session.add(adventure)
        await session.commit()
        return adventure.id


async def test_gpx_download(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    adventure_id = await store_adventure("user@example.com", PLAN)
    response = await client.get(f"/api/adventures/{adventure_id}/gpx", headers=user_headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/gpx+xml")
    assert 'filename="huttentour-garmisch-innsbruck.gpx"' in response.headers["content-disposition"]
    assert response.headers["x-gpx-has-track"] == "true"

    root = ET.fromstring(response.content)
    assert root.get("version") == "1.1"
    names = [w.findtext("g:name", namespaces=NS) for w in root.findall("g:wpt", NS)]
    assert names == ["Start: Garmisch-Partenkirchen", "Tag 1: Mittenwald", "Testhütte", "Tag 2: Innsbruck"]
    points = root.findall("g:trk/g:trkseg/g:trkpt", NS)
    assert len(points) == 3
    assert points[1].get("lat") == "47.440000"
    assert points[1].findtext("g:ele", namespaces=NS) == "912.5"



async def test_gpx_per_day(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    adventure_id = await store_adventure("user@example.com", PLAN)
    response = await client.get(f"/api/adventures/{adventure_id}/gpx?day=2", headers=user_headers)

    assert response.status_code == 200
    assert 'filename="huttentour-garmisch-innsbruck_2.gpx"' in response.headers["content-disposition"]
    assert response.headers["x-gpx-has-track"] == "true"
    root = ET.fromstring(response.content)
    assert root.findtext("g:metadata/g:name", namespaces=NS) == "Hüttentour Garmisch → Innsbruck – Tag 2"
    names = [w.findtext("g:name", namespaces=NS) for w in root.findall("g:wpt", NS)]
    assert names == ["Start: Mittenwald", "Tag 2: Innsbruck"]
    # Der Gesamttrack wird am Etappenziel von Tag 1 geteilt
    points = root.findall("g:trk/g:trkseg/g:trkpt", NS)
    assert [p.get("lat") for p in points] == ["47.440000", "47.260000"]

    day_one = ET.fromstring(build_gpx(PLAN, day=1))
    assert [w.findtext("g:name", namespaces=NS) for w in day_one.findall("g:wpt", NS)] == [
        "Start: Garmisch", "Tag 1: Mittenwald", "Testhütte",
    ]  # fmt: skip
    assert len(day_one.findall("g:trk/g:trkseg/g:trkpt", NS)) == 2

    assert (await client.get(f"/api/adventures/{adventure_id}/gpx?day=3", headers=user_headers)).status_code == 404


def test_gpx_joins_partial_routes_into_one_track() -> None:
    # Eine Teilroute je Tag ergibt trotzdem nur einen Track, sonst bietet Komoot die Tage einzeln zur Auswahl an
    routes = {"r-1": [[47.49, 11.09, 708.0], [47.44, 11.26, 912.5]], "r-2": [[47.44, 11.26, 912.5], [47.26, 11.39, 574.0]]}
    root = ET.fromstring(build_gpx(PLAN | {"route_ids": ["r-1", "r-2"], "routes": routes}))
    tracks = root.findall("g:trk", NS)
    assert len(tracks) == 1 and tracks[0].findtext("g:name", namespaces=NS) == PLAN["title"]
    assert [p.get("lat") for p in root.findall("g:trk/g:trkseg/g:trkpt", NS)] == ["47.490000", "47.440000", "47.260000"]


def test_gpx_names_wikivoyage_sources() -> None:
    attribution = {
        "publisher": "Wikivoyage", "title": "Testradweg", "url": "https://de.wikivoyage.org/wiki/Testradweg",
        "revision_date": "2026-10-03", "history_url": "https://de.wikivoyage.org/wiki/Testradweg?action=history",
        "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/deed.de",
    }  # fmt: skip
    root = ET.fromstring(build_gpx(PLAN | {"attributions": [attribution]}))
    link = root.find("g:metadata/g:link", NS)
    assert link is not None and link.get("href") == attribution["url"]
    assert "CC BY-SA 4.0" in link.findtext("g:text", namespaces=NS)


async def test_gpx_without_route_has_only_waypoints(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    adventure_id = await store_adventure("user@example.com", PLAN | {"route_ids": [], "routes": {}})
    response = await client.get(f"/api/adventures/{adventure_id}/gpx", headers=user_headers)
    root = ET.fromstring(response.content)
    assert response.headers["x-gpx-has-track"] == "false"
    assert root.findall("g:trk", NS) == []
    assert len(root.findall("g:wpt", NS)) == 4


async def test_gpx_requires_plan_and_ownership(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    draft_id = await store_adventure("user@example.com", None)
    assert (await client.get(f"/api/adventures/{draft_id}/gpx", headers=user_headers)).status_code == 404

    planned_id = await store_adventure("user@example.com", PLAN)
    await create_user("other@example.com")
    other = await auth_headers(client, "other@example.com")
    assert (await client.get(f"/api/adventures/{planned_id}/gpx", headers=other)).status_code == 404


def test_simplify_track_keeps_corners_and_drops_straight_points() -> None:
    straight = [[47.0 + i * 0.001, 11.0, 500.0] for i in range(50)]
    assert simplify_track(straight) == [straight[0], straight[-1]]

    corner = straight + [[47.049, 11.0 + i * 0.001, 500.0] for i in range(1, 50)]
    simplified = simplify_track(corner)
    assert len(simplified) == 3
    assert simplified[1] == straight[-1]


def test_export_filename() -> None:
    assert export_filename("Radtour Ulm → Lindau (Camping)", "gpx") == "radtour-ulm-lindau-camping.gpx"
    assert export_filename("Straße über'n Paß", "gpx") == "strasse-uber-n-pass.gpx"
    assert export_filename("→→→", "gpx") == "abenteuer.gpx"
    assert export_filename("Radtour", "pdf") == "radtour.pdf"
