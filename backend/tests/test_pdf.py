import io

import httpx
import pytest
import respx
from PIL import Image
from pypdf import PdfReader

from app.export import basemap
from app.export.basemap import ATTRIBUTION, PADDING, TARGET_WIDTH_PX, TILE_SIZE, fit_viewport
from app.export.pdf import build_pdf, map_viewport
from tests.conftest import auth_headers, create_user
from tests.test_gpx import PLAN, store_adventure


TILE_PATTERN = r"https://tile\.openstreetmap\.org/\d+/\d+/\d+\.png"


@pytest.fixture(autouse=True)
def empty_tile_cache(monkeypatch) -> None:
    monkeypatch.setattr(basemap, "_cache", {})


def png_tile() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (TILE_SIZE, TILE_SIZE), (170, 211, 223)).save(buffer, format="PNG")
    return buffer.getvalue()


def pdf_text(content: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(content)).pages)


@respx.mock
async def test_pdf_download(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    tiles = respx.get(url__regex=TILE_PATTERN).respond(content=png_tile(), headers={"Content-Type": "image/png"})
    adventure_id = await store_adventure("user@example.com", PLAN)
    response = await client.get(f"/api/adventures/{adventure_id}/pdf", headers=user_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert 'filename="huttentour-garmisch-innsbruck.pdf"' in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")

    text = pdf_text(response.content)
    assert "Hüttentour Garmisch → Innsbruck" in text
    assert "Fr., 06.11.2026 – Sa., 07.11.2026" in text
    assert "45,5 km" in text
    assert "Höhenprofil" in text
    assert "Garmisch → Mittenwald" in text
    assert "Übernachtung: Testhütte (Hütte)" in text
    assert "https://huette.example" in text
    # Kartenhintergrund eingebettet, mit OSM-Namensnennung
    assert tiles.called
    assert ATTRIBUTION in text
    assert len(PdfReader(io.BytesIO(response.content)).pages[0].images) == 1


@respx.mock
async def test_pdf_without_tile_server_has_no_basemap(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    respx.get(url__regex=TILE_PATTERN).respond(status_code=503)
    adventure_id = await store_adventure("user@example.com", PLAN)
    response = await client.get(f"/api/adventures/{adventure_id}/pdf", headers=user_headers)

    assert response.status_code == 200
    assert ATTRIBUTION not in pdf_text(response.content)
    assert len(PdfReader(io.BytesIO(response.content)).pages[0].images) == 0


def test_map_viewport_contains_route_and_markers() -> None:
    viewport = map_viewport(PLAN)
    assert TARGET_WIDTH_PX / 2 < viewport.width <= TARGET_WIDTH_PX
    for lat, lon, *_ in PLAN["routes"]["r-1"]:
        fx, fy = viewport.project(lat, lon)
        assert PADDING * 0.99 <= fx <= 1 - PADDING * 0.99
        assert 0 <= fy <= 1


def test_fit_viewport_single_point() -> None:
    viewport = fit_viewport([(47.49, 11.09)], aspect=2)
    assert viewport.project(47.49, 11.09) == pytest.approx((0.5, 0.5))
    assert viewport.zoom == basemap.MAX_ZOOM


ATTRIBUTION_ENTRY = {
    "publisher": "Wikivoyage", "title": "Rhein (Fluss)",
    "url": "https://de.wikivoyage.org/wiki/Rhein_(Fluss)", "revision_date": "2026-01-02",
    "history_url": "https://de.wikivoyage.org/wiki/Rhein_(Fluss)?action=history",
    "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/deed.de",
}  # fmt: skip


def test_pdf_without_route_and_with_optional_sections() -> None:
    plan = PLAN | {
        "route_ids": [], "routes": {},
        "weather_summary": "Sonnig, 12 °C", "packing_tips": ["Stirnlampe"],
        "safety_notes": ["Notruf 112"], "sources": ["Open-Meteo"], "attributions": [ATTRIBUTION_ENTRY],
    }  # fmt: skip
    content = build_pdf(plan)
    text = pdf_text(content)
    assert "Höhenprofil" not in text
    assert "Luftlinie" in text
    for expected in ("Wetter", "Sonnig, 12 °C", "Packliste & Tipps", "Stirnlampe", "Sicherheit & Regeln", "Quellen"):
        assert expected in text

    # Namensnennung nach CC BY-SA, mit klickbaren Links auf Artikel, Versionsgeschichte und Lizenz
    assert "„Rhein (Fluss)“ – Wikivoyage, Version vom 02.01.2026" in text
    assert "zusammengefasst und bearbeitet" in text
    assert "CC BY-SA 4.0" in text
    links = {
        annotation.get_object()["/A"]["/URI"]
        for page in PdfReader(io.BytesIO(content)).pages
        for annotation in page.get("/Annots", [])
    }
    assert {
        "https://de.wikivoyage.org/wiki/Rhein_(Fluss)",
        "https://de.wikivoyage.org/wiki/Rhein_%28Fluss%29?action=history",
        "https://creativecommons.org/licenses/by-sa/4.0/deed.de",
    } <= links


def test_pdf_handles_long_plans_across_pages() -> None:
    stage = PLAN["stages"][1] | {"notes": "Lange Beschreibung. " * 40}
    plan = PLAN | {"duration_days": 14, "stages": [stage | {"day": day} for day in range(1, 15)]}
    reader = PdfReader(io.BytesIO(build_pdf(plan)))
    assert len(reader.pages) > 2
    assert f"Seite {len(reader.pages)}/{len(reader.pages)}" in reader.pages[-1].extract_text()


async def test_pdf_requires_plan_and_ownership(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    draft_id = await store_adventure("user@example.com", None)
    assert (await client.get(f"/api/adventures/{draft_id}/pdf", headers=user_headers)).status_code == 404

    planned_id = await store_adventure("user@example.com", PLAN)
    await create_user("pdf-other@example.com")
    other = await auth_headers(client, "pdf-other@example.com")
    assert (await client.get(f"/api/adventures/{planned_id}/pdf", headers=other)).status_code == 404
