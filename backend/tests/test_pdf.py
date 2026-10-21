import io

import httpx
from pypdf import PdfReader

from app.export.pdf import build_pdf
from tests.conftest import auth_headers, create_user
from tests.test_gpx import PLAN, store_adventure


def pdf_text(content: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(content)).pages)


async def test_pdf_download(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
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


def test_pdf_without_route_and_with_optional_sections() -> None:
    plan = PLAN | {
        "route_ids": [], "routes": {},
        "weather_summary": "Sonnig, 12 °C", "packing_tips": ["Stirnlampe"],
        "safety_notes": ["Notruf 112"], "sources": ["Open-Meteo"],
    }  # fmt: skip
    text = pdf_text(build_pdf(plan))
    assert "Höhenprofil" not in text
    assert "Luftlinie" in text
    for expected in ("Wetter", "Sonnig, 12 °C", "Packliste & Tipps", "Stirnlampe", "Sicherheit & Regeln", "Quellen"):
        assert expected in text


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
