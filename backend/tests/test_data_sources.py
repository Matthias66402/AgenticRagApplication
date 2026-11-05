import xml.etree.ElementTree as ET
from typing import Any

from app.agent.data_sources import data_sources, describe
from app.export.gpx import build_gpx
from app.export.pdf import build_pdf
from tests.test_gpx import NS, PLAN
from tests.test_pdf import pdf_links, pdf_text


def tool_return(tool_name: str, content: Any) -> dict[str, Any]:
    return {"kind": "request", "parts": [{"part_kind": "tool-return", "tool_name": tool_name, "content": content}]}


ROUTE = {"route_id": "r-1", "estimated": False, "source": "OpenRouteService foot-hiking", "distance_km": 12.0}
HISTORY = [
    {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Hüttenwanderung"}]},
    tool_return("geocode", [{"name": "Garmisch-Partenkirchen", "lat": 47.49, "lon": 11.09}]),
    tool_return("geocode", {"error": "Nichts gefunden"}),
    tool_return("plan_hut_tour", {"huts": [{"name": "Meilerhütte"}], "route": ROUTE, "other_huts": []}),
    tool_return("find_accommodation", {"results": [{"name": "Hotel"}], "source": "Nominatim"}),
    tool_return("find_accommodation", {"results": [], "source": "Overpass", "note": "Keine Unterkünfte"}),
    tool_return("get_weather", {"days": []}),
    tool_return("find_train_connections", {"connections": []}),
    # Luftlinien-Schätzung: keine Datenquelle
    tool_return("plan_route", {"estimated": True, "distance_km": 30.0}),
    tool_return(
        "plan_route",
        {"route_id": "r-2", "estimated": False, "source": "BRouter trekking entlang Elberadweg",
         "named_route": {"routes": ["Elberadweg"], "source": "waymarked"}},
    ),  # fmt: skip
]


def test_data_sources_from_successful_tool_results() -> None:
    sources = data_sources(HISTORY)
    assert [(s["name"], s["purpose"]) for s in sources] == [
        ("Nominatim", "Ortssuche, Hüttensuche, Unterkunftssuche"),
        ("Waymarked Trails", "Verlauf des Fernwegs"),
        ("OpenRouteService", "Routenberechnung"),
        ("BRouter", "Routenberechnung"),
        ("Open-Meteo", "Wetter"),
    ]
    nominatim = sources[0]
    assert nominatim["data"] == "© OpenStreetMap-Mitwirkende" and nominatim["license"] == "ODbL 1.0"
    assert sources[4]["license"] == "CC BY 4.0"


def test_named_route_without_recorded_source() -> None:
    """Ältere Verläufe ohne named_route.source: Verlauf allgemein aus OpenStreetMap."""
    route = ROUTE | {"named_route": {"routes": ["Elberadweg"]}}
    assert [s["name"] for s in data_sources([tool_return("plan_route", route)])] == ["OpenStreetMap", "OpenRouteService"]


def test_several_named_routes_from_different_sources() -> None:
    route = ROUTE | {"named_route": {"routes": ["Hunteradweg", "Weser-Radweg"], "source": "waymarked,catalog"}}
    names = [s["name"] for s in data_sources([tool_return("plan_route", route)])]
    assert "Waymarked Trails" in names and "Fernweg-Katalog (aus OpenStreetMap importiert)" in names


def test_data_sources_without_tools() -> None:
    assert data_sources([]) == []
    assert data_sources([tool_return("estimate_hiking_time", {"walking_hours": 3.0})]) == []


def test_describe_names_data_and_license() -> None:
    nominatim, *_, weather = data_sources(HISTORY)
    assert describe(nominatim) == (
        "Nominatim (Ortssuche, Hüttensuche, Unterkunftssuche) – Daten © OpenStreetMap-Mitwirkende, Lizenz ODbL 1.0"
    )
    assert describe(weather) == "Open-Meteo (Wetter) – Open-Meteo.com, Lizenz CC BY 4.0"
    transitous = data_sources([tool_return("find_train_connections", {"connections": [{"departure": "x"}]})])[0]
    assert describe(transitous) == "Transitous (Bahnverbindungen) – Fahrplandaten der Verkehrsunternehmen"


def test_exports_name_data_sources() -> None:
    plan = PLAN | {"data_sources": data_sources(HISTORY)}
    root = ET.fromstring(build_gpx(plan))
    links = {link.get("href"): link.findtext("g:text", namespaces=NS) for link in root.findall("g:metadata/g:link", NS)}
    assert "ODbL 1.0" in links["https://nominatim.openstreetmap.org"]
    assert "CC BY 4.0" in links["https://open-meteo.com"]

    content = build_pdf(plan)
    text = pdf_text(content)
    assert "Daten von:" in text and "Lizenz ODbL 1.0" in text
    assert {"https://nominatim.openstreetmap.org", "https://open-meteo.com"} <= pdf_links(content)
