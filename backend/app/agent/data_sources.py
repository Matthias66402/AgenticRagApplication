"""Datenquellen eines Plans: welche externen Dienste Daten für den Plan geliefert haben, mit Lizenz.

Wie die Quellenangaben der Wissensbasis (app.rag.attribution) deterministisch aus dem Agent-Verlauf bestimmt, nicht
aus den Freitext-`sources` des LLM. Das nannte Datendienste uneinheitlich ("OpenRouteService (r-0665adad)",
"plan_hut_tour Ergebnis") oder ließ sie weg, obwohl OpenStreetMap (ODbL) und Open-Meteo (CC BY 4.0) eine
Quellenangabe verlangen. Gezählt wird jedes erfolgreiche Tool-Ergebnis, auch aus früheren Nachrichten und aus den
Vorab-Aufrufen des Graphen (sie stehen ebenfalls im Verlauf).
"""

from dataclasses import asdict, dataclass
from typing import Any

OSM_DATA = "© OpenStreetMap-Mitwirkende"
ODBL = ("ODbL 1.0", "https://www.openstreetmap.org/copyright")


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    data: str | None = None  # Urheber der Daten, falls nicht der Dienst selbst
    license: str | None = None
    license_url: str | None = None


def _osm(name: str, url: str) -> Provider:
    return Provider(name, url, OSM_DATA, *ODBL)


# Schlüssel: Dienst; die Reihenfolge bestimmt die Anzeige
PROVIDERS: dict[str, Provider] = {
    "nominatim": _osm("Nominatim", "https://nominatim.openstreetmap.org"),
    "overpass": _osm("Overpass API", "https://overpass-api.de"),
    "catalog": _osm("Fernweg-Katalog (aus OpenStreetMap importiert)", "https://www.openstreetmap.org"),
    "waymarked": _osm("Waymarked Trails", "https://waymarkedtrails.org"),
    # Fernweg-Verlauf aus Verläufen, die die Herkunft noch nicht mitschrieben
    "openstreetmap": _osm("OpenStreetMap", "https://www.openstreetmap.org"),
    "ors": _osm("OpenRouteService", "https://openrouteservice.org"),
    "brouter": _osm("BRouter", "https://brouter.de"),
    "open-meteo": Provider(
        "Open-Meteo",
        "https://open-meteo.com",
        "Open-Meteo.com",
        "CC BY 4.0",
        "https://creativecommons.org/licenses/by/4.0/",
    ),
    "transitous": Provider("Transitous", "https://transitous.org", "Fahrplandaten der Verkehrsunternehmen"),
}

# Router-Namen im Feld `source` von plan_route (app.agent.tools.routing.PROVIDER_NAMES)
ROUTER_PREFIXES = {"OpenRouteService": "ors", "BRouter": "brouter"}
# find_accommodation: `source` ist "Nominatim" oder "Overpass"
ACCOMMODATION_SOURCES = {"Nominatim": "nominatim", "Overpass": "overpass"}


def _route_uses(route: Any) -> list[tuple[str, str]]:
    """Router und ggf. Herkunft des Fernweg-Verlaufs; Luftlinien-Schätzungen haben keine Datenquelle."""
    if not isinstance(route, dict) or route.get("estimated", True) or "error" in route:
        return []
    uses = [
        (key, "Routenberechnung")
        for prefix, key in ROUTER_PREFIXES.items()
        if str(route.get("source", "")).startswith(prefix)
    ]
    named = route.get("named_route")
    if isinstance(named, dict):
        # Mehrere Fernwege (Komma-getrennt in route_name) können aus verschiedenen Quellen stammen
        sources = str(named.get("source") or "").split(",")
        for source in dict.fromkeys(s if s in PROVIDERS else "openstreetmap" for s in sources):
            uses.append((source, "Verlauf des Fernwegs"))
    if isinstance(route.get("main_roads"), dict):
        uses.append(("overpass", "Erkennung von Hauptstraßen"))
    return uses


def tool_uses(tool_name: str, content: Any) -> list[tuple[str, str]]:
    """(Dienst, Zweck) für ein Tool-Ergebnis; leer bei Fehlern und Ergebnissen ohne Daten."""
    if isinstance(content, dict) and "error" in content:
        return []
    if tool_name == "geocode":
        return [("nominatim", "Ortssuche")] if isinstance(content, list) and content else []
    if tool_name == "find_accommodation":
        if isinstance(content, dict) and content.get("results") and content.get("source") in ACCOMMODATION_SOURCES:
            return [(ACCOMMODATION_SOURCES[content["source"]], "Unterkunftssuche")]
        return []
    if tool_name == "plan_route":
        return _route_uses(content)
    if tool_name == "plan_hut_tour":
        if not isinstance(content, dict) or not (content.get("huts") or content.get("other_huts")):
            return []
        return [("nominatim", "Hüttensuche"), *_route_uses(content.get("route"))]
    if tool_name == "get_weather":
        return [("open-meteo", "Wetter")] if isinstance(content, dict) else []
    if tool_name == "find_train_connections":
        return [("transitous", "Bahnverbindungen")] if isinstance(content, dict) and content.get("connections") else []
    return []


def data_sources(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Genutzte Datendienste aus dem Agent-Verlauf (PydanticAI-JSON), je Dienst mit allen Zwecken."""
    purposes: dict[str, dict[str, None]] = {}
    for message in history:
        for part in message.get("parts", []):
            if part.get("part_kind") != "tool-return":
                continue
            for key, purpose in tool_uses(part.get("tool_name", ""), part.get("content")):
                purposes.setdefault(key, {}).setdefault(purpose, None)
    return [asdict(PROVIDERS[key]) | {"purpose": ", ".join(purposes[key])} for key in PROVIDERS if key in purposes]


def describe(source: dict[str, Any]) -> str:
    """Eine Zeile für PDF und GPX, z. B. "Nominatim (Ortssuche) – Daten © OpenStreetMap-Mitwirkende, Lizenz ODbL 1.0"."""
    text = f"{source['name']} ({source['purpose']})"
    data = source.get("data") or ""
    details = [f"Daten {data}" if data.startswith("©") else data]
    if source.get("license"):
        details.append(f"Lizenz {source['license']}")
    if details := [d for d in details if d]:
        text += " – " + ", ".join(details)
    return text
