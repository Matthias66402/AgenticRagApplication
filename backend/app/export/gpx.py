"""Export eines gespeicherten Abenteuerplans als GPX 1.1."""

import xml.etree.ElementTree as ET
from typing import Any

GPX_NS = "http://www.topografix.com/GPX/1/1"
ACCOMMODATION_LABELS = {"hut": "Hütte", "camping": "Camping", "hotel": "Hotel", "hostel": "Hostel", "bivouac": "Biwak"}


def _short(name: str) -> str:
    return name.split(",")[0].strip()


def _add_point(parent: ET.Element, tag: str, lat: float, lon: float, ele: float | None = None) -> ET.Element:
    point = ET.SubElement(parent, tag, lat=f"{lat:.6f}", lon=f"{lon:.6f}")
    if ele is not None:
        ET.SubElement(point, "ele").text = f"{ele:.1f}"
    return point


def _add_waypoint(gpx: ET.Element, lat: float, lon: float, name: str, desc: str, symbol: str) -> None:
    wpt = _add_point(gpx, "wpt", lat, lon)
    ET.SubElement(wpt, "name").text = name
    ET.SubElement(wpt, "desc").text = desc
    ET.SubElement(wpt, "sym").text = symbol


def has_track(plan: dict[str, Any]) -> bool:
    return any(plan.get("routes", {}).get(rid) for rid in plan.get("route_ids", []))


def build_gpx(plan: dict[str, Any]) -> str:
    """Track aus den berechneten Routen (inkl. Höhe) plus Wegpunkte für Start, Etappenziele und Unterkünfte."""
    gpx = ET.Element("gpx", version="1.1", creator="AgenticRAG-Abenteuerplaner", xmlns=GPX_NS)

    metadata = ET.SubElement(gpx, "metadata")
    ET.SubElement(metadata, "name").text = plan["title"]
    ET.SubElement(metadata, "desc").text = plan.get("summary", "")
    # Namensnennung für genutzte Wissensbasis-Inhalte (z. B. Wikivoyage, CC BY-SA)
    for attribution in plan.get("attributions", []):
        link = ET.SubElement(metadata, "link", href=attribution["url"])
        ET.SubElement(link, "text").text = (
            f"{attribution['publisher']}: „{attribution['title']}“, Version vom {attribution['revision_date']}, "
            f"Autoren: {attribution['history_url']}, Lizenz: {attribution['license']} ({attribution['license_url']})"
        )

    start = plan["start"]
    _add_waypoint(gpx, start["lat"], start["lon"], f"Start: {_short(start['name'])}", plan["start_date"], "Flag, Green")
    for stage in plan["stages"]:
        end = stage["end"]
        details = [f"{stage['distance_km']:g} km"]
        if stage.get("ascent_m"):
            details.append(f"↗ {stage['ascent_m']:g} Hm")
        if stage.get("descent_m"):
            details.append(f"↘ {stage['descent_m']:g} Hm")
        _add_waypoint(
            gpx, end["lat"], end["lon"], f"Tag {stage['day']}: {_short(end['name'])}",
            f"{stage['date']} · {', '.join(details)}", "Flag, Blue",
        )  # fmt: skip
        if acc := stage.get("accommodation"):
            label = ACCOMMODATION_LABELS.get(acc.get("type", ""), "Unterkunft")
            desc = f"{label}, Übernachtung nach Tag {stage['day']}" + (f" · {acc['website']}" if acc.get("website") else "")
            _add_waypoint(gpx, acc["lat"], acc["lon"], acc["name"], desc, "Lodging")

    routes = plan.get("routes", {})
    for index, route_id in enumerate(plan.get("route_ids", []), start=1):
        points = routes.get(route_id)
        if not points:
            continue
        trk = ET.SubElement(gpx, "trk")
        ET.SubElement(trk, "name").text = plan["title"] if len(plan["route_ids"]) == 1 else f"{plan['title']} ({index})"
        ET.SubElement(trk, "type").text = plan.get("transport", "")
        segment = ET.SubElement(trk, "trkseg")
        for point in points:
            _add_point(segment, "trkpt", point[0], point[1], point[2] if len(point) > 2 else None)

    ET.indent(gpx)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(gpx, encoding="unicode")

