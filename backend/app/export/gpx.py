"""Export eines gespeicherten Abenteuerplans als GPX 1.1."""

import xml.etree.ElementTree as ET
from typing import Any

from app.agent.tools.common import haversine_km
from app.export.trip import is_train_trip

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


def _track(plan: dict[str, Any]) -> list[list[float]]:
    """Alle Teilrouten als ein durchgehender Track; der doppelte Punkt am Übergang zweier Teilrouten entfällt."""
    routes = plan.get("routes", {})
    track: list[list[float]] = []
    for point in (point for rid in plan.get("route_ids", []) for point in routes.get(rid, [])):
        if not track or point[:2] != track[-1][:2]:
            track.append(point)
    return track


def has_track(plan: dict[str, Any]) -> bool:
    return any(plan.get("routes", {}).get(rid) for rid in plan.get("route_ids", []))


def stage_tracks(plan: dict[str, Any]) -> list[list[list[float]]]:
    """Teilt den Gesamttrack an den Etappenzielen (nächster Trackpunkt, fortlaufend gesucht) in einen Track je Tag."""
    track = _track(plan)
    stages = plan["stages"]
    if not track:
        return [[] for _ in stages]
    cuts = [0]
    for stage in stages[:-1]:
        end = stage["end"]
        cuts.append(
            min(range(cuts[-1], len(track)), key=lambda i: haversine_km(end["lat"], end["lon"], track[i][0], track[i][1]))
        )
    cuts.append(len(track) - 1)
    return [track[a : b + 1] for a, b in zip(cuts, cuts[1:])]


def build_gpx(plan: dict[str, Any], day: int | None = None) -> str:
    """Track aus den berechneten Routen (inkl. Höhe) plus Wegpunkte für Start, Etappenziele und Unterkünfte.

    Mit ``day`` nur die Tagesetappe: Wegpunkte für deren Start, Ziel und Unterkunft und der Trackabschnitt bis zum
    Etappenziel.
    """
    stages = plan["stages"] if day is None else [s for s in plan["stages"] if s["day"] == day]
    title = plan["title"] if day is None else f"{plan['title']} – Tag {day}"
    gpx = ET.Element("gpx", version="1.1", creator="AgenticRAG-Abenteuerplaner", xmlns=GPX_NS)

    metadata = ET.SubElement(gpx, "metadata")
    ET.SubElement(metadata, "name").text = title
    ET.SubElement(metadata, "desc").text = plan.get("summary", "")
    # Namensnennung für genutzte Wissensbasis-Inhalte (z. B. Wikivoyage, CC BY-SA)
    for attribution in plan.get("attributions", []):
        link = ET.SubElement(metadata, "link", href=attribution["url"])
        ET.SubElement(link, "text").text = (
            f"{attribution['publisher']}: „{attribution['title']}“, Version vom {attribution['revision_date']}, "
            f"Autoren: {attribution['history_url']}, Lizenz: {attribution['license']} ({attribution['license_url']})"
        )

    start = plan["start"] if day is None else stages[0]["start"]
    start_date = plan["start_date"] if day is None else stages[0]["date"]
    _add_waypoint(gpx, start["lat"], start["lon"], f"Start: {_short(start['name'])}", start_date, "Flag, Green")
    for stage in stages:
        end = stage["end"]
        details = [f"{stage['distance_km']:g} km"]
        if stage.get("ascent_m"):
            details.append(f"↗ {stage['ascent_m']:g} Hm")
        if stage.get("descent_m"):
            details.append(f"↘ {stage['descent_m']:g} Hm")
        label = "Bahnhof" if is_train_trip(plan) else f"Tag {stage['day']}"
        _add_waypoint(
            gpx, end["lat"], end["lon"], f"{label}: {_short(end['name'])}",
            f"{stage['date']} · {', '.join(details)}", "Flag, Blue",
        )  # fmt: skip
        if acc := stage.get("accommodation"):
            label = ACCOMMODATION_LABELS.get(acc.get("type", ""), "Unterkunft")
            desc = f"{label}, Übernachtung nach Tag {stage['day']}" + (f" · {acc['website']}" if acc.get("website") else "")
            _add_waypoint(gpx, acc["lat"], acc["lon"], acc["name"], desc, "Lodging")

    # Immer nur ein Track: Komoot & Co. bieten mehrere Tracks einer Datei als getrennte Touren zur Auswahl an
    points = _track(plan) if day is None else stage_tracks(plan)[plan["stages"].index(stages[0])]
    if points:
        trk = ET.SubElement(gpx, "trk")
        ET.SubElement(trk, "name").text = title
        ET.SubElement(trk, "type").text = plan.get("transport", "")
        segment = ET.SubElement(trk, "trkseg")
        for point in points:
            _add_point(segment, "trkpt", point[0], point[1], point[2] if len(point) > 2 else None)

    ET.indent(gpx)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(gpx, encoding="unicode")
