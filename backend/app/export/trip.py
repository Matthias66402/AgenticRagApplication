"""Reisekarten (Bahnfahrt, optional mit Weg zum bzw. vom Bahnhof) in der Form eines Tourplans für PDF- und GPX-Export."""

from typing import Any

from app.agent.plan_schema import PLAN_KIND_TRAIN_TRIP


def is_train_trip(plan: dict[str, Any]) -> bool:
    return plan.get("kind") == PLAN_KIND_TRAIN_TRIP


def station_route_labels(plan: dict[str, Any]) -> tuple[str, str]:
    """Überschrift des Wegs und Bezeichnung seines Endpunkts: zum Bahnhof bzw. vom Zielbahnhof zum Ziel."""
    if plan.get("station_route") == "egress":
        return "Weg vom Bahnhof", "Ziel"
    return "Weg zum Bahnhof", "Bahnhof"


def exportable_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Tourpläne unverändert; Reisekarten mit dem Weg zum bzw. vom Bahnhof als einziger Etappe und der Bahnfahrt als
    Hinfahrt.

    Gibt es beide Wege, enthält der Export den Weg am Zielort (unbekannte Gegend): Beide zusammen ergäben einen Track
    quer durchs Land. Ohne Weg hat die Reisekarte keine Etappen und keinen Startpunkt (Verbindungen haben keine
    Koordinaten): Das PDF zeigt dann nur Überblick und Verbindung, ein GPX gibt es nicht.
    """
    if not is_train_trip(plan):
        return plan
    journey = plan["outbound_journey"]
    common = {
        "kind": PLAN_KIND_TRAIN_TRIP,
        "title": plan["title"],
        "summary": plan.get("summary", ""),
        "start_date": journey["departure"][:10],
        "duration_days": 1,
        "destination": {"name": journey["legs"][-1]["to_stop"]},
        "outbound_journey": journey,
        "return_journey": plan.get("return_journey"),
        "routes": plan.get("routes", {}),
        "attributions": plan.get("attributions", []),
        "data_sources": plan.get("data_sources", []),
    }
    if egress := plan.get("egress"):
        leg, kind = egress, "egress"
        notes = f"Ankunft {journey['arrival'][11:16]}, Losfahren um {egress['departure'][11:16]}."
    elif access := plan.get("access"):
        leg, kind = access, "access"
        notes = f"Losfahren um {access['departure'][11:16]}, Verbindung ab {journey['departure'][11:16]}."
    else:
        return common | {"transport": "public_transport", "start": None, "stages": [], "route_ids": []}
    stage = {
        "day": 1,
        "label": f"ab {leg['departure'][11:16]}",  # statt "Tag 1"
        "date": leg["departure"][:10],
        "start": leg["start"],
        "end": leg["end"],
        "distance_km": leg["distance_km"],
        "duration_hours": leg["duration_hours"],
        "notes": notes,
    }
    return common | {
        "transport": leg["transport"],
        "station_route": kind,
        "start": leg["start"],
        "stages": [stage],
        "route_ids": [leg["route_id"]],
    } | ({"destination": leg["end"]} if kind == "egress" else {})
