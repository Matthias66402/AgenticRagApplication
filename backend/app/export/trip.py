"""Reisekarten (Bahnfahrt, optional mit Weg zum Bahnhof) in der Form eines Tourplans für PDF- und GPX-Export."""

from typing import Any

from app.agent.plan_schema import PLAN_KIND_TRAIN_TRIP


def is_train_trip(plan: dict[str, Any]) -> bool:
    return plan.get("kind") == PLAN_KIND_TRAIN_TRIP


def exportable_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Tourpläne unverändert; Reisekarten mit dem Weg zum Bahnhof als einziger Etappe und der Bahnfahrt als Hinfahrt.

    Ohne Weg zum Bahnhof hat die Reisekarte keine Etappen und keinen Startpunkt (Verbindungen haben keine
    Koordinaten): Das PDF zeigt dann nur Überblick und Verbindung, ein GPX gibt es nicht.
    """
    if not is_train_trip(plan):
        return plan
    journey = plan["outbound_journey"]
    access = plan.get("access")
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
    }
    if access is None:
        return common | {"transport": "public_transport", "start": None, "stages": [], "route_ids": []}
    stage = {
        "day": 1,
        "label": f"ab {access['departure'][11:16]}",  # statt "Tag 1"
        "date": journey["departure"][:10],
        "start": access["start"],
        "end": access["end"],
        "distance_km": access["distance_km"],
        "duration_hours": access["duration_hours"],
        "notes": f"Losfahren um {access['departure'][11:16]}, Verbindung ab {journey['departure'][11:16]}.",
    }
    return common | {
        "transport": access["transport"],
        "start": access["start"],
        "stages": [stage],
        "route_ids": [access["route_id"]],
    }
