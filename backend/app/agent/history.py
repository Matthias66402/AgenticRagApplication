"""Auswertung des gespeicherten Planer-Verlaufs (PydanticAI-JSON), gemeinsam für Ausgabeprüfungen und Intake.

Beide müssen dasselbe entscheiden: Hält das Intake eine Recherche für erledigt, die der Planer nicht gelten lässt,
kostet das eine zusätzliche Planer-Runde.
"""

import datetime as dt
import json
from collections.abc import Iterator
from typing import Any

# search_knowledge: frühere Abenteuer, in denen der Planer die Wissensbasis noch selbst durchsucht hat
RESEARCH_TOOLS = {"research_knowledge", "search_knowledge"}
OUTPUT_TOOLS = {"final_plan", "ask_user", "final_train_trip", "answer_user"}
# Läufe, die so enden, gehören zu keiner Tourenplanung (z. B. Fragen zur Arbeitsweise, Bahnauskunft)
NON_PLANNING_OUTPUTS = {"answer_user", "final_train_trip"}


def tool_args(part: dict[str, Any]) -> dict[str, Any]:
    """Argumente eines Tool-Aufrufs; im Verlauf als Objekt oder (bei manchen Modellen) als JSON-Text."""
    args = part.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {}
    return args if isinstance(args, dict) else {}


def runs(history_json: list[Any]) -> Iterator[list[dict[str, Any]]]:
    """Teilt den Verlauf in Läufe: je eine Nutzernachricht mit den folgenden Nachrichten bis zur nächsten."""
    run: list[dict[str, Any]] = []
    for message in history_json:
        starts_run = message.get("kind") == "request" and any(
            part.get("part_kind") == "user-prompt" for part in message.get("parts", [])
        )
        if starts_run and run:
            yield run
            run = []
        run.append(message)
    if run:
        yield run


def tool_calls(messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for message in messages:
        if message.get("kind") == "response":
            yield from (part for part in message.get("parts", []) if part.get("part_kind") == "tool-call")


def research_transports(history_json: list[Any]) -> set[str | None]:
    """Fortbewegungsarten der Recherchen, die für einen Plan zählen; None für eine Recherche ohne Angabe
    (search_knowledge in älteren Abenteuern), die für jede gilt.

    Recherchen aus Läufen, die mit einer Textantwort bzw. Reisekarte endeten, zählen nicht: In Abenteuer 196 hatte
    der Planer bei Fragen zur Routenwahl "Routing algorithms" recherchiert, der spätere Plan galt damit als
    recherchiert. Ein noch offener Lauf (der aktuelle) zählt.
    """
    transports: set[str | None] = set()
    for run in runs(history_json):
        calls = list(tool_calls(run))
        outputs = [call["tool_name"] for call in calls if call.get("tool_name") in OUTPUT_TOOLS]
        if outputs and outputs[-1] in NON_PLANNING_OUTPUTS:
            continue
        for call in calls:
            if call.get("tool_name") in RESEARCH_TOOLS:
                transport = tool_args(call).get("transport")
                transports.add(transport if isinstance(transport, str) else None)
    return transports


def researched_for(history_json: list[Any], transport: str) -> bool:
    transports = research_transports(history_json)
    return None in transports or transport in transports


def weather_results(history_json: list[Any], day: dt.date) -> list[Any]:
    """Ergebnisse der `get_weather`-Aufrufe, deren Zeitraum den Tag enthält (None, solange ein Ergebnis fehlt)."""
    results = {
        part.get("tool_call_id"): part.get("content")
        for message in history_json
        if message.get("kind") == "request"
        for part in message.get("parts", [])
        if part.get("part_kind") == "tool-return" and part.get("tool_name") == "get_weather"
    }
    covering = []
    for call in tool_calls(history_json):
        if call.get("tool_name") != "get_weather":
            continue
        args = tool_args(call)
        try:
            start = dt.date.fromisoformat(str(args.get("start_date")))
            days = max(1, int(args.get("days", 1)))
        except (TypeError, ValueError):
            continue
        if start <= day < start + dt.timedelta(days=days):
            covering.append(results.get(call.get("tool_call_id")))
    return covering
