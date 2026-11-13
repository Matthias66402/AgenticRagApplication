import datetime as dt
from typing import Any

from app.agent.history import research_transports, researched_for, weather_results


def user(text: str) -> dict[str, Any]:
    return {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": text}]}


def calls(*tools: tuple[str, Any]) -> dict[str, Any]:
    return {
        "kind": "response",
        "parts": [
            {"part_kind": "tool-call", "tool_name": name, "args": args, "tool_call_id": f"c-{name}-{i}"}
            for i, (name, args) in enumerate(tools)
        ],
    }


def returns(*results: tuple[str, str, Any]) -> dict[str, Any]:
    return {
        "kind": "request",
        "parts": [
            {"part_kind": "tool-return", "tool_name": name, "tool_call_id": call_id, "content": content}
            for name, call_id, content in results
        ],
    }


ROUTING_RESEARCH = {"region": "Routing algorithms for cycling in Europe", "transport": "cycling"}


def test_research_from_answer_runs_does_not_count() -> None:
    """Abenteuer 196: Recherchen bei Fragen zur Routenwahl gelten nicht für den späteren Plan."""
    history = [
        user("Wie wählst du die Routen aus?"),
        calls(("research_knowledge", ROUTING_RESEARCH)),
        calls(("answer_user", {"message": "Bei Fernwegen …"})),
        user("Ich will heute nach Wilhelmshaven radeln"),  # aktueller Lauf ohne Recherche
        calls(("geocode", {"place_name": "Wilhelmshaven"})),
    ]
    assert research_transports(history) == set()
    assert not researched_for(history, "cycling")


def test_research_from_planning_runs_counts_per_transport() -> None:
    history = [
        user("Wanderung ab Garmisch"),
        calls(("research_knowledge", '{"region": "Garmisch", "transport": "hiking"}')),  # Argumente als JSON-Text
        calls(("ask_user", {"question": "Wie lange?"})),
        user("Zwei Tage"),
        calls(("final_plan", {})),
    ]
    assert researched_for(history, "hiking")
    assert not researched_for(history, "cycling")  # andere Fortbewegungsart: neue Recherche
    # Aktueller, noch offener Lauf zählt; eine Ausgabe, die zurückgewiesen und dann ersetzt wurde, entscheidet nicht
    current = [user("Radtour"), calls(("research_knowledge", ROUTING_RESEARCH)), calls(("final_plan", {}))]
    assert researched_for(current, "cycling")
    assert not researched_for([*current, calls(("answer_user", {"message": "…"}))], "cycling")


def test_legacy_search_counts_for_any_transport() -> None:
    history = [user("Radtour"), calls(("search_knowledge", {"query": "Harz"})), calls(("final_plan", {}))]
    assert research_transports(history) == {None} and researched_for(history, "cycling")


def test_weather_results_cover_start_date() -> None:
    day = dt.date(2026, 11, 10)
    history = [
        user("Radtour"),
        calls(
            ("get_weather", {"lat": 53.1, "lon": 8.2, "start_date": "2026-11-08", "days": 3}),
            ("get_weather", {"lat": 53.1, "lon": 8.2, "start_date": "2026-11-11"}),
        ),
        returns(("get_weather", "c-get_weather-0", {"kind": "forecast", "days": []})),
    ]
    assert weather_results(history, day) == [{"kind": "forecast", "days": []}]
    assert weather_results(history, day + dt.timedelta(days=1)) == [None]  # days fehlt: 1 Tag, Ergebnis fehlt
    assert weather_results(history, day + dt.timedelta(days=2)) == []
