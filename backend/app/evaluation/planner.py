"""Misst Aufwand und Ergebnisqualität des Planungs-Agenten anhand fester Anfragen.

Aufruf: ``python -m app.evaluation.planner [--repeat 2] [--case ID] [--label NAME] [--output bericht.json]``

Jede Anfrage läuft wie im Chat (echtes Modell, echte Tools) als erste Nachricht eines neuen Abenteuers. Gemessen:
- Aufwand: Laufzeit, Modellaufrufe, Eingabe-/Cache-/Ausgabe-/Reasoning-Tokens und Kosten des Planers
  (ohne die Relevanzprüfung der Wissensbasis)
- Wiederholungen: Rückmeldungen an das Modell durch Validatoren oder Schemafehler
- Qualität: erwarteter Ausgabetyp und fallabhängige Prüfungen (Pflichtsuchen, Tipps, Unterkünfte, Verbindungen)

Kostet echte LLM-Aufrufe (ca. 0,02-0,05 $ je Anfrage). Varianten vergleicht man, indem man das Skript vor und nach
einer Änderung laufen lässt, z. B. ``LLM_REASONING_EFFORT=low python -m app.evaluation.planner --label low``.
"""

import argparse
import asyncio
import datetime as dt
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic_ai import (
    AgentRunResultEvent, FunctionToolCallEvent, FunctionToolResultEvent, ModelMessage, ModelRequest, ModelResponse,
    RetryPromptPart, ToolCallPart, ToolReturnPart,
)  # fmt: skip

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AdventurePlan, ClarificationQuestion, TextAnswer, TrainTripPlan
from app.agent.planner import planner_agent
from app.agent.runner import USAGE_LIMITS
from app.config import get_settings
from app.db import SessionLocal, engine

CASES_FILE = Path(__file__).parent / "planner_cases.json"
OUTPUT_NAMES = {
    AdventurePlan: "final_plan",
    TrainTripPlan: "final_train_trip",
    ClarificationQuestion: "ask_user",
    TextAnswer: "answer_user",
}
# Pflichtsuchen vor jedem neuen Plan (Region + Ausrüstung/Sicherheit, siehe INSTRUCTIONS)
MIN_KNOWLEDGE_SEARCHES = 2


@dataclass
class CaseRun:
    id: str
    run: int
    output: str | None = None
    # Rückfrage, Antworttext bzw. Plantitel, um Abweichungen nachvollziehen zu können
    text: str = ""
    error: str | None = None
    seconds: float = 0.0
    requests: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    # Wartezeit auf Tools (parallele Aufrufe zählen einmal); der Rest der Laufzeit entfällt auf das Modell
    tool_seconds: float = 0.0
    tool_calls: dict[str, int] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.error is None and all(self.checks.values())


def tool_returns(messages: list[ModelMessage], tool_name: str) -> list[Any]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


def check_output(case: dict[str, Any], output: Any, messages: list[ModelMessage], deps: AgentDeps) -> dict[str, bool]:
    """Fallabhängige Qualitätsprüfungen; der Ausgabetyp wird immer geprüft."""
    checks = {"output_type": OUTPUT_NAMES.get(type(output)) == case["expected_output"]}
    calls = [p.tool_name for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)]
    if isinstance(output, AdventurePlan):
        checks["transport"] = output.transport.value == case.get("transport", output.transport.value)
        checks["duration_days"] = output.duration_days == case.get("duration_days", output.duration_days)
        checks["knowledge_searches"] = calls.count("search_knowledge") >= MIN_KNOWLEDGE_SEARCHES
        checks["packing_and_safety"] = bool(output.packing_tips) and bool(output.safety_notes)
        checks["sources"] = bool(output.sources)
        checks["route_ids"] = bool(output.route_ids) and all(r in deps.routes for r in output.route_ids)
        if output.duration_days > 1:
            found = {
                entry["name"]
                for result in tool_returns(messages, "find_accommodation")
                if isinstance(result, dict)
                for entry in result.get("results", [])
            }
            stays = [s.accommodation for s in output.stages[:-1]]
            checks["accommodation_from_tool"] = all(a is not None and a.name in found for a in stays)
        if case.get("journeys"):
            checks["journeys"] = output.outbound_journey is not None and output.return_journey is not None
    if isinstance(output, TrainTripPlan) and case.get("access"):
        checks["access"] = output.access is not None
    if isinstance(output, TextAnswer):
        checks["train_query"] = "find_train_connections" in calls
    return checks


def measure(run: CaseRun, messages: list[ModelMessage]) -> None:
    for message in messages:
        if isinstance(message, ModelResponse):
            usage = message.usage
            run.requests += 1
            run.input_tokens += usage.input_tokens
            run.cache_read_tokens += usage.cache_read_tokens
            run.output_tokens += usage.output_tokens
            run.reasoning_tokens += usage.details.get("reasoning_tokens", 0)
            try:
                run.cost_usd += float(message.cost().total_price)
            except Exception:  # Modell ohne Preisangabe (z. B. in Tests)
                pass
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    run.tool_calls[part.tool_name] = run.tool_calls.get(part.tool_name, 0) + 1
        else:
            run.retries += sum(isinstance(part, RetryPromptPart) for part in message.parts)


def covered_seconds(spans: list[tuple[float, float]]) -> float:
    """Gesamtdauer der Zeitspannen, überlappende (parallel laufende Tools) nur einmal gezählt."""
    total, end = 0.0, float("-inf")
    for start, stop in sorted(spans):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


async def run_case(case: dict[str, Any], number: int, http: httpx.AsyncClient) -> CaseRun:
    run = CaseRun(id=case["id"], run=number)
    deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Testnutzer")
    started = time.perf_counter()
    tool_started: dict[str, float] = {}
    tool_spans: list[tuple[float, float]] = []
    result = None
    try:
        async with planner_agent.run_stream_events(
            case["message"], deps=deps, usage_limits=USAGE_LIMITS
        ) as events:
            async for event in events:
                if isinstance(event, FunctionToolCallEvent):
                    tool_started[event.part.tool_call_id] = time.perf_counter()
                elif isinstance(event, FunctionToolResultEvent) and event.part.tool_call_id in tool_started:
                    tool_spans.append((tool_started.pop(event.part.tool_call_id), time.perf_counter()))
                elif isinstance(event, AgentRunResultEvent):
                    result = event.result
    except Exception as exc:  # Abbruch zählt als nicht bestanden, die übrigen Fälle laufen weiter
        run.error = f"{type(exc).__name__}: {exc}"
    run.seconds = round(time.perf_counter() - started, 1)
    run.tool_seconds = round(covered_seconds(tool_spans), 1)
    if result is None:
        run.error = run.error or "Kein Ergebnis"
        return run
    messages = result.all_messages()
    measure(run, messages)
    run.output = OUTPUT_NAMES.get(type(result.output))
    output = result.output
    run.text = (
        output.question if isinstance(output, ClarificationQuestion)
        else output.message if isinstance(output, TextAnswer)
        else output.title
    )  # fmt: skip
    run.checks = check_output(case, result.output, messages, deps)
    run.cost_usd = round(run.cost_usd, 4)
    return run


def summarize(runs: list[CaseRun]) -> dict[str, Any]:
    def mean(attr: str) -> float:
        return round(statistics.mean(getattr(r, attr) for r in runs), 1) if runs else 0.0

    return {
        "runs": len(runs),
        "passed": sum(r.passed for r in runs),
        "errors": sum(r.error is not None for r in runs),
        "retries": sum(r.retries for r in runs),
        "mean_seconds": mean("seconds"),
        "mean_tool_seconds": mean("tool_seconds"),
        "mean_requests": mean("requests"),
        "mean_input_tokens": mean("input_tokens"),
        "mean_cache_read_tokens": mean("cache_read_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_reasoning_tokens": mean("reasoning_tokens"),
        "total_cost_usd": round(sum(r.cost_usd for r in runs), 3),
    }


async def run(cases: list[dict[str, Any]], repeat: int) -> list[CaseRun]:
    results = []
    async with httpx.AsyncClient(timeout=get_settings().http_timeout_seconds) as http:
        for case in cases:
            for number in range(1, repeat + 1):
                result = await run_case(case, number, http)
                status = "ok" if result.passed else f"FEHLER {result.error or result.checks}"
                print(f"  {case['id']} #{number}: {result.seconds:.0f} s, {result.requests} Aufrufe, {status}",
                      file=sys.stderr, flush=True)  # fmt: skip
                results.append(result)
    return results


def print_report(report: dict[str, Any]) -> None:
    s, c = report["summary"], report["config"]
    print(f"Variante {c['label']!r} · {c['llm_model']} · Reasoning {c['reasoning_effort'] or 'Standard'}")
    print(f"Bestanden {s['passed']}/{s['runs']} · Abbrüche {s['errors']} · Wiederholungen {s['retries']} · "
          f"Kosten {s['total_cost_usd']:.3f} $")  # fmt: skip
    print(f"Mittel je Anfrage: {s['mean_seconds']:.0f} s (davon Tools {s['mean_tool_seconds']:.0f} s) · "
          f"{s['mean_requests']:.1f} Aufrufe · "
          f"Eingabe {s['mean_input_tokens']:.0f} (Cache {s['mean_cache_read_tokens']:.0f}) · "
          f"Ausgabe {s['mean_output_tokens']:.0f} (Reasoning {s['mean_reasoning_tokens']:.0f})")  # fmt: skip
    print(f"\n{'Fall':18} {'Lauf':>4} {'Zeit':>6} {'Tools':>6} {'Aufr.':>5} {'Eingabe':>8} {'Ausgabe':>8} {'Reas.':>6} "
          f"{'Wdh.':>4}  Ergebnis")  # fmt: skip
    for r in report["cases"]:
        failed = [name for name, ok in r["checks"].items() if not ok]
        result = r["error"] or (
            f"{r['output']} ✗ {', '.join(failed)}: {r['text'][:120]}" if failed else f"{r['output']} ✓"
        )
        print(f"{r['id']:18} {r['run']:>4} {r['seconds']:>5.0f}s {r['tool_seconds']:>5.0f}s {r['requests']:>5} {r['input_tokens']:>8} "
              f"{r['output_tokens']:>8} {r['reasoning_tokens']:>6} {r['retries']:>4}  {result}")  # fmt: skip


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=2, help="Läufe je Anfrage (LLM-Ergebnisse streuen)")
    parser.add_argument("--case", action="append", help="Nur diese Fälle (mehrfach angebbar)")
    parser.add_argument("--label", default="aktuell", help="Name der Variante im Bericht")
    parser.add_argument("--output", type=Path, help="Bericht zusätzlich als JSON speichern")
    args = parser.parse_args(argv)
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    settings = get_settings()
    try:
        runs = await run(cases, args.repeat)
    finally:
        await engine.dispose()
    report = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": {
            "label": args.label,
            "llm_model": settings.llm_model,
            "reasoning_effort": settings.llm_reasoning_effort or None,
            "repeat": args.repeat,
        },
        "summary": summarize(runs),
        "cases": [asdict(r) | {"passed": r.passed} for r in runs],
    }
    print_report(report)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBericht gespeichert: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
