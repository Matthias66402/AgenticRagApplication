"""Misst Aufwand und Ergebnisqualität des Planungs-Agenten anhand fester Anfragen.

Aufruf: ``python -m app.evaluation.planner [--repeat 2] [--case ID] [--label NAME] [--output bericht.json]``

Jede Anfrage läuft wie im Chat über den LangGraph-Graphen (echte Modelle, echte Tools) als erste Nachricht eines
neuen Abenteuers. Gemessen:
- Aufwand: Laufzeit, Modellaufrufe (auch je Modell, wegen des Anfragelimits je Modell), Eingabe-/Cache-/Ausgabe-/
  Reasoning-Tokens und Kosten von Intake, Planer, Recherche-Agent und Relevanzprüfung der Wissensbasis
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
from collections.abc import AsyncGenerator, Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic_ai import (
    Agent, ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, RetryFeedbackPart, RunContext,
    ToolCallPart, ToolReturnPart,
)  # fmt: skip
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from app.agent.deps import AgentDeps
from app.agent.graph import planner_graph
from app.agent.intake import intake_agent
from app.agent.plan_schema import AdventurePlan, ClarificationQuestion, TextAnswer, TrainTripPlan
from app.agent.planner import planner_agent
from app.agent.research import research_agent
from app.config import get_settings
from app.db import SessionLocal, engine
from app.rag.relevance import get_relevance_agent

CASES_FILE = Path(__file__).parent / "planner_cases.json"
OUTPUT_NAMES = {
    AdventurePlan: "final_plan",
    TrainTripPlan: "final_train_trip",
    ClarificationQuestion: "ask_user",
    TextAnswer: "answer_user",
}
# Pflichtsuchen des Recherche-Agenten vor jedem neuen Plan (Region + Ausrüstung/Sicherheit, app.agent.research)
MIN_KNOWLEDGE_SEARCHES = 2
# (Tool, Ergebnisliste) mit Unterkünften, die der Plan übernehmen darf
ACCOMMODATION_RESULTS = [("find_accommodation", "results"), ("plan_hut_tour", "huts"), ("plan_hut_tour", "other_huts")]


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
    # Modellname -> Aufrufe; OpenAI zählt das Anfragelimit je Modell
    requests_by_model: dict[str, int] = field(default_factory=dict)
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
    # Zum Nachvollziehen: Tool-Aufrufe in Reihenfolge (Argumente, Ergebnis gekürzt) und Rückmeldungen der Retries
    calls: list[dict[str, Any]] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    # Dauer je Tool-Aufruf (tool_call_id -> s, vom Aufruf bis zum Ergebnis) und Zeitpunkt, zu dem jeder Graph-Knoten
    # fertig war (s ab Start), z. B. Ende der Vorab-Phase und des Planers
    tool_durations: dict[str, float] = field(default_factory=dict)
    node_done_seconds: dict[str, float] = field(default_factory=dict)
    # Was das Intake erkannt hat (Orte, Recherche-Auftrag, Startdatum, Dauer)
    intake: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(self.checks.values())


def retry_feedback(part: Any) -> str | None:
    """Rückmeldung eines Retrys an das Modell, für andere Nachrichtenteile None.

    Retries eines Tool-Aufrufs sind ToolReturnParts mit outcome="retried", Retries der Ausgabe RetryFeedbackParts.
    """
    if isinstance(part, ToolReturnPart) and part.outcome == "retried":
        return part.model_response_str()
    if isinstance(part, RetryFeedbackPart):
        return part.model_response()
    return None


def tool_returns(messages: list[ModelMessage], tool_name: str) -> list[Any]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name and retry_feedback(part) is None
    ]


def check_output(case: dict[str, Any], output: Any, messages: list[ModelMessage], deps: AgentDeps) -> dict[str, bool]:
    """Fallabhängige Qualitätsprüfungen; der Ausgabetyp wird immer geprüft."""
    checks = {"output_type": OUTPUT_NAMES.get(type(output)) == case["expected_output"]}
    calls = [p.tool_name for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)]
    if isinstance(output, AdventurePlan):
        checks["transport"] = output.transport.value == case.get("transport", output.transport.value)
        checks["duration_days"] = output.duration_days == case.get("duration_days", output.duration_days)
        searches = sum(
            len(result.get("queries", [])) for result in tool_returns(messages, "research_knowledge")
            if isinstance(result, dict)
        )
        checks["knowledge_searches"] = searches >= MIN_KNOWLEDGE_SEARCHES
        checks["packing_and_safety"] = bool(output.packing_tips) and bool(output.safety_notes)
        checks["sources"] = bool(output.sources)
        checks["route_ids"] = bool(output.route_ids) and all(r in deps.routes for r in output.route_ids)
        if output.duration_days > 1:
            found = {
                entry["name"]
                for tool_name, key in ACCOMMODATION_RESULTS
                for result in tool_returns(messages, tool_name)
                if isinstance(result, dict)
                for entry in result.get(key, [])
            }
            stays = [s.accommodation for s in output.stages[:-1]]
            checks["accommodation_from_tool"] = all(a is not None and a.name in found for a in stays)
        if case.get("named_route"):
            # Mindestens eine Route des Plans folgt dem Fernweg ohne Teilstücke abseits davon
            along = {
                result["route_id"]
                for result in tool_returns(messages, "plan_route")
                if isinstance(result, dict) and result.get("named_route", {}).get("off_route_legs") == []
            }
            checks["named_route"] = any(route_id in along for route_id in output.route_ids)
        if case.get("journeys"):
            checks["journeys"] = output.outbound_journey is not None and output.return_journey is not None
    if isinstance(output, TrainTripPlan) and case.get("access"):
        checks["access"] = output.access is not None
    if isinstance(output, TextAnswer):
        checks["train_query"] = "find_train_connections" in calls
    return checks


def measure(run: CaseRun, messages: list[ModelMessage]) -> None:
    """Tool-Aufrufe und Retries aus dem Verlauf des Planers (inkl. vom Graphen eingefügter Tool-Aufrufe)."""
    calls_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    run.tool_calls[part.tool_name] = run.tool_calls.get(part.tool_name, 0) + 1
                    call = {"tool": part.tool_name, "id": part.tool_call_id, "args": part.args_as_dict()}
                    calls_by_id[part.tool_call_id] = call
                    run.calls.append(call)
        else:
            for part in message.parts:
                if (feedback := retry_feedback(part)) is not None:
                    run.retries += 1
                    run.feedback.append(feedback[:CALL_PREVIEW_CHARS])
                elif isinstance(part, ToolReturnPart) and (call := calls_by_id.get(part.tool_call_id)):
                    call["result"] = _short(part.content)


CALL_PREVIEW_CHARS = 1500


def _short(content: Any) -> str:
    # Lange Checkpoint-Listen von plan_route nur zählen, damit named_route und note nicht abgeschnitten werden
    if isinstance(content, dict) and isinstance(content.get("named_route"), dict):
        named = content["named_route"]
        content = content | {"named_route": named | {"checkpoints": len(named.get("checkpoints") or [])}}
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    return text if len(text) <= CALL_PREVIEW_CHARS else text[:CALL_PREVIEW_CHARS] + " …"


def measure_usage(run: CaseRun, responses: list[ModelResponse]) -> None:
    """Aufrufe, Tokens und Kosten aller aufgezeichneten Modellantworten (Planer, Recherche-Agent, …)."""
    for response in responses:
        usage = response.usage
        run.requests += 1
        model = response.model_name or "unbekannt"
        run.requests_by_model[model] = run.requests_by_model.get(model, 0) + 1
        run.input_tokens += usage.input_tokens
        run.cache_read_tokens += usage.cache_read_tokens
        run.output_tokens += usage.output_tokens
        run.reasoning_tokens += usage.details.get("reasoning_tokens", 0)
        try:
            run.cost_usd += float(response.cost().total_price)
        except Exception:  # Modell ohne Preisangabe (z. B. in Tests)
            pass


class RecordingModel(WrapperModel):
    """Zeichnet jede Modellantwort auf, auch die von Agenten, die als Tool oder eigener Knoten laufen."""

    def __init__(self, wrapped: Model | str, responses: list[ModelResponse]) -> None:
        super().__init__(wrapped)
        self.responses = responses

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        self.responses.append(response)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            yield stream
        self.responses.append(stream.get())


# Agenten, deren Aufwand zählt. Die Relevanzprüfung gehört dazu: Sie belastet das Anfragelimit ihres Modells und
# bremst darüber die Recherche (ihre Qualität bewertet app.evaluation.retrieval).
MEASURED_AGENTS: list[Agent[Any, Any]] = [intake_agent, planner_agent, research_agent, get_relevance_agent()]


@contextmanager
def recording(responses: list[ModelResponse]) -> Iterator[None]:
    with ExitStack() as stack:
        for agent in MEASURED_AGENTS:
            # Das gerade wirksame Modell inkl. eines äußeren override (z. B. in Tests); agent.model kennt das nicht.
            # Private Methode von PydanticAI, öffentlich gibt es dafür nichts.
            try:
                model = agent._get_model_outside_run()  # pyright: ignore[reportPrivateUsage]
            except UserError:  # ohne Modell abgeschaltet, z. B. RELEVANCE_MODEL leer
                continue
            stack.enter_context(agent.override(model=RecordingModel(model, responses)))
        yield


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
    tool_durations: dict[str, float] = {}
    responses: list[ModelResponse] = []
    state: dict[str, Any] = {}
    modes = ["custom", "values", "updates"]
    try:
        with recording(responses):
            graph_input = {"message": case["message"], "history_json": []}
            async for mode, chunk in planner_graph.astream(graph_input, context=deps, stream_mode=modes):
                now = time.perf_counter()
                if mode == "values":
                    state = chunk
                elif mode == "updates":
                    for node in chunk:  # Knoten fertig; parallele Knoten melden sich einzeln
                        run.node_done_seconds[node] = round(now - started, 1)
                elif chunk["type"] == "tool_call":
                    tool_started[chunk["id"]] = now
                elif chunk["type"] == "tool_result" and chunk["id"] in tool_started:
                    tool_spans.append((tool_started[chunk["id"]], now))
                    tool_durations[chunk["id"]] = round(now - tool_started.pop(chunk["id"]), 1)
    except Exception as exc:  # Abbruch zählt als nicht bestanden, die übrigen Fälle laufen weiter
        run.error = f"{type(exc).__name__}: {exc}"
    run.seconds = round(time.perf_counter() - started, 1)
    run.tool_seconds = round(covered_seconds(tool_spans), 1)
    run.tool_durations = tool_durations
    if (prefetch := state.get("prefetch")) is not None:
        run.intake = prefetch.model_dump(mode="json")
    measure_usage(run, responses)
    run.cost_usd = round(run.cost_usd, 4)
    output = state.get("output")
    if output is None or not state.get("new_history_json"):
        run.error = run.error or state.get("error") or "Kein Ergebnis"
        return run
    messages = ModelMessagesTypeAdapter.validate_python(state["new_history_json"])
    measure(run, messages)
    run.output = OUTPUT_NAMES.get(type(output))
    run.text = (
        output.question if isinstance(output, ClarificationQuestion)
        else output.message if isinstance(output, TextAnswer)
        else output.title
    )  # fmt: skip
    for call in run.calls:
        if (seconds := tool_durations.get(call["id"])) is not None:
            call["seconds"] = seconds
    run.checks = check_output(case, output, messages, deps)
    return run


PREFETCH_NODES = ("intake", "prefetch_geocode", "prefetch_research")


def phase_seconds(run: CaseRun) -> dict[str, float]:
    """Dauer der Phasen eines Laufs: Vorab-Phase (Intake und prefetch_*) und Planer."""
    done = run.node_done_seconds
    prefetch_end = max((done[n] for n in PREFETCH_NODES if n in done), default=0.0)
    phases = {"intake": done.get("intake", 0.0), "prefetch": prefetch_end}
    if "planner" in done:
        phases["planner"] = round(done["planner"] - prefetch_end, 1)
    return phases


def tool_time_stats(runs: list[CaseRun]) -> dict[str, dict[str, float]]:
    """Je Tool: Anzahl Aufrufe, mittlere und längste Dauer, Summe (s); Vorab-Aufrufe getrennt mit Präfix 'vorab:'."""
    durations: dict[str, list[float]] = {}
    for r in runs:
        for call in r.calls:
            if "seconds" in call:
                name = ("vorab:" if call["id"].startswith("pre_") else "") + call["tool"]
                durations.setdefault(name, []).append(call["seconds"])
    return {
        name: {
            "calls": len(values),
            "mean_s": round(statistics.mean(values), 1),
            "max_s": round(max(values), 1),
            "total_s": round(sum(values), 1),
        }
        for name, values in sorted(durations.items(), key=lambda item: -sum(item[1]))
    }


def summarize(runs: list[CaseRun]) -> dict[str, Any]:
    def mean(attr: str) -> float:
        return round(statistics.mean(getattr(r, attr) for r in runs), 1) if runs else 0.0

    models = sorted({model for r in runs for model in r.requests_by_model})
    return {
        "runs": len(runs),
        "passed": sum(r.passed for r in runs),
        "errors": sum(r.error is not None for r in runs),
        "retries": sum(r.retries for r in runs),
        "mean_seconds": mean("seconds"),
        "mean_tool_seconds": mean("tool_seconds"),
        "mean_requests": mean("requests"),
        "mean_requests_by_model": {
            model: round(statistics.mean(r.requests_by_model.get(model, 0) for r in runs), 1) for model in models
        },
        "mean_input_tokens": mean("input_tokens"),
        "mean_cache_read_tokens": mean("cache_read_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_reasoning_tokens": mean("reasoning_tokens"),
        "total_cost_usd": round(sum(r.cost_usd for r in runs), 3),
        "mean_phase_seconds": {
            phase: round(statistics.mean(phase_seconds(r).get(phase, 0.0) for r in runs), 1)
            for phase in ("intake", "prefetch", "planner")
        } if runs else {},
        "tool_seconds": tool_time_stats(runs),
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
    by_model = " · ".join(f"{model} {n:.1f}" for model, n in s["mean_requests_by_model"].items())
    print(f"Mittlere Aufrufe je Modell: {by_model or '–'}")
    if phases := s.get("mean_phase_seconds"):
        print(f"Phasen (Mittel): Intake fertig nach {phases['intake']:.0f} s · Vorab-Phase fertig nach "
              f"{phases['prefetch']:.0f} s · Planer danach {phases['planner']:.0f} s")  # fmt: skip
    if tools := s.get("tool_seconds"):
        print(f"\n{'Tool':28} {'Aufrufe':>7} {'Mittel':>7} {'Max':>6} {'Summe':>7}")
        for name, t in tools.items():
            print(f"{name:28} {t['calls']:>7} {t['mean_s']:>6.1f}s {t['max_s']:>5.0f}s {t['total_s']:>6.0f}s")
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
            "research_model": settings.research_model,
            "intake_model": settings.intake_model or None,
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
