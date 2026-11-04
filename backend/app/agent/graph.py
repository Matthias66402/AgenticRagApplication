"""Steuerung eines Chat-Durchlaufs als LangGraph-Graph.

    START → intake ─┬─(Orte)──────→ prefetch_geocode ──┐
                    ├─(neue Tour)─→ prefetch_research ─┤  (parallel)
                    └─(sonst)──────────────────────────┴→ planner → finish_plan | finish_question
                                                                    | finish_answer | finish_error → END

- `intake`: Ein kleines LLM bestimmt, was der Planer für diese Nachricht sicher braucht (app.agent.intake).
- `prefetch_geocode` / `prefetch_research`: führen `geocode` bzw. `research_knowledge` ohne Planer-Runde aus.
- `planner`: Der PydanticAI-Planer; Vorab-Aufrufe stehen als gewöhnliche Tool-Aufrufe in seinem Verlauf, so wirken
  Validatoren, gespeicherter Verlauf und Evaluation unverändert. Er kann jedes Tool trotzdem selbst aufrufen.
- `finish_*`: Nachbearbeitung je nach Art des Ergebnisses.

UI-Events (`token`, `tool_call`, `tool_result`) gehen über den Stream-Writer (stream_mode "custom"); das Ergebnis
steht im State. Die Abhängigkeiten (HTTP-Client, Session-Factory) sind nicht serialisierbar und kommen deshalb als
Laufzeit-Kontext statt im State.
"""

import asyncio
import json
import logging
import operator
import socket
import uuid
from types import SimpleNamespace
from typing import Annotated, Any, Literal, TypedDict

import httpx
import openai
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic_ai import (
    AgentRunResultEvent,
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    RunUsage,
    UsageLimits,
    UserPromptPart,
)
from pydantic_ai.exceptions import AgentRunError, ModelConnectionError, ModelRateLimitError, ModelRequestTimeout

from app.agent.deps import AgentDeps
from app.agent.intake import Prefetch, plan_prefetch
from app.agent.plan_schema import AdventurePlan, ClarificationQuestion, TextAnswer, TrainTripPlan
from app.agent.planner import PlannerOutput, planner_agent
from app.agent.tools.geocode import geocode
from app.agent.tools.research import research_knowledge
from app.agent.tools.transit import format_train_queries

logger = logging.getLogger(__name__)

RESULT_PREVIEW_CHARS = 600
USAGE_LIMITS = UsageLimits(request_limit=40)

# Statt "Request timed out": Meist war die Internet- bzw. DNS-Verbindung kurz weg (z. B. nach dem Ruhezustand oder
# einem WLAN-Wechsel), dann scheitern Sprachmodell und Tools gleichzeitig ("Temporary failure in name resolution")
NETWORK_ERROR_MESSAGE = (
    "Keine Verbindung zum Sprachmodell (Zeitüberschreitung bzw. Netzwerkfehler). Meist ist die Internet- oder "
    "DNS-Verbindung kurz unterbrochen. Bitte die Verbindung prüfen und die Nachricht erneut senden."
)
NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    ModelConnectionError, ModelRequestTimeout, httpx.TransportError, socket.gaierror, TimeoutError,
    openai.APIConnectionError,
)  # fmt: skip


def is_network_error(exc: BaseException) -> bool:
    """Zeitüberschreitung oder Verbindungsfehler, auch als Ursache einer anderen Ausnahme (z. B. im Stream)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, NETWORK_ERRORS):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class PrefetchedCall(TypedDict):
    """Vom Graphen statt vom Planer ausgeführter Tool-Aufruf."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    result: Any


class PlannerState(TypedDict, total=False):
    message: str
    # Bisheriger Verlauf im PydanticAI-Format (Adventure.agent_history)
    history_json: list[Any]
    # Vom Intake bestimmte Vorab-Aufrufe
    prefetch: Prefetch
    # Ergebnisse der parallelen prefetch_*-Knoten; der Reducer fügt ihre Listen zusammen
    prefetched: Annotated[list[PrefetchedCall], operator.add]
    output: PlannerOutput | None
    # Verlauf inkl. dieses Laufs, None bei Fehler
    new_history_json: list[Any] | None
    # Antwort ohne Plan, inkl. der in diesem Lauf gefundenen Bahnverbindungen
    answer: str | None
    error: str | None


def _preview(content: Any) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    return text if len(text) <= RESULT_PREVIEW_CHARS else text[:RESULT_PREVIEW_CHARS] + " …"


def ui_event(event: AgentStreamEvent | AgentRunResultEvent[Any]) -> dict[str, Any] | None:
    """Übersetzt ein PydanticAI-Event in ein UI-Event; None für Events ohne Entsprechung."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content:
        return {"type": "token", "content": event.part.content}
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return {"type": "token", "content": event.delta.content_delta}
    if isinstance(event, FunctionToolCallEvent):
        return tool_call_event(event.part.tool_call_id, event.part.tool_name, event.part.args_as_dict())
    if isinstance(event, FunctionToolResultEvent):
        return tool_result_event(event.part.tool_call_id, getattr(event.part, "content", None))
    return None


def tool_call_event(tool_call_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call", "id": tool_call_id, "tool": tool_name, "args": args}


def tool_result_event(tool_call_id: str, content: Any) -> dict[str, Any]:
    failed = isinstance(content, dict) and "error" in content
    return {"type": "tool_result", "id": tool_call_id, "result": _preview(content), "failed": failed}


async def intake(state: PlannerState, runtime: Runtime[AgentDeps]) -> PlannerState:
    """Bestimmt die Vorab-Aufrufe. Fehler sind unkritisch: Der Planer ruft die Tools dann selbst auf."""
    try:
        prefetch = await plan_prefetch(state["message"], state.get("history_json") or [], runtime.context)
    except Exception as exc:
        logger.warning("Intake fehlgeschlagen, Planer startet ohne Vorab-Aufrufe: %s: %s", type(exc).__name__, exc)
        prefetch = Prefetch()
    return {"prefetch": prefetch}


def route_intake(state: PlannerState) -> list[Literal["prefetch_geocode", "prefetch_research", "planner"]]:
    """Die prefetch_*-Knoten laufen parallel; der Planer startet erst, wenn alle fertig sind."""
    prefetch = state.get("prefetch") or Prefetch()
    targets: list[Literal["prefetch_geocode", "prefetch_research", "planner"]] = []
    if prefetch.places:
        targets.append("prefetch_geocode")
    if prefetch.research:
        targets.append("prefetch_research")
    return targets or ["planner"]


def _tool_context(runtime: Runtime[AgentDeps]) -> Any:
    """Ersatz für den RunContext eines Tools: Die Tools nutzen davon nur deps und usage."""
    return SimpleNamespace(deps=runtime.context, usage=RunUsage())


async def _run_prefetched(runtime: Runtime[AgentDeps], tool_name: str, args: BaseModel, tool: Any) -> PrefetchedCall:
    """Führt ein Tool wie der Planer aus, inkl. UI-Events; Ausnahmen werden zu Tool-Fehlern.

    Verlauf und Events bekommen die Argumente als JSON, das Tool die Python-Werte (z. B. Enums wie bei PydanticAI).
    """
    call: PrefetchedCall = {
        "tool_call_id": f"pre_{uuid.uuid4().hex[:16]}", "tool_name": tool_name,
        "args": args.model_dump(mode="json", exclude_none=True), "result": None,
    }  # fmt: skip
    runtime.stream_writer(tool_call_event(call["tool_call_id"], tool_name, call["args"]))
    try:
        call["result"] = await tool(_tool_context(runtime), **args.model_dump(exclude_none=True))
    except Exception as exc:  # wie ein Tool-Fehler: der Planer kann es selbst erneut versuchen
        call["result"] = {"error": f"Vorab-Aufruf fehlgeschlagen: {type(exc).__name__}: {exc}"}
    runtime.stream_writer(tool_result_event(call["tool_call_id"], call["result"]))
    return call


async def prefetch_geocode(state: PlannerState, runtime: Runtime[AgentDeps]) -> PlannerState:
    """Sucht die Orte aus dem Intake (Nominatim drosselt selbst auf 1 Anfrage/s)."""
    places = state["prefetch"].places
    calls = await asyncio.gather(*(_run_prefetched(runtime, "geocode", place, geocode) for place in places))
    return {"prefetched": list(calls)}


async def prefetch_research(state: PlannerState, runtime: Runtime[AgentDeps]) -> PlannerState:
    """Lässt den Recherche-Agenten die Wissensbasis durchsuchen, parallel zum Geocoding."""
    research = state["prefetch"].research
    assert research is not None
    return {"prefetched": [await _run_prefetched(runtime, "research_knowledge", research, research_knowledge)]}


def planner_history(state: PlannerState) -> tuple[str | None, list[ModelMessage] | None]:
    """Prompt und Verlauf für den Planer. Mit Vorab-Aufrufen steht die Nachricht schon im Verlauf, gefolgt von den
    Aufrufen und ihren Ergebnissen, als hätte der Planer sie selbst ausgeführt; er läuft dann ohne neuen Prompt."""
    history_json = state.get("history_json")
    history = ModelMessagesTypeAdapter.validate_python(history_json) if history_json else []
    # Feste Reihenfolge (Orte vor Recherche), unabhängig davon, welcher parallele Knoten zuerst fertig war
    prefetched = sorted(state.get("prefetched") or [], key=lambda c: c["tool_name"] != "geocode")
    if not prefetched:
        return state["message"], history or None
    history += [
        ModelRequest(parts=[UserPromptPart(state["message"])]),
        ModelResponse(parts=[ToolCallPart(c["tool_name"], c["args"], tool_call_id=c["tool_call_id"]) for c in prefetched]),
        ModelRequest(
            parts=[ToolReturnPart(c["tool_name"], c["result"], tool_call_id=c["tool_call_id"]) for c in prefetched]
        ),
    ]
    return None, history


async def planner(state: PlannerState, runtime: Runtime[AgentDeps]) -> PlannerState:
    """Führt den PydanticAI-Planer aus und streamt dessen Arbeit als UI-Events."""
    prompt, history = planner_history(state)
    # Startdatum aus dem Intake: über die Anweisungen dieses Laufs, nicht als Teil der Nutzernachricht
    runtime.context.intake_note = (state.get("prefetch") or Prefetch()).planner_note()
    try:
        async with planner_agent.run_stream_events(
            prompt, deps=runtime.context, message_history=history, usage_limits=USAGE_LIMITS
        ) as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    return {
                        "output": event.result.output,
                        "new_history_json": json.loads(event.result.all_messages_json()),
                    }
                if (ui := ui_event(event)) is not None:
                    runtime.stream_writer(ui)
    except ModelRateLimitError:
        # Trotz Drosselung möglich, z. B. wenn ein anderer Prozess (Evaluation) dasselbe Limit nutzt
        return {"error": "Das Sprachmodell ist gerade ausgelastet (Anfragelimit erreicht). Bitte in einer Minute erneut senden."}
    except Exception as exc:
        if is_network_error(exc):
            logger.warning("Planer ohne Verbindung zum Sprachmodell: %s: %s", type(exc).__name__, exc)
            return {"error": NETWORK_ERROR_MESSAGE}
        if isinstance(exc, AgentRunError):
            return {"error": f"Der Agent konnte die Anfrage nicht abschließen: {exc}"}
        # Anbieterfehler sollen den Stream sauber beenden
        return {"error": f"Fehler bei der Planung: {type(exc).__name__}: {exc}"}
    return {}  # Stream ohne Ergebnis-Event: finish_error, Antworttext wie bisher aus app.api.chat


def route_output(
    state: PlannerState,
) -> Literal["finish_plan", "finish_question", "finish_answer", "finish_error"]:
    output = state.get("output")
    if state.get("error") or output is None:
        return "finish_error"
    if isinstance(output, (AdventurePlan, TrainTripPlan)):
        return "finish_plan"
    if isinstance(output, ClarificationQuestion):
        return "finish_question"
    return "finish_answer"


async def finish_plan(state: PlannerState) -> PlannerState:
    """Plan bzw. Reisekarte; Aufbereitung und Speichern übernimmt bisher app.api.chat."""
    return {}


async def finish_question(state: PlannerState) -> PlannerState:
    return {}


async def finish_answer(state: PlannerState, runtime: Runtime[AgentDeps]) -> PlannerState:
    """Antwort ohne Plan: Die in diesem Lauf gefundenen Bahnverbindungen werden darunter angehängt."""
    output = state["output"]
    assert isinstance(output, TextAnswer)
    answer = output.message.strip()
    if queries := list(runtime.context.train_queries.values()):
        answer += "\n\n" + format_train_queries(queries)
    return {"answer": answer}


async def finish_error(state: PlannerState) -> PlannerState:
    return {}


def build_graph() -> CompiledStateGraph[PlannerState, AgentDeps, PlannerState, PlannerState]:
    graph = StateGraph(PlannerState, context_schema=AgentDeps)
    graph.add_node("intake", intake)
    graph.add_node("prefetch_geocode", prefetch_geocode)
    graph.add_node("prefetch_research", prefetch_research)
    graph.add_node("planner", planner)
    graph.add_node("finish_plan", finish_plan)
    graph.add_node("finish_question", finish_question)
    graph.add_node("finish_answer", finish_answer)
    graph.add_node("finish_error", finish_error)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_intake, ["prefetch_geocode", "prefetch_research", "planner"])
    # Beide Vorab-Knoten münden in den Planer; LangGraph startet ihn einmal, wenn alle im Schritt fertig sind
    graph.add_edge("prefetch_geocode", "planner")
    graph.add_edge("prefetch_research", "planner")
    graph.add_conditional_edges("planner", route_output)
    for node in ("finish_plan", "finish_question", "finish_answer", "finish_error"):
        graph.add_edge(node, END)
    return graph.compile(name="adventure-planner-graph")


planner_graph = build_graph()
