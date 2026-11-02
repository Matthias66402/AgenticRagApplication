"""Steuerung eines Chat-Durchlaufs als LangGraph-Graph.

    START → intake ─(Orte erkannt)→ prefetch_geocode ─┐
                   └───────────────────────────────────┴→ planner → finish_plan | finish_question
                                                                    | finish_answer | finish_error → END

- `intake`: Ein kleines LLM bestimmt die Orte, die der Planer für diese Nachricht nachschlagen muss (app.agent.intake).
- `prefetch_geocode`: sucht sie mit dem `geocode`-Tool, ohne Planer-Runde.
- `planner`: Der PydanticAI-Planer; vorab gesuchte Orte stehen als gewöhnliche `geocode`-Aufrufe in seinem Verlauf,
  so wirken Validatoren, gespeicherter Verlauf und Evaluation unverändert. Er kann jeden Ort trotzdem selbst suchen.
- `finish_*`: Nachbearbeitung je nach Art des Ergebnisses.

UI-Events (`token`, `tool_call`, `tool_result`) gehen über den Stream-Writer (stream_mode "custom"); das Ergebnis
steht im State. Die Abhängigkeiten (HTTP-Client, Session-Factory) sind nicht serialisierbar und kommen deshalb als
Laufzeit-Kontext statt im State.
"""

import asyncio
import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
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
    UsageLimits,
    UserPromptPart,
)
from pydantic_ai.exceptions import AgentRunError, ModelRateLimitError

from app.agent.deps import AgentDeps
from app.agent.intake import PlaceQuery, places_to_geocode
from app.agent.plan_schema import AdventurePlan, ClarificationQuestion, TextAnswer, TrainTripPlan
from app.agent.planner import PlannerOutput, planner_agent
from app.agent.tools.geocode import geocode
from app.agent.tools.transit import format_train_queries

logger = logging.getLogger(__name__)

RESULT_PREVIEW_CHARS = 600
USAGE_LIMITS = UsageLimits(request_limit=40)


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
    # Vom Intake erkannte Orte für die Vorab-Suche
    places: list[PlaceQuery]
    prefetched: list[PrefetchedCall]
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
    """Bestimmt die Orte für die Vorab-Suche. Fehler sind unkritisch: Der Planer sucht dann selbst."""
    try:
        places = await places_to_geocode(state["message"], state.get("history_json") or [], runtime.context)
    except Exception as exc:
        logger.warning("Intake fehlgeschlagen, Planer startet ohne Vorab-Suche: %s: %s", type(exc).__name__, exc)
        places = []
    return {"places": places}


def route_intake(state: PlannerState) -> Literal["prefetch_geocode", "planner"]:
    return "prefetch_geocode" if state.get("places") else "planner"


async def prefetch_geocode(state: PlannerState, runtime: Runtime[AgentDeps]) -> PlannerState:
    """Sucht die Orte aus dem Intake mit dem geocode-Tool (Nominatim drosselt selbst auf 1 Anfrage/s)."""
    # Die Tools nutzen vom RunContext nur deps
    ctx: Any = SimpleNamespace(deps=runtime.context)
    calls: list[PrefetchedCall] = []
    for place in state.get("places", []):
        args = place.model_dump(exclude_none=True)
        call: PrefetchedCall = {
            "tool_call_id": f"pre_{uuid.uuid4().hex[:16]}", "tool_name": "geocode", "args": args, "result": None,
        }  # fmt: skip
        runtime.stream_writer(tool_call_event(call["tool_call_id"], "geocode", args))
        calls.append(call)

    async def run(call: PrefetchedCall) -> None:
        try:
            call["result"] = await geocode(ctx, **call["args"])
        except Exception as exc:  # wie ein Tool-Fehler: der Planer kann es selbst erneut versuchen
            call["result"] = {"error": f"Vorab-Suche fehlgeschlagen: {type(exc).__name__}: {exc}"}
        runtime.stream_writer(tool_result_event(call["tool_call_id"], call["result"]))

    await asyncio.gather(*(run(call) for call in calls))
    return {"prefetched": calls}


def planner_history(state: PlannerState) -> tuple[str | None, list[ModelMessage] | None]:
    """Prompt und Verlauf für den Planer. Mit Vorab-Aufrufen steht die Nachricht schon im Verlauf, gefolgt von den
    Aufrufen und ihren Ergebnissen, als hätte der Planer sie selbst ausgeführt; er läuft dann ohne neuen Prompt."""
    history_json = state.get("history_json")
    history = ModelMessagesTypeAdapter.validate_python(history_json) if history_json else []
    prefetched = state.get("prefetched")
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
    except AgentRunError as exc:
        return {"error": f"Der Agent konnte die Anfrage nicht abschließen: {exc}"}
    except Exception as exc:  # Anbieter-/Netzwerkfehler sollen den Stream sauber beenden
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
    graph.add_node("planner", planner)
    graph.add_node("finish_plan", finish_plan)
    graph.add_node("finish_question", finish_question)
    graph.add_node("finish_answer", finish_answer)
    graph.add_node("finish_error", finish_error)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_intake)
    graph.add_edge("prefetch_geocode", "planner")
    graph.add_conditional_edges("planner", route_output)
    for node in ("finish_plan", "finish_question", "finish_answer", "finish_error"):
        graph.add_edge(node, END)
    return graph.compile(name="adventure-planner-graph")


planner_graph = build_graph()
