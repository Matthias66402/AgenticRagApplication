"""Steuerung eines Chat-Durchlaufs als LangGraph-Graph.

Der PydanticAI-Planer läuft unverändert als Knoten `planner`; danach verzweigt der Graph nach der Art des Ergebnisses
in einen `finish_*`-Knoten für die Nachbearbeitung:

    START → planner → finish_plan | finish_question | finish_answer | finish_error → END

UI-Events (`token`, `tool_call`, `tool_result`) gibt der Planer-Knoten über den Stream-Writer aus
(stream_mode "custom"); das Ergebnis steht im State. Die Abhängigkeiten (HTTP-Client, Session-Factory) sind nicht
serialisierbar und kommen deshalb als Laufzeit-Kontext statt im State.
"""

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic_ai import (
    AgentRunResultEvent,
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    UsageLimits,
)
from pydantic_ai.exceptions import AgentRunError, ModelRateLimitError

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AdventurePlan, ClarificationQuestion, TextAnswer, TrainTripPlan
from app.agent.planner import PlannerOutput, planner_agent
from app.agent.tools.transit import format_train_queries

RESULT_PREVIEW_CHARS = 600
USAGE_LIMITS = UsageLimits(request_limit=40)


class PlannerState(TypedDict, total=False):
    message: str
    # Bisheriger Verlauf im PydanticAI-Format (Adventure.agent_history)
    history_json: list[Any]
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
        return {
            "type": "tool_call",
            "id": event.part.tool_call_id,
            "tool": event.part.tool_name,
            "args": event.part.args_as_dict(),
        }
    if isinstance(event, FunctionToolResultEvent):
        content = getattr(event.part, "content", None)
        failed = isinstance(content, dict) and "error" in content
        return {"type": "tool_result", "id": event.part.tool_call_id, "result": _preview(content), "failed": failed}
    return None


async def planner(state: PlannerState, runtime: Runtime[AgentDeps]) -> PlannerState:
    """Führt den PydanticAI-Planer aus und streamt dessen Arbeit als UI-Events."""
    history_json = state.get("history_json")
    history = ModelMessagesTypeAdapter.validate_python(history_json) if history_json else None
    try:
        async with planner_agent.run_stream_events(
            state["message"], deps=runtime.context, message_history=history, usage_limits=USAGE_LIMITS
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
    graph.add_node("planner", planner)
    graph.add_node("finish_plan", finish_plan)
    graph.add_node("finish_question", finish_question)
    graph.add_node("finish_answer", finish_answer)
    graph.add_node("finish_error", finish_error)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_output)
    for node in ("finish_plan", "finish_question", "finish_answer", "finish_error"):
        graph.add_edge(node, END)
    return graph.compile(name="adventure-planner-graph")


planner_graph = build_graph()
