"""Führt den Agenten aus und übersetzt PydanticAI-Events in einfache, UI-taugliche Events."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    UsageLimits,
)
from pydantic_ai.exceptions import AgentRunError

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AdventurePlan, ClarificationQuestion
from app.agent.planner import planner_agent

RESULT_PREVIEW_CHARS = 600
USAGE_LIMITS = UsageLimits(request_limit=40)


@dataclass
class RunOutcome:
    """Wird während des Streams befüllt und danach vom Aufrufer gespeichert."""

    plan: AdventurePlan | None = None
    question: ClarificationQuestion | None = None
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    history_json: list[Any] | None = None
    error: str | None = None


def _preview(content: Any) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    return text if len(text) <= RESULT_PREVIEW_CHARS else text[:RESULT_PREVIEW_CHARS] + " …"


async def run_planner(
    message: str, history_json: list[Any], deps: AgentDeps, outcome: RunOutcome
) -> AsyncIterator[dict[str, Any]]:
    """Streamt Events: token, tool_call, tool_result, plan, question, error."""
    history = ModelMessagesTypeAdapter.validate_python(history_json) if history_json else None
    calls_by_id: dict[str, dict[str, Any]] = {}

    try:
        async with planner_agent.run_stream_events(
            message, deps=deps, message_history=history, usage_limits=USAGE_LIMITS
        ) as events:
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content:
                    outcome.text += event.part.content
                    yield {"type": "token", "content": event.part.content}
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    outcome.text += event.delta.content_delta
                    yield {"type": "token", "content": event.delta.content_delta}
                elif isinstance(event, FunctionToolCallEvent):
                    call = {"id": event.part.tool_call_id, "tool": event.part.tool_name, "args": event.part.args_as_dict()}
                    calls_by_id[call["id"]] = call
                    outcome.tool_calls.append(call)
                    yield {"type": "tool_call", **call}
                elif isinstance(event, FunctionToolResultEvent):
                    content = getattr(event.part, "content", None)
                    preview = _preview(content)
                    failed = isinstance(content, dict) and "error" in content
                    if call := calls_by_id.get(event.part.tool_call_id):
                        call["result"] = preview
                        call["failed"] = failed
                    yield {"type": "tool_result", "id": event.part.tool_call_id, "result": preview, "failed": failed}
                elif isinstance(event, AgentRunResultEvent):
                    output = event.result.output
                    outcome.history_json = json.loads(event.result.all_messages_json())
                    if isinstance(output, AdventurePlan):
                        outcome.plan = output
                    elif isinstance(output, ClarificationQuestion):
                        outcome.question = output
    except AgentRunError as exc:
        outcome.error = f"Der Agent konnte die Anfrage nicht abschließen: {exc}"
    except Exception as exc:  # Anbieter-/Netzwerkfehler sollen den Stream sauber beenden
        outcome.error = f"Fehler bei der Planung: {type(exc).__name__}: {exc}"

    if outcome.error:
        yield {"type": "error", "message": outcome.error}
