import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.agent.deps import AgentDeps
from app.agent.runner import RunOutcome, run_planner
from app.api.adventures import get_owned_adventure
from app.auth.deps import CurrentUser, SessionDep
from app.config import get_settings
from app.db import SessionLocal
from app.models import Adventure, AdventureStatus, ChatMessage
from app.schemas.adventure import ChatRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


def sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def build_stored_plan(outcome: RunOutcome, deps: AgentDeps, previous: dict[str, Any] | None) -> dict[str, Any]:
    """Plan + Routengeometrien für die Karte. Geometrien früherer Läufe bleiben erhalten."""
    assert outcome.plan is not None
    known_routes = (previous or {}).get("routes", {}) | deps.routes
    return outcome.plan.model_dump(mode="json") | {
        "routes": {rid: known_routes[rid] for rid in outcome.plan.route_ids if rid in known_routes}
    }


def assistant_text(outcome: RunOutcome) -> str:
    if outcome.question:
        return outcome.question.question
    if outcome.plan:
        return outcome.text.strip() or f"Hier ist Dein Plan „{outcome.plan.title}“. {outcome.plan.summary}"
    return outcome.error or outcome.text.strip() or "Keine Antwort erhalten."


async def save_outcome(adventure_id: int, user_message: str, outcome: RunOutcome, deps: AgentDeps) -> None:
    async with SessionLocal() as session:
        adventure = await session.get(Adventure, adventure_id)
        if adventure is None:  # während des Laufs gelöscht
            return
        session.add(ChatMessage(adventure_id=adventure_id, role="user", content=user_message, tool_calls=[]))
        session.add(
            ChatMessage(
                adventure_id=adventure_id,
                role="assistant",
                content=assistant_text(outcome),
                tool_calls=outcome.tool_calls,
            )
        )
        if outcome.history_json is not None:
            adventure.agent_history = outcome.history_json
        if outcome.plan is not None:
            adventure.plan = build_stored_plan(outcome, deps, adventure.plan)
            adventure.title = outcome.plan.title
            adventure.status = AdventureStatus.planned
        await session.commit()


@router.post("/{adventure_id}")
async def chat(
    adventure_id: int, data: ChatRequest, user: CurrentUser, session: SessionDep, request: Request
) -> StreamingResponse:
    """Schickt eine Nachricht an den Planungs-Agenten und streamt dessen Arbeit als Server-Sent Events.

    Events: `token`, `tool_call`, `tool_result`, `question`, `plan`, `error`, `done`.
    """
    adventure = await get_owned_adventure(session, adventure_id, user.id)
    history = list(adventure.agent_history or [])
    user_name = user.display_name
    # Die Request-Session wird nicht in den Stream mitgenommen; Tools und Speichern öffnen eigene Sessions.
    await session.close()

    async def stream() -> AsyncIterator[str]:
        settings = get_settings()
        outcome = RunOutcome()
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            deps = AgentDeps(settings=settings, http=http, session_factory=SessionLocal, user_name=user_name)
            async with aclosing(run_planner(data.message, history, deps, outcome)) as events:
                async for event in events:
                    if await request.is_disconnected():
                        break
                    yield sse(event)

        await save_outcome(adventure_id, data.message, outcome, deps)
        if outcome.plan is not None:
            yield sse({"type": "plan", "plan": build_stored_plan(outcome, deps, adventure.plan)})
        elif outcome.question is not None:
            yield sse({"type": "question", **outcome.question.model_dump()})
        yield sse({"type": "done"})

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
