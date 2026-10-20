from typing import Any

from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.tools.common import tool_error
from app.rag.retrieve import search_chunks


async def search_knowledge(ctx: RunContext[AgentDeps], query: str) -> list[dict[str, Any]] | dict[str, Any]:
    """Durchsucht die Wissensbasis (RAG) nach Regeln, Tipps und Hintergrundwissen.

    Beispiele: Hüttenregeln und Reservierung, Wildcampen-Regeln je Land, Ausrüstungslisten,
    Sicherheit bei Gewitter, Tipps für Radreisen oder Kanutouren.
    Mehrere gezielte Suchen sind besser als eine sehr allgemeine.

    Args:
        query: Konkrete Suchanfrage in natürlicher Sprache.
    """
    try:
        async with ctx.deps.session_factory() as session:
            chunks = await search_chunks(session, query)
    except Exception as exc:  # Embedding-Anbieter oder DB nicht erreichbar
        return tool_error(f"Wissensbasis nicht verfügbar: {exc}")
    if not chunks:
        return tool_error("Nichts Passendes in der Wissensbasis gefunden.")
    return [c.model_dump() for c in chunks]
