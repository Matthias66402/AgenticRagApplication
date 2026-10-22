from typing import Any

from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.tools.common import tool_error
from app.observability import observe
from app.rag.relevance import grade_hits
from app.rag.retrieve import MIN_SCORE, TOP_K, search_chunks


async def search_knowledge(ctx: RunContext[AgentDeps], query: str) -> list[dict[str, Any]] | dict[str, Any]:
    """Durchsucht die Wissensbasis (RAG) nach Regeln, Tipps und Hintergrundwissen.

    Beispiele: Radrouten in einer Region, Hüttenregeln und Reservierung, Wildcampen-Regeln je Land,
    Ausrüstungslisten, Sicherheit bei Gewitter, Tipps für Radreisen oder Kanutouren.
    Eine Suche je Thema: Sammelanfragen mit vielen Stichworten finden nur allgemeine Tipps.

    Args:
        query: Kurze Suchanfrage zu genau einem Thema, z. B. "Radrouten rund um Oldenburg".
    """
    with observe(
        "retrieve-knowledge", as_type="retriever", input=query, metadata={"top_k": TOP_K, "min_score": MIN_SCORE}
    ) as retrieval:
        try:
            async with ctx.deps.session_factory() as session:
                chunks = await search_chunks(session, query)
        except Exception as exc:  # Embedding-Anbieter oder DB nicht erreichbar
            retrieval.update(level="ERROR", status_message=str(exc))
            return tool_error(f"Wissensbasis nicht verfügbar: {exc}")
        retrieval.update(output=[{"document": c.document, "heading": c.heading, "score": c.score} for c in chunks])
    if not chunks:
        return tool_error("Nichts Passendes in der Wissensbasis gefunden.")
    graded = await grade_hits(query, chunks)
    if not graded.kept:
        titles = ", ".join(dict.fromkeys(f"„{c.document}“" for c in graded.discarded))
        return tool_error(
            f"Die Wissensbasis enthält dazu nichts Passendes (gefunden, aber unpassend: {titles}). "
            "Anders formulieren oder ohne diese Informationen planen und im Plan darauf hinweisen."
        )
    return [c.model_dump() for c in graded.kept]
