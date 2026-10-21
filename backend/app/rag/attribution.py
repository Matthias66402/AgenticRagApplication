"""Quellenangaben für Wissensbasis-Inhalte, die der Agent für einen Plan abgerufen hat.

Wird deterministisch aus dem Agent-Verlauf bestimmt, nicht aus den Freitext-`sources` des LLM: Jedes
Dokument mit Lizenz-Kopfzeile (z. B. Wikivoyage, CC BY-SA), das `search_knowledge` geliefert hat, wird genannt.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.rag.wikivoyage import parse_attribution

KNOWLEDGE_TOOL = "search_knowledge"


def retrieved_documents(history: list[dict[str, Any]]) -> list[str]:
    """Titel der Dokumente aus allen search_knowledge-Ergebnissen, in Reihenfolge des ersten Auftretens."""
    titles: dict[str, None] = {}
    for message in history:
        for part in message.get("parts", []):
            if part.get("part_kind") != "tool-return" or part.get("tool_name") != KNOWLEDGE_TOOL:
                continue
            content = part.get("content")
            for hit in content if isinstance(content, list) else []:
                if isinstance(hit, dict) and hit.get("document"):
                    titles.setdefault(hit["document"], None)
    return list(titles)


async def knowledge_attributions(session: AsyncSession, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    titles = retrieved_documents(history)
    if not titles:
        return []
    # Die Kopfzeile mit Quelle und Lizenz steht im ersten Abschnitt des Dokuments
    rows = await session.execute(
        select(Document.title, Chunk.text).join(Chunk).where(Document.title.in_(titles), Chunk.position == 0)
    )
    by_title = {title: parse_attribution(text) for title, text in rows}
    return [attribution.model_dump() for title in titles if (attribution := by_title.get(title))]


def merge_attributions(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Vereinigt Quellenangaben (eindeutig nach URL); frühere bleiben erhalten, auch wenn das Dokument gelöscht wurde."""
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for attribution in group:
            merged.setdefault(attribution["url"], attribution)
    return list(merged.values())
