from typing import Any

from app.db import SessionLocal
from app.rag.attribution import knowledge_attributions, merge_attributions, retrieved_documents
from app.rag.ingest import ingest_document

WIKIVOYAGE_DOC = """# Testradweg

> Quelle: Wikivoyage, Artikel „Testradweg“ (https://de.wikivoyage.org/wiki/Testradweg), Version vom 2026-10-03. Autoren: siehe https://de.wikivoyage.org/wiki/Testradweg?action=history. Lizenz: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/deed.de).

Einleitung zum Radweg.

## Sicherheit

Helm tragen.
""".encode()

OWN_DOC = "# Hüttenknigge\n\nHüttenschlafsack mitnehmen.\n".encode()


def knowledge_return(*documents: str, tool_name: str = "search_knowledge") -> dict[str, Any]:
    hits = [{"document": d, "heading": "", "text": "…", "score": 0.5} for d in documents]
    return {"kind": "request", "parts": [{"part_kind": "tool-return", "tool_name": tool_name, "content": hits}]}


HISTORY = [
    {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Radtour bitte"}]},
    knowledge_return("Testradweg", "Hüttenknigge"),
    knowledge_return("Nicht gesucht", tool_name="geocode"),
    {"kind": "request", "parts": [{"part_kind": "tool-return", "tool_name": "search_knowledge", "content": {"error": "x"}}]},
    knowledge_return("Hüttenknigge", "Testradweg"),
]


def test_retrieved_documents_only_from_knowledge_search() -> None:
    assert retrieved_documents(HISTORY) == ["Testradweg", "Hüttenknigge"]
    assert retrieved_documents([]) == []


async def test_knowledge_attributions_only_for_licensed_documents() -> None:
    async with SessionLocal() as session:
        await ingest_document(session, "wikivoyage-testradweg.md", WIKIVOYAGE_DOC)
        await ingest_document(session, "huettenknigge.md", OWN_DOC)
        attributions = await knowledge_attributions(session, HISTORY)

    assert attributions == [
        {
            "publisher": "Wikivoyage",
            "title": "Testradweg",
            "url": "https://de.wikivoyage.org/wiki/Testradweg",
            "revision_date": "2026-10-03",
            "history_url": "https://de.wikivoyage.org/wiki/Testradweg?action=history",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/deed.de",
        }
    ]


def test_merge_keeps_previous_attributions() -> None:
    old = {"url": "https://a", "title": "A"}
    new = {"url": "https://b", "title": "B"}
    assert merge_attributions([old], [new, old]) == [old, new]


def test_retrieved_documents_from_research_agent() -> None:
    research = {"kind": "request", "parts": [
        {"part_kind": "tool-return", "tool_name": "research_knowledge", "content": {"documents": ["Testradweg"]}},
        {"part_kind": "tool-return", "tool_name": "research_knowledge", "content": {"error": "nichts Passendes"}},
    ]}  # fmt: skip
    assert retrieved_documents([research, knowledge_return("Hüttenknigge")]) == ["Testradweg", "Hüttenknigge"]
