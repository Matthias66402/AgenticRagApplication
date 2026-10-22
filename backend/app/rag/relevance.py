"""Relevanzprüfung der Suchtreffer (Corrective RAG): Ein kleines LLM bewertet, ob eine Fundstelle die Suchanfrage
tatsächlich beantwortet oder nur thematisch ähnlich ist.

Die Mindestähnlichkeit filtert nur nach Vektorabstand. Thematisch nahe, aber unpassende Abschnitte (z. B. Wildcampen
in DACH auf eine Frage zum Jedermannsrecht in Schweden) kommen darüber hinweg; siehe python -m app.evaluation.retrieval.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ToolOutput

from app.config import get_settings
from app.rag.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

MAX_CHARS_PER_HIT = 1000
MAX_OUTPUT_TOKENS = 1000
# Die Prüfung darf die Planung nicht ausbremsen: kein Warten auf Wiederholungen (z. B. nach HTTP 429 bei knappem
# Ratenlimit), sondern nach kurzer Zeit ungeprüft weiter. Eine Bewertung dauert normalerweise 1-2 s.
TIMEOUT_SECONDS = 10.0

INSTRUCTIONS = """\
Du prüfst Suchtreffer aus der Wissensbasis eines Outdoor-Tourenplaners (Wandern, Rad, Kanu, Hütten, Camping).
Entscheide für jede Fundstelle, ob sie Informationen enthält, die zur Beantwortung der Suchanfrage beitragen.

Relevant: Die Fundstelle beantwortet die Anfrage ganz oder teilweise oder enthält direkt nutzbare Regeln, Tipps oder
Fakten zum gefragten Thema.
Nicht relevant: Die Fundstelle teilt nur das Oberthema, behandelt ein anderes Land, eine andere Region oder Aktivität
als gefragt, oder erwähnt den gesuchten Begriff nur nebenbei.

Gib für jede Fundstelle ihre Nummer und Deine Entscheidung aus.
"""


class Judgement(BaseModel):
    index: int = Field(description="Nummer der Fundstelle")
    relevant: bool


class RelevanceResult(BaseModel):
    judgements: list[Judgement]


@dataclass
class GradedHits:
    kept: list[RetrievedChunk]
    discarded: list[RetrievedChunk] = field(default_factory=list)
    failed: bool = False  # Prüfung nicht möglich, `kept` sind die ungeprüften Treffer


@lru_cache
def get_relevance_agent() -> Agent[None, RelevanceResult]:
    """In Tests: ``with get_relevance_agent().override(model=FunctionModel(...))``."""
    return Agent(
        get_settings().relevance_model or None,
        name="relevance-grader",  # Name des Agent-Spans im Tracing
        output_type=ToolOutput(RelevanceResult, name="bewertung", strict=True),
        instructions=INSTRUCTIONS,
        model_settings={"max_tokens": MAX_OUTPUT_TOKENS, "timeout": TIMEOUT_SECONDS, "max_retries": 0},
        defer_model_check=True,
    )


def _prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    hits = "\n\n".join(
        f"[{i}] {c.document} – {c.heading}\n{c.text[:MAX_CHARS_PER_HIT]}" for i, c in enumerate(chunks)
    )
    return f"Suchanfrage: {query}\n\nFundstellen:\n\n{hits}"


async def grade_hits(query: str, chunks: list[RetrievedChunk]) -> GradedHits:
    """Trennt relevante von unpassenden Fundstellen. Ohne Modell oder bei einem Fehler bleiben alle erhalten."""
    if not chunks or not get_settings().relevance_model:
        return GradedHits(kept=chunks)
    try:
        result = await get_relevance_agent().run(_prompt(query, chunks))
    except Exception as exc:  # Bewertung ist eine Verbesserung, kein Muss: lieber ungefiltert als gar nichts
        logger.warning("Relevanzprüfung fehlgeschlagen, Treffer ungefiltert: %s", exc)
        return GradedHits(kept=chunks, failed=True)
    rejected = {j.index for j in result.output.judgements if not j.relevant}
    # Fehlt eine Bewertung, bleibt der Treffer erhalten (Recall vor Präzision)
    return GradedHits(
        kept=[c for i, c in enumerate(chunks) if i not in rejected],
        discarded=[c for i, c in enumerate(chunks) if i in rejected],
    )
