"""Relevanzprüfung der Suchtreffer (Corrective RAG): Ein kleines LLM bewertet, ob eine Fundstelle die Suchanfrage
tatsächlich beantwortet oder nur thematisch ähnlich ist.

Die Mindestähnlichkeit filtert nur nach Vektorabstand. Thematisch nahe, aber unpassende Abschnitte (z. B. Wildcampen
in DACH auf eine Frage zum Jedermannsrecht in Schweden) kommen darüber hinweg; siehe python -m app.evaluation.retrieval.

Ein Aufruf bewertet die Treffer mehrerer Suchanfragen. Der Recherche-Agent stellt seine Suchen parallel; mit einem
gemeinsamen `batch_key` sammelt `grade_hits` die gleichzeitig eintreffenden Bewertungen kurz und prüft sie zusammen.
So kostet eine Recherche eine Anfrage statt einer je Suche und bleibt unter dem Anfragelimit des Modells.
"""

import asyncio
import logging
from collections.abc import Hashable
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ToolOutput

from app.agent.models import build_model
from app.config import get_settings
from app.rag.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

MAX_CHARS_PER_HIT = 1000
MAX_OUTPUT_TOKENS = 1500
# Die Prüfung darf die Planung nicht ausbremsen: kein Warten auf Wiederholungen (z. B. nach HTTP 429 bei knappem
# Ratenlimit), sondern nach kurzer Zeit ungeprüft weiter. Eine Bewertung dauert normalerweise 1-3 s.
TIMEOUT_SECONDS = 15.0
# So lange sammelt grade_hits Bewertungen mit demselben batch_key. Parallele Suchen kommen nach Embedding und
# Datenbankabfrage meist innerhalb weniger Zehntelsekunden an; später eintreffende bilden einen neuen Aufruf.
BATCH_WINDOW_SECONDS = 1.0

INSTRUCTIONS = """\
Du prüfst Suchtreffer aus der Wissensbasis eines Outdoor-Tourenplaners (Wandern, Rad, Kanu, Hütten, Camping).
Du bekommst eine oder mehrere Suchanfragen, jede mit ihren Fundstellen. Entscheide für jede Fundstelle, ob sie
Informationen enthält, die zur Beantwortung IHRER Suchanfrage beitragen. Bewerte jede Fundstelle nur gegen die
Suchanfrage, unter der sie steht, nicht gegen die anderen.

Relevant: Die Fundstelle beantwortet die Anfrage ganz oder teilweise oder enthält direkt nutzbare Regeln, Tipps oder
Fakten zum gefragten Thema.
Nicht relevant: Die Fundstelle teilt nur das Oberthema, behandelt ein anderes Land, eine andere Region oder Aktivität
als gefragt, oder erwähnt den gesuchten Begriff nur nebenbei.

Gib für jede Fundstelle die Nummer ihrer Suchanfrage, ihre eigene Nummer und Deine Entscheidung aus.
"""


class Judgement(BaseModel):
    query: int = Field(description="Nummer der Suchanfrage")
    index: int = Field(description="Nummer der Fundstelle innerhalb ihrer Suchanfrage")
    relevant: bool


class RelevanceResult(BaseModel):
    judgements: list[Judgement]


@dataclass
class GradedHits:
    kept: list[RetrievedChunk]
    discarded: list[RetrievedChunk] = field(default_factory=list)
    failed: bool = False  # Prüfung nicht möglich, `kept` sind die ungeprüften Treffer


GradeRequest = tuple[str, list[RetrievedChunk]]


@lru_cache
def get_relevance_agent() -> Agent[None, RelevanceResult]:
    """In Tests: ``with get_relevance_agent().override(model=FunctionModel(...))``."""
    settings = get_settings()
    # Agenten mit demselben Modell teilen sich dessen Anfragelimit (app.agent.rate_limit.shared_budget)
    return Agent(
        build_model(settings.relevance_model, settings.llm_requests_per_minute) if settings.relevance_model else None,
        name="relevance-grader",  # Name des Agent-Spans im Tracing
        output_type=ToolOutput(RelevanceResult, name="bewertung", strict=True),
        instructions=INSTRUCTIONS,
        model_settings={"max_tokens": MAX_OUTPUT_TOKENS, "timeout": TIMEOUT_SECONDS, "max_retries": 0},
        defer_model_check=True,
    )


def _prompt(requests: list[GradeRequest]) -> str:
    """Suchanfragen ab 1 nummeriert, Fundstellen je Anfrage ab 0: [Anfrage.Fundstelle]."""
    sections = []
    for q, (query, chunks) in enumerate(requests, start=1):
        hits = "\n\n".join(
            f"[{q}.{i}] {c.document} – {c.heading}\n{c.text[:MAX_CHARS_PER_HIT]}" for i, c in enumerate(chunks)
        )
        sections.append(f"Suchanfrage {q}: {query}\n\nFundstellen zu Suchanfrage {q}:\n\n{hits}")
    return "\n\n---\n\n".join(sections)


async def grade_batch(requests: list[GradeRequest]) -> list[GradedHits]:
    """Bewertet die Treffer mehrerer Suchanfragen in einem Aufruf. Ohne Modell oder bei einem Fehler bleiben alle
    erhalten; fehlt eine einzelne Bewertung, bleibt der Treffer ebenfalls erhalten (Recall vor Präzision)."""
    graded_requests = [(i, r) for i, r in enumerate(requests) if r[1]]
    results = [GradedHits(kept=chunks) for _, chunks in requests]
    if not graded_requests or not get_settings().relevance_model:
        return results
    try:
        result = await get_relevance_agent().run(_prompt([r for _, r in graded_requests]))
    except Exception as exc:  # Bewertung ist eine Verbesserung, kein Muss: lieber ungefiltert als gar nichts
        logger.warning("Relevanzprüfung fehlgeschlagen, Treffer ungefiltert: %s", exc)
        return [GradedHits(kept=chunks, failed=bool(chunks)) for _, chunks in requests]
    rejected = {(j.query, j.index) for j in result.output.judgements if not j.relevant}
    for q, (position, (_, chunks)) in enumerate(graded_requests, start=1):
        results[position] = GradedHits(
            kept=[c for i, c in enumerate(chunks) if (q, i) not in rejected],
            discarded=[c for i, c in enumerate(chunks) if (q, i) in rejected],
        )
    return results


@dataclass
class _Batch:
    requests: list[GradeRequest] = field(default_factory=list)
    futures: list[asyncio.Future[GradedHits]] = field(default_factory=list)


_open_batches: dict[Hashable, _Batch] = {}
_flush_tasks: set[asyncio.Task[None]] = set()  # Referenzen, damit laufende Tasks nicht eingesammelt werden


async def _flush_after_window(key: Hashable, batch: _Batch) -> None:
    try:
        await asyncio.sleep(BATCH_WINDOW_SECONDS)
        if _open_batches.get(key) is batch:
            del _open_batches[key]
        results = await grade_batch(batch.requests)  # fängt Fehler des Modells selbst ab
    except asyncio.CancelledError:  # Abbruch (z. B. Server fährt herunter): Wartende nicht hängen lassen
        if _open_batches.get(key) is batch:
            del _open_batches[key]
        for future in batch.futures:
            future.cancel()
        raise
    for future, graded in zip(batch.futures, results, strict=True):
        if not future.done():
            future.set_result(graded)


async def grade_hits(query: str, chunks: list[RetrievedChunk], batch_key: Hashable | None = None) -> GradedHits:
    """Trennt relevante von unpassenden Fundstellen.

    Ohne `batch_key` sofort in einem eigenen Aufruf. Mit `batch_key` (z. B. je Chat-Durchlauf) werden Bewertungen,
    die innerhalb von BATCH_WINDOW_SECONDS nach der ersten eintreffen, gemeinsam in einem Aufruf geprüft.
    """
    if not chunks or not get_settings().relevance_model:
        return GradedHits(kept=chunks)
    if batch_key is None:
        return (await grade_batch([(query, chunks)]))[0]
    loop = asyncio.get_running_loop()
    future: asyncio.Future[GradedHits] = loop.create_future()
    batch = _open_batches.get(batch_key)
    if batch is None:
        batch = _open_batches[batch_key] = _Batch()
        task = loop.create_task(_flush_after_window(batch_key, batch))
        _flush_tasks.add(task)
        task.add_done_callback(_flush_tasks.discard)
    batch.requests.append((query, chunks))
    batch.futures.append(future)
    return await future
