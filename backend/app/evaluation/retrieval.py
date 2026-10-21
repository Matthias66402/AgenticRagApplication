"""Misst die Retrieval-Qualität der Wissensbasis anhand eines festen Testsatzes.

Aufruf: ``python -m app.evaluation.retrieval [--output bericht.json]``

Für jede Testfrage werden die besten RANK_DEPTH Abschnitte ohne Mindestähnlichkeit abgerufen. Daraus ergeben sich:
- Hit@1 / Hit@k: Anteil der Fragen, bei denen ein relevanter Abschnitt auf Platz 1 bzw. unter den ersten k liegt
- MRR: mittlerer Kehrwert des Rangs des ersten relevanten Abschnitts
- Recall wie im Agenten: Treffer unter den ersten k *und* über der Mindestähnlichkeit (das, was search_knowledge liefert)
- Fehlkontext: Anteil der nicht beantwortbaren Fragen, für die trotzdem Abschnitte über der Mindestähnlichkeit kommen
- Schwellen-Sweep: Recall und Fehlkontext für verschiedene Mindestähnlichkeiten
- mit --grade zusätzlich Recall und Fehlkontext nach der Relevanzprüfung (app/rag/relevance.py), wie sie
  search_knowledge anwendet; kostet je Frage einen Aufruf von RELEVANCE_MODEL
"""

import argparse
import asyncio
import logging
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.config import get_settings
from app.db import SessionLocal, engine
from app.models import Chunk, Document
from app.rag.relevance import grade_hits
from app.rag.retrieve import MIN_SCORE, TOP_K, RetrievedChunk, search_chunks

CASES_FILE = Path(__file__).parent / "retrieval_cases.json"
RANK_DEPTH = 10
# Bei knappem Ratenlimit des LLM-Anbieters: fehlgeschlagene Bewertungen nach einer Pause wiederholen
GRADE_ATTEMPTS = 6
GRADE_RETRY_SECONDS = 20
THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(9)]  # 0,20 … 0,60


@dataclass
class CaseResult:
    id: str
    category: str
    query: str
    answerable: bool
    rank: int | None  # Rang des ersten relevanten Abschnitts (1-basiert), None = nicht unter den ersten RANK_DEPTH
    document_rank: int | None  # wie rank, aber nur das Dokument muss stimmen
    relevant_score: float | None  # Ähnlichkeit des ersten relevanten Abschnitts
    top_score: float
    results: list[dict[str, Any]] = field(default_factory=list)
    # Nach der Relevanzprüfung (nur mit --grade): kommt ein relevanter bzw. überhaupt ein Abschnitt beim Agenten an?
    graded_delivered: bool | None = None
    graded_context: bool | None = None


def is_relevant(chunk: RetrievedChunk, relevant: list[dict[str, str]]) -> bool:
    """Treffer, wenn Dokument stimmt und der Überschriftenpfad mit dem erwarteten Abschnitt endet."""
    for expected in relevant:
        if chunk.document != expected["document"]:
            continue
        section = expected.get("section")
        if not section or chunk.heading == section or chunk.heading.endswith(f" > {section}"):
            return True
    return False


def first_rank(flags: list[bool]) -> int | None:
    return next((i + 1 for i, flag in enumerate(flags) if flag), None)


def evaluate_case(case: dict[str, Any], chunks: list[RetrievedChunk], kept: set[int] | None = None) -> CaseResult:
    """`kept`: Positionen der Abschnitte, die nach der Relevanzprüfung beim Agenten ankommen (None = nicht geprüft)."""
    relevant = case["relevant"]
    hits = [is_relevant(c, relevant) for c in chunks]
    documents = {r["document"] for r in relevant}
    rank = first_rank(hits)
    return CaseResult(
        id=case["id"],
        category=case["category"],
        query=case["query"],
        answerable=bool(relevant),
        rank=rank,
        document_rank=first_rank([c.document in documents for c in chunks]),
        relevant_score=chunks[rank - 1].score if rank else None,
        top_score=chunks[0].score if chunks else 0.0,
        results=[
            {"document": c.document, "heading": c.heading, "score": c.score, "relevant": hit}
            | ({"kept": i in kept} if kept is not None else {})
            for i, (c, hit) in enumerate(zip(chunks, hits, strict=True))
        ],
        graded_delivered=None if kept is None else any(hits[i] for i in kept),
        graded_context=None if kept is None else bool(kept),
    )


def delivered(result: CaseResult, threshold: float, top_k: int = TOP_K) -> bool:
    """Liefert search_knowledge bei dieser Mindestähnlichkeit einen relevanten Abschnitt?"""
    return result.rank is not None and result.rank <= top_k and (result.relevant_score or 0) >= threshold


def false_context(result: CaseResult, threshold: float) -> bool:
    """Kommt bei einer nicht beantwortbaren Frage trotzdem Kontext beim LLM an?"""
    return result.top_score >= threshold


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    answerable = [r for r in results if r.answerable]
    negatives = [r for r in results if not r.answerable]

    def share(items: list[CaseResult], predicate: Any) -> float:
        return round(sum(1 for r in items if predicate(r)) / len(items), 3) if items else 0.0

    by_category: dict[str, dict[str, Any]] = {}
    for category in dict.fromkeys(r.category for r in answerable):
        items = [r for r in answerable if r.category == category]
        by_category[category] = {
            "cases": len(items),
            "hit_at_1": share(items, lambda r: r.rank == 1),
            "hit_at_k": share(items, lambda r: r.rank is not None and r.rank <= TOP_K),
            "recall_deployed": share(items, lambda r: delivered(r, MIN_SCORE)),
        }

    graded = {}
    if any(r.graded_context is not None for r in results):
        graded = {
            "recall_graded": share(answerable, lambda r: bool(r.graded_delivered)),
            "false_context_graded": share(negatives, lambda r: bool(r.graded_context)),
        }

    return graded | {
        "answerable_cases": len(answerable),
        "negative_cases": len(negatives),
        "hit_at_1": share(answerable, lambda r: r.rank == 1),
        "hit_at_k": share(answerable, lambda r: r.rank is not None and r.rank <= TOP_K),
        "document_hit_at_k": share(answerable, lambda r: r.document_rank is not None and r.document_rank <= TOP_K),
        "mrr": round(sum(1 / r.rank for r in answerable if r.rank) / len(answerable), 3) if answerable else 0.0,
        "recall_deployed": share(answerable, lambda r: delivered(r, MIN_SCORE)),
        "false_context_rate": share(negatives, lambda r: false_context(r, MIN_SCORE)),
        "by_category": by_category,
        "threshold_sweep": [
            {
                "threshold": t,
                "recall": share(answerable, lambda r, t=t: delivered(r, t)),
                "false_context_rate": share(negatives, lambda r, t=t: false_context(r, t)),
            }
            for t in THRESHOLDS
        ],
    }


async def grade_with_retry(query: str, hits: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], int]:
    """Bewertet wie search_knowledge, wiederholt aber Fehlschläge, statt ungeprüft weiterzugeben.

    Gibt die behaltenen Treffer und die Zahl der Fehlschläge zurück; scheitern alle Versuche, gilt alles als behalten.
    """
    failures = 0
    for attempt in range(GRADE_ATTEMPTS):
        graded = await grade_hits(query, hits)
        if not graded.failed:
            return graded.kept, failures
        failures += 1
        if attempt < GRADE_ATTEMPTS - 1:
            await asyncio.sleep(GRADE_RETRY_SECONDS)
    return hits, failures


async def run(grade: bool = False) -> dict[str, Any]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    settings = get_settings()
    async with SessionLocal() as session:
        documents = await session.scalar(select(func.count(Document.id)))
        chunks = await session.scalar(select(func.count(Chunk.id)))
        results = []
        grade_failures = 0
        ungraded = 0
        for number, case in enumerate(cases, start=1):
            retrieved = await search_chunks(session, case["query"], top_k=RANK_DEPTH, min_score=0.0)
            kept = None
            if grade:
                # Was search_knowledge weitergibt: Top-k über der Mindestähnlichkeit, danach Relevanzprüfung
                delivered_hits = [c for c in retrieved[:TOP_K] if c.score >= MIN_SCORE]
                kept_hits, failures = await grade_with_retry(case["query"], delivered_hits)
                grade_failures += failures
                ungraded += failures == GRADE_ATTEMPTS
                kept = {i for i, c in enumerate(retrieved) if any(c is k for k in kept_hits)}
                print(f"  {number}/{len(cases)} {case['id']}" + (f" ({failures} Fehlversuche)" if failures else ""),
                      file=sys.stderr, flush=True)  # fmt: skip
            results.append(evaluate_case(case, retrieved, kept))
    return {
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": {
            "embedding_model": settings.embedding_model,
            "relevance_model": settings.relevance_model if grade else None,
            "grade_failures": grade_failures if grade else None,
            "ungraded_cases": ungraded if grade else None,
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "rank_depth": RANK_DEPTH,
            "documents": documents,
            "chunks": chunks,
        },
        "summary": summarize(results),
        "cases": [asdict(r) for r in results],
    }


def print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    print(f"{s['answerable_cases']} beantwortbare, {s['negative_cases']} nicht beantwortbare Fragen")
    print(f"Hit@1 {s['hit_at_1']:.0%} · Hit@{TOP_K} {s['hit_at_k']:.0%} · MRR {s['mrr']:.2f} · "
          f"Dokument-Hit@{TOP_K} {s['document_hit_at_k']:.0%}")  # fmt: skip
    print(f"Recall wie im Agenten (Top {TOP_K}, ≥ {MIN_SCORE}): {s['recall_deployed']:.0%} · "
          f"Fehlkontext bei nicht beantwortbaren Fragen: {s['false_context_rate']:.0%}")  # fmt: skip
    if "recall_graded" in s:
        c = report["config"]
        print(f"Nach Relevanzprüfung ({c['relevance_model']}): Recall {s['recall_graded']:.0%} · "
              f"Fehlkontext {s['false_context_graded']:.0%} · {c['grade_failures']} Fehlversuche, "
              f"{c['ungraded_cases']} Fragen ungeprüft")  # fmt: skip
        for case in report["cases"]:
            if case["answerable"] and not case["graded_delivered"]:
                print(f"  verworfen, obwohl passend: {case['id']}")
            if not case["answerable"] and case["graded_context"]:
                kept = [f"{r['document']} > {r['heading'].split(' > ')[-1]}" for r in case["results"] if r.get("kept")]
                print(f"  Fehlkontext trotz Prüfung: {case['id']} → {kept}")
    print("\nFragen ohne relevanten Treffer unter den ersten Plätzen:")
    for case in report["cases"]:
        if case["answerable"] and not (case["rank"] and case["rank"] <= TOP_K):
            top = case["results"][0]
            print(f"  {case['id']:22} Rang {case['rank'] or '–':>2}  Platz 1: {top['document']} > {top['heading']} ({top['score']})")
    print("\nSchwellen-Sweep (Mindestähnlichkeit → Recall / Fehlkontext):")
    for row in s["threshold_sweep"]:
        print(f"  {row['threshold']:.2f} → {row['recall']:.0%} / {row['false_context_rate']:.0%}")


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="Bericht zusätzlich als JSON speichern")
    parser.add_argument("--grade", action="store_true", help="Auch die Relevanzprüfung messen (LLM-Aufrufe)")
    args = parser.parse_args(argv)
    # Fehlversuche zählt die Auswertung selbst; die Warnung je Fehlschlag würde die Ausgabe überfluten
    logging.getLogger("app.rag.relevance").setLevel(logging.ERROR)
    try:
        report = await run(grade=args.grade)
    finally:
        await engine.dispose()
    print_report(report)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBericht gespeichert: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
