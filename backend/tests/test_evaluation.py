import json

from app.evaluation.retrieval import CASES_FILE, delivered, evaluate_case, false_context, is_relevant, summarize
from app.rag.retrieve import RetrievedChunk

RELEVANT = [{"document": "Hüttenknigge", "section": "Reservierung"}]


def chunk(document: str, heading: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(document=document, heading=heading, text="…", score=score)


def test_relevance_matches_last_heading_segment() -> None:
    assert is_relevant(chunk("Hüttenknigge", "Hüttenknigge > Reservierung", 0.5), RELEVANT)
    assert is_relevant(chunk("Hüttenknigge", "Reservierung", 0.5), RELEVANT)
    assert not is_relevant(chunk("Hüttenknigge", "Hüttenknigge > Online-Reservierung", 0.5), RELEVANT)
    assert not is_relevant(chunk("Andere", "Andere > Reservierung", 0.5), RELEVANT)


def test_metrics_for_answerable_and_unanswerable_cases() -> None:
    answerable = {"id": "a", "category": "Hütten", "query": "q", "relevant": RELEVANT}
    first = evaluate_case(answerable, [chunk("Hüttenknigge", "Hüttenknigge > Reservierung", 0.6)])
    second = evaluate_case(
        answerable,
        [chunk("Hüttenknigge", "Hüttenknigge", 0.7), chunk("Hüttenknigge", "Hüttenknigge > Reservierung", 0.3)],
    )
    missed = evaluate_case(answerable, [chunk("Andere", "Andere", 0.4)])
    negative = evaluate_case({"id": "n", "category": "Nicht beantwortbar", "query": "q", "relevant": []},
                             [chunk("Andere", "Andere", 0.55)])  # fmt: skip

    assert (first.rank, second.rank, second.document_rank, missed.rank) == (1, 2, 1, None)
    assert delivered(second, 0.25) and not delivered(second, 0.35)
    assert false_context(negative, 0.5) and not false_context(negative, 0.6)

    summary = summarize([first, second, missed, negative])
    assert summary["answerable_cases"] == 3 and summary["negative_cases"] == 1
    assert summary["hit_at_1"] == 0.333
    assert summary["hit_at_k"] == 0.667
    assert summary["mrr"] == 0.5  # (1 + 1/2 + 0) / 3
    assert summary["false_context_rate"] == 1.0
    sweep = {row["threshold"]: row for row in summary["threshold_sweep"]}
    assert sweep[0.6]["recall"] == 0.333 and sweep[0.6]["false_context_rate"] == 0.0


def test_case_file_is_consistent() -> None:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(c["query"].strip() and "category" in c for c in cases)
    assert all(r["document"] and r["section"] for c in cases for r in c["relevant"])
