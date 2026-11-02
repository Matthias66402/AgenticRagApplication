import asyncio
from types import SimpleNamespace

from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agent.tools.knowledge import search_knowledge
from app.db import SessionLocal
from app.evaluation.retrieval import batch_groups, evaluate_case, summarize
from app.rag.ingest import ingest_document
from app.rag.relevance import get_relevance_agent, grade_batch, grade_hits
from app.rag.retrieve import RetrievedChunk


def judge(relevant: dict[int, bool] | dict[tuple[int, int], bool], calls: list[str] | None = None) -> FunctionModel:
    """Bewertet die Fundstellen aus `relevant` (Nummer in Suchanfrage 1 oder (Suchanfrage, Nummer)); der Prompt
    muss jede davon enthalten. `calls` sammelt die Prompts, um die Zahl der Aufrufe zu prüfen."""
    keys = {(k if isinstance(k, tuple) else (1, k)): r for k, r in relevant.items()}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = messages[-1].parts[-1].content
        if calls is not None:
            calls.append(prompt)
        assert all(f"[{q}.{i}]" in prompt for q, i in keys)
        judgements = [{"query": q, "index": i, "relevant": r} for (q, i), r in keys.items()]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"judgements": judgements})])

    return FunctionModel(respond)


def hit(document: str, score: float = 0.6) -> RetrievedChunk:
    return RetrievedChunk(document=document, heading=document, text="…", score=score)


async def test_grade_hits_drops_irrelevant() -> None:
    hits = [hit("Wildcampen DACH"), hit("Hüttenknigge"), hit("Kanu")]
    with get_relevance_agent().override(model=judge({0: False, 1: True})):
        graded = await grade_hits("Jedermannsrecht Schweden", hits)
    # Nr. 2 ohne Bewertung bleibt erhalten
    assert [c.document for c in graded.kept] == ["Hüttenknigge", "Kanu"]
    assert [c.document for c in graded.discarded] == ["Wildcampen DACH"]


async def test_grade_hits_keeps_everything_on_failure() -> None:
    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("Anbieter nicht erreichbar")

    hits = [hit("A"), hit("B")]
    with get_relevance_agent().override(model=FunctionModel(broken)):
        graded = await grade_hits("Frage", hits)
    assert graded.kept == hits and graded.discarded == [] and graded.failed


async def test_grade_batch_keeps_judgements_per_query() -> None:
    swedish, huts = [hit("Wildcampen DACH"), hit("Jedermannsrecht")], [hit("Hüttenknigge"), hit("Kanu")]
    model = judge({(1, 0): False, (1, 1): True, (2, 0): True, (2, 1): False})
    with get_relevance_agent().override(model=model):
        first, empty, second = await grade_batch([("Jedermannsrecht Schweden", swedish), ("leer", []), ("Hütten", huts)])
    assert [c.document for c in first.kept] == ["Jedermannsrecht"]
    assert empty.kept == [] and not empty.failed  # ohne Treffer nicht im Prompt
    # Die leere Anfrage zählt nicht mit: "Hütten" ist Suchanfrage 2 im Prompt
    assert [c.document for c in second.kept] == ["Hüttenknigge"] and [c.document for c in second.discarded] == ["Kanu"]


async def test_parallel_searches_share_one_grading_call() -> None:
    """Gleichzeitige Bewertungen mit demselben batch_key gehen in einem Aufruf an das Modell, andere Schlüssel nicht."""
    calls: list[str] = []
    model = judge({(1, 0): False, (2, 0): True}, calls)
    with get_relevance_agent().override(model=model):
        a, b = await asyncio.gather(
            grade_hits("Wildcampen Norwegen", [hit("Wildcampen DACH")], batch_key="lauf-1"),
            grade_hits("Hüttenschlafsack", [hit("Hüttenknigge")], batch_key="lauf-1"),
        )
    assert len(calls) == 1 and "Suchanfrage 1: Wildcampen Norwegen" in calls[0] and "Suchanfrage 2: Hüttenschlafsack" in calls[0]
    assert a.kept == [] and [c.document for c in b.kept] == ["Hüttenknigge"]

    calls.clear()
    with get_relevance_agent().override(model=judge({0: True}, calls)):
        await asyncio.gather(
            grade_hits("A", [hit("A")], batch_key="lauf-1"), grade_hits("B", [hit("B")], batch_key="lauf-2")
        )
    assert len(calls) == 2


async def test_failed_batch_keeps_all_hits() -> None:
    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("Anbieter nicht erreichbar")

    with get_relevance_agent().override(model=FunctionModel(broken)):
        a, b = await asyncio.gather(
            grade_hits("A", [hit("A")], batch_key="lauf"), grade_hits("B", [hit("B")], batch_key="lauf")
        )
    assert a.failed and b.failed and [c.document for c in a.kept + b.kept] == ["A", "B"]


async def test_search_knowledge_reports_when_nothing_relevant(monkeypatch) -> None:
    async with SessionLocal() as session:
        await ingest_document(session, "huetten.md", "# Hüttenknigge\n\nHüttenschlafsack mitnehmen.\n".encode())

    async def everything(session, query, **kwargs):  # Testembeddings haben keine sinnvollen Ähnlichkeiten
        return [hit("Hüttenknigge")]

    monkeypatch.setattr("app.agent.tools.knowledge.search_chunks", everything)
    ctx = SimpleNamespace(deps=SimpleNamespace(session_factory=SessionLocal))
    with get_relevance_agent().override(model=judge({0: False})):
        result = await search_knowledge(ctx, "Lawinenlagebericht lesen")
    assert "nichts Passendes" in result["error"]
    assert "„Hüttenknigge“" in result["error"]

    with get_relevance_agent().override(model=judge({0: True})):
        result = await search_knowledge(ctx, "Hüttenschlafsack")
    assert [r["document"] for r in result] == ["Hüttenknigge"]


def test_evaluation_batches_mix_categories() -> None:
    groups = batch_groups(10, 4)
    assert groups == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]  # reihum, höchstens 4 je Gruppe, jede Frage genau einmal
    assert batch_groups(3, 1) == [[0], [1], [2]]


def test_evaluation_metrics_after_grading() -> None:
    relevant = [{"document": "A", "section": "A"}]
    answerable = {"id": "a", "category": "x", "query": "q", "relevant": relevant}
    negative = {"id": "n", "category": "y", "query": "q", "relevant": []}
    kept_relevant = evaluate_case(answerable, [hit("A"), hit("B")], kept={0})
    dropped_relevant = evaluate_case(answerable, [hit("A")], kept=set())
    negative_filtered = evaluate_case(negative, [hit("B")], kept=set())

    summary = summarize([kept_relevant, dropped_relevant, negative_filtered])
    assert summary["recall_graded"] == 0.5
    assert summary["false_context_graded"] == 0.0
    assert kept_relevant.results[1]["kept"] is False
    assert "recall_graded" not in summarize([evaluate_case(answerable, [hit("A")])])
