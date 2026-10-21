from types import SimpleNamespace

from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agent.tools.knowledge import search_knowledge
from app.db import SessionLocal
from app.evaluation.retrieval import evaluate_case, summarize
from app.rag.ingest import ingest_document
from app.rag.relevance import get_relevance_agent, grade_hits
from app.rag.retrieve import RetrievedChunk


def judge(relevant: dict[int, bool]) -> FunctionModel:
    """Bewertet die Fundstellen mit den Nummern aus `relevant`; der Prompt muss jede davon enthalten."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = messages[-1].parts[-1].content
        assert all(f"[{i}]" in prompt for i in relevant)
        judgements = [{"index": i, "relevant": r} for i, r in relevant.items()]
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
