from types import SimpleNamespace

from pydantic_ai import ModelMessage, ModelResponse, RunUsage, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agent.models import build_model
from app.agent.plan_schema import TransportMode
from app.agent.rate_limit import RateLimitedModel, shared_budget
from app.agent.research import research_agent
from app.agent.tools.research import research_knowledge
from app.db import SessionLocal
from app.rag.retrieve import RetrievedChunk

BRIEF = {
    "region_notes": ["Der Radweg folgt dem Deich."],
    "rules": [],
    "packing_tips": ["Windjacke"],
    "safety_notes": ["Gegenwind einplanen"],
}


def researcher(*queries: str) -> FunctionModel:
    """Sucht in der ersten Runde parallel nach `queries` und fasst danach zusammen."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            prompt = messages[0].parts[-1].content
            assert "Nordseeküsten-Radweg" in prompt and "cycling" in prompt
            return ModelResponse(parts=[ToolCallPart("search_knowledge", {"query": q}) for q in queries])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, BRIEF)])

    return FunctionModel(respond)


def context() -> SimpleNamespace:
    return SimpleNamespace(deps=SimpleNamespace(session_factory=SessionLocal), usage=RunUsage())


async def test_research_agent_summarizes_searches(monkeypatch) -> None:
    async def search(session, query, **kwargs):
        return [RetrievedChunk(document=f"Dokument zu {query}", heading="", text="…", score=0.6)]

    monkeypatch.setattr("app.agent.tools.knowledge.search_chunks", search)
    ctx = context()
    with research_agent.override(model=researcher("Nordseeküsten-Radweg", "Ausrüstung Radtour")):
        result = await research_knowledge(ctx, "Nordseeküsten-Radweg", TransportMode.cycling)

    assert result["packing_tips"] == ["Windjacke"] and result["safety_notes"] == ["Gegenwind einplanen"]
    assert result["queries"] == ["Nordseeküsten-Radweg", "Ausrüstung Radtour"]
    assert result["documents"] == ["Dokument zu Nordseeküsten-Radweg", "Dokument zu Ausrüstung Radtour"]
    # Die Anfragen des Recherche-Agenten zählen zur Nutzung des Planer-Laufs
    assert ctx.usage.requests == 2


async def test_research_without_hits_is_an_error(monkeypatch) -> None:
    async def nothing(session, query, **kwargs):
        return []

    monkeypatch.setattr("app.agent.tools.knowledge.search_chunks", nothing)
    with research_agent.override(model=researcher("Nordseeküsten-Radweg")):
        result = await research_knowledge(context(), "Nordseeküsten-Radweg", TransportMode.cycling)
    assert "nichts Passendes" in result["error"] and "Nordseeküsten-Radweg" in result["error"]


async def test_failed_research_is_an_error() -> None:
    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("Anbieter nicht erreichbar")

    with research_agent.override(model=FunctionModel(broken)):
        result = await research_knowledge(context(), "Harz", TransportMode.hiking)
    assert result["error"].startswith("Recherche nicht möglich")


def test_agents_with_same_model_share_budget() -> None:
    first, second = build_model("test", 5), build_model("test", 5)
    assert isinstance(first, RateLimitedModel) and isinstance(second, RateLimitedModel)
    assert first.budget is second.budget is shared_budget("test", 5)
    assert shared_budget("openai:gpt-5-mini", 5) is not first.budget
    # Geändertes Limit: neues Budget statt des alten
    assert shared_budget("test", 3).per_minute == 3

