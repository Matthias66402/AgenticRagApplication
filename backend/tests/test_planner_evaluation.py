import json
from typing import Any

import httpx
from pydantic_ai import (
    ModelRequest, ModelResponse, RequestUsage, RetryFeedbackPart, RetryInfo, ToolCallPart, ToolReturnPart,
)  # fmt: skip

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AdventurePlan, ClarificationQuestion, TextAnswer
from app.config import get_settings
from app.agent.planner import planner_agent
from app.evaluation.planner import (
    CASES_FILE, OUTPUT_NAMES, CaseRun, check_output, covered_seconds, measure, measure_usage, run_case, summarize,
)  # fmt: skip
from tests.test_agent import RESEARCH, make_plan, scripted_model


def deps(routes: dict[str, Any] | None = None) -> AgentDeps:
    return AgentDeps(settings=get_settings(), http=None, session_factory=None, user_name="Test", routes=routes or {})  # type: ignore[arg-type]


def call(tool: str, model: str = "gpt-5-mini") -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool, {})], usage=RequestUsage(input_tokens=100, output_tokens=10), model_name=model
    )


def test_plan_checks() -> None:
    case = {"id": "t", "expected_output": "final_plan", "transport": "hiking", "duration_days": 2}
    data = make_plan(days=2, accommodation="hut") | {"route_ids": ["r-1"], "packing_tips": ["Regenjacke"]}
    data["stages"][0]["accommodation"] = {"name": "Meilerhütte", "type": "hut", "lat": 47.4, "lon": 11.1}
    plan = AdventurePlan.model_validate(data)
    found = ModelRequest(parts=[ToolReturnPart("find_accommodation", {"results": [{"name": "Meilerhütte"}]})])
    research = ModelRequest(parts=[ToolReturnPart("research_knowledge", {"queries": ["Harz", "Ausrüstung"]})])
    messages = [call("research_knowledge"), found, research]

    checks = check_output(case, plan, messages, deps({"r-1": []}))

    assert checks.pop("packing_and_safety") is False  # keine safety_notes
    assert all(checks.values()), checks
    # Nur eine Suche des Recherche-Agenten statt der beiden Pflichtsuchen
    single = ModelRequest(parts=[ToolReturnPart("research_knowledge", {"queries": ["Harz"]})])
    assert check_output(case, plan, [found, single], deps({"r-1": []}))["knowledge_searches"] is False


def test_other_outputs() -> None:
    assert check_output({"expected_output": "ask_user"}, ClarificationQuestion(question="Wohin?"), [], deps()) == {
        "output_type": True
    }
    answer = TextAnswer(message="Die schnellste Verbindung …")
    checks = check_output({"expected_output": "answer_user"}, answer, [call("geocode")], deps())
    assert checks == {"output_type": True, "train_query": False}


def test_measure_and_summarize() -> None:
    run = CaseRun(id="t", run=1)
    # Retry eines Tool-Aufrufs und Retry der Ausgabe (z. B. abgelehnter Plan) zählen beide
    tool_retry = ToolReturnPart(
        "geocode", "Ort nicht gefunden", "c-1", outcome="retried", retry=RetryInfo(cause="model_retry")
    )
    output_retry = RetryFeedbackPart("Etappe zu lang", retry=RetryInfo(cause="model_retry"))
    measure(run, [call("geocode"), ModelRequest(parts=[tool_retry, output_retry]), call("geocode")])
    assert run.tool_calls == {"geocode": 2} and run.retries == 2
    assert [c["tool"] for c in run.calls] == ["geocode", "geocode"]
    assert run.feedback and "Etappe zu lang" in run.feedback[-1]
    # Aufwand aus den aufgezeichneten Antworten aller Agenten, z. B. auch des Recherche-Agenten
    measure_usage(run, [call("geocode"), call("search_knowledge", "gpt-4.1-mini"), call("geocode")])
    assert (run.requests, run.input_tokens, run.output_tokens) == (3, 300, 30)
    assert run.requests_by_model == {"gpt-5-mini": 2, "gpt-4.1-mini": 1}

    run.checks = {"output_type": True}
    failed = CaseRun(id="t", run=2, error="Abbruch")
    summary = summarize([run, failed])
    assert (summary["runs"], summary["passed"], summary["errors"], summary["mean_requests"]) == (2, 1, 1, 1.5)
    assert summary["mean_requests_by_model"] == {"gpt-4.1-mini": 0.5, "gpt-5-mini": 1.0}


async def test_run_case_measures_planner_and_research() -> None:
    """Ganzer Fall über den Graphen mit geskripteten Modellen: Aufzeichnung umfasst auch den Recherche-Agenten."""
    case = {"id": "t", "message": "Wanderung ab Garmisch morgen", "expected_output": "final_plan", "transport": "hiking"}
    model = scripted_model(("research_knowledge", RESEARCH), ("final_plan", make_plan()))
    async with httpx.AsyncClient() as http:
        with planner_agent.override(model=model):
            run = await run_case(case, 1, http)
    assert run.error is None and run.output == "final_plan"
    assert run.tool_calls == {"research_knowledge": 1, "final_plan": 1}
    # 2 Planer-Aufrufe (FunctionModel) + Intake (1) und Recherche-Agent (2: Suche + Zusammenfassung), beide TestModel
    assert run.requests_by_model == {"function::stream": 2, "test": 3}
    assert run.requests == 5


def test_parallel_tool_time_counts_once() -> None:
    assert covered_seconds([(0, 4), (1, 3), (2, 5), (8, 9)]) == 6
    assert covered_seconds([]) == 0


def test_case_file_is_consistent() -> None:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(c["message"].strip() and c["expected_output"] in OUTPUT_NAMES.values() for c in cases)
