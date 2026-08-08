"""Stage 3 — CrewAI Analyst crew: wiring, prompt loading, offline tool runs.

All tests are offline: they build and inspect the crew, drive the deterministic
tools directly (with the network blocked), and exercise the run path with the
LLM factory and ``kickoff`` mocked — never a real OpenAI call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retail_clickstream_ai.crews import prompts as prompt_loader
from retail_clickstream_ai.crews.analyst import models as m
from retail_clickstream_ai.crews.analyst.crew import build_analyst_crew, run_analyst_crew
from retail_clickstream_ai.crews.analyst.tools import build_analyst_tools
from retail_clickstream_ai.crews.context import AnalystRunContext
from retail_clickstream_ai.pipeline.analyst_pipeline import prepare_run
from retail_clickstream_ai.validation.contract import DatasetContract

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "clickstream_sample.csv"

_ROLES = ("source_quality", "data_engineer", "eda_business")
_EXPECTED_MODEL = {
    "source_quality": m.SourceQualityReview,
    "data_engineer": m.DataEngineeringHandoff,
    "eda_business": m.AnalystCrewHandoff,
}


@pytest.fixture
def ctx(monkeypatch, tmp_path) -> AnalystRunContext:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    context = AnalystRunContext.build(
        FIXTURE, run_id="test", pin_hash=False, require_all_months=True
    )
    context.run_dir.mkdir(parents=True, exist_ok=True)
    prepare_run(FIXTURE, DatasetContract.build(), context.run_dir, require_all_months=True)
    return context


# --- prompt loading -------------------------------------------------------- #
def test_all_analyst_prompt_files_parse() -> None:
    assert prompt_loader.load_shared_rules().strip()
    for path in prompt_loader.ANALYST_PROMPT_FILES:
        p = prompt_loader.load_agent_prompt(path)
        assert p.role and p.goal and p.backstory
        assert p.task_prompt and p.expected_output


# --- crew wiring ----------------------------------------------------------- #
def test_crew_has_three_distinct_agents_sequential(ctx) -> None:
    bundle = build_analyst_crew(ctx, llm=None)
    agents = bundle.crew.agents
    assert len(agents) == 3
    assert len({id(a) for a in agents}) == 3
    assert str(bundle.crew.process) == "Process.sequential"
    # Roles match the parsed runtime prompts exactly.
    for role_key, path in zip(_ROLES, prompt_loader.ANALYST_PROMPT_FILES, strict=True):
        parsed = prompt_loader.load_agent_prompt(path)
        assert bundle.agents[role_key].role == parsed.role
        assert bundle.agents[role_key].goal == parsed.goal
        assert bundle.agents[role_key].backstory == parsed.backstory


def test_tasks_are_ordered_with_growing_context(ctx) -> None:
    bundle = build_analyst_crew(ctx, llm=None)
    ctx_lengths = [len(bundle.tasks[r].context) for r in _ROLES]
    assert ctx_lengths == [0, 1, 2]  # each task references its predecessors
    for role_key in _ROLES:
        assert bundle.tasks[role_key].output_pydantic is _EXPECTED_MODEL[role_key]


def test_each_task_loads_shared_rules_and_its_role_prompt(ctx) -> None:
    mapping = ctx.placeholder_map()
    bundle = build_analyst_crew(ctx, llm=None)
    for role_key, path in zip(_ROLES, prompt_loader.ANALYST_PROMPT_FILES, strict=True):
        parsed = prompt_loader.load_agent_prompt(path)
        filled = prompt_loader.fill_placeholders(parsed.task_prompt, mapping)
        desc = bundle.tasks[role_key].description
        assert "Shared Runtime Rules" in desc  # shared rules present
        assert filled in desc  # the correct role-specific prompt present
        assert ctx.run_id in desc  # placeholders were substituted


# --- offline tool invocation (what the agents call) ------------------------ #
def test_tools_run_offline_without_network(ctx, no_network) -> None:
    tools = {t.name: t for group in build_analyst_tools(ctx).values() for t in group}
    clean_result = json.loads(tools["run_cleaning_pipeline"].run())
    assert clean_result["rows_preserved"] is True
    assert (ctx.analyst_dir / "clean_data.csv").exists()

    validated = json.loads(tools["validate_cleaned_handoff"].run())
    assert validated["passed"] is True

    eda_result = json.loads(tools["run_eda_pipeline"].run())
    assert eda_result["figure_count"] >= 5
    tools["render_analyst_reports"].run()
    assert json.loads(tools["validate_analyst_artifacts"].run())["passed"] is True


def test_write_tools_reject_false_pass(ctx) -> None:
    from pydantic import ValidationError

    tools = {t.name: t for group in build_analyst_tools(ctx).values() for t in group}
    # A PASS review with handoff_ready False must be rejected by the model.
    with pytest.raises(ValidationError):
        tools["write_source_quality_review"]._run(
            record={"status": "PASS", "summary": "x", "source": {}, "handoff_ready": False}
        )


# --- mocked run path (no real OpenAI call) --------------------------------- #
def test_run_analyst_crew_uses_mocked_llm_and_kickoff(ctx, monkeypatch, no_network) -> None:
    from crewai import Crew

    import retail_clickstream_ai.crews.llm as llm_mod

    calls: dict[str, object] = {}

    def fake_make_llm(settings=None):
        calls["llm"] = True
        return None  # Agents accept llm=None; no client is constructed

    def fake_kickoff(self, inputs=None):
        calls["kickoff_inputs"] = inputs
        return "SENTINEL_RESULT"

    monkeypatch.setattr(llm_mod, "make_llm", fake_make_llm)
    monkeypatch.setattr(Crew, "kickoff", fake_kickoff)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-model")

    result, context = run_analyst_crew(
        str(FIXTURE), run_id="mocked", pin_hash=False, require_all_months=True
    )
    assert result == "SENTINEL_RESULT"
    assert calls["llm"] is True
    assert calls["kickoff_inputs"]["run_id"] == context.run_id
