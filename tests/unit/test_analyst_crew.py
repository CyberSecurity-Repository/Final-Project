"""CrewAI Analyst crew: wiring, prompt loading, offline tool runs.

All tests are offline: they build and inspect the crew, drive the deterministic
tools directly (with the network blocked), and exercise the run path with the
LLM factory and ``kickoff`` mocked — never a real OpenAI call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from crewai import TaskOutput

from retail_clickstream_ai.crews import prompts as prompt_loader
from retail_clickstream_ai.crews.analyst.crew import build_analyst_crew, run_analyst_crew
from retail_clickstream_ai.crews.analyst.specs import ANALYST_SPECS
from retail_clickstream_ai.crews.analyst.tools import build_analyst_tools
from retail_clickstream_ai.crews.context import AnalystRunContext
from retail_clickstream_ai.pipeline.analyst_pipeline import prepare_run
from retail_clickstream_ai.validation.contract import DatasetContract

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "clickstream_sample.csv"

_ROLES = ("source_quality", "data_engineer", "eda_business")


def _guardrail_probe_output() -> TaskOutput:
    """A minimal, valid TaskOutput — only description/agent are required."""
    return TaskOutput(description="task", agent="agent", raw="agent's raw final answer")


@pytest.fixture
def ctx(monkeypatch, tmp_path) -> AnalystRunContext:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    context = AnalystRunContext.build(
        FIXTURE, run_id="test", pin_hash=False, require_all_months=True
    )
    context.run_dir.mkdir(parents=True, exist_ok=True)
    prepare_run(FIXTURE, DatasetContract.build(), context.run_dir, require_all_months=True)
    return context


# --- prompt specs ---------------------------------------------------------- #
def test_all_analyst_specs_populated() -> None:
    assert prompt_loader.SHARED_RUNTIME_RULES.strip()
    assert tuple(ANALYST_SPECS) == _ROLES
    for spec in ANALYST_SPECS.values():
        assert spec.role and spec.goal and spec.backstory
        assert spec.task_prompt and spec.expected_output


# --- crew wiring ----------------------------------------------------------- #
def test_crew_has_three_distinct_agents_sequential(ctx) -> None:
    bundle = build_analyst_crew(ctx, llm=None)
    agents = bundle.crew.agents
    assert len(agents) == 3
    assert len({id(a) for a in agents}) == 3
    assert str(bundle.crew.process) == "Process.sequential"
    # Roles match the inline runtime specs exactly.
    for role_key in _ROLES:
        spec = ANALYST_SPECS[role_key]
        assert bundle.agents[role_key].role == spec.role
        assert bundle.agents[role_key].goal == spec.goal
        assert bundle.agents[role_key].backstory == spec.backstory


def test_tasks_are_ordered_with_growing_context(ctx) -> None:
    bundle = build_analyst_crew(ctx, llm=None)
    ctx_lengths = [len(bundle.tasks[r].context) for r in _ROLES]
    assert ctx_lengths == [0, 1, 2]  # each task references its predecessors
    for role_key in _ROLES:
        # Tasks intentionally set no output_pydantic (weak models can't author the
        # machine-stamped handoff fields); the write_* tools persist the validated
        # handoff to disk and the deterministic gates are the authority.
        assert bundle.tasks[role_key].output_pydantic is None


def test_tasks_are_wired_with_handoff_guardrails(ctx) -> None:
    """Every task has a completion guardrail — the backstop for a weak model that
    stops calling tools before reaching its terminal write_* tool."""
    bundle = build_analyst_crew(ctx, llm=None)
    expected_retries = {"source_quality": 3, "data_engineer": 3, "eda_business": 3}
    for role_key in _ROLES:
        task = bundle.tasks[role_key]
        assert task.guardrail is not None
        assert task.guardrail_max_retries == expected_retries[role_key]


def test_data_engineer_guardrail_blocks_until_handoff_written(ctx) -> None:
    bundle = build_analyst_crew(ctx, llm=None)
    guardrail = bundle.tasks["data_engineer"].guardrail

    # No tool has run yet in this run — the guardrail must refuse.
    ok, message = guardrail(_guardrail_probe_output())
    assert ok is False
    assert "write_data_engineering_handoff" in message

    # Drive the real tool chain, then the same stored guardrail must pass.
    tools = {t.name: t for group in build_analyst_tools(ctx).values() for t in group}
    tools["run_cleaning_pipeline"]._run()
    tools["write_data_engineering_handoff"]._run()
    ok2, output2 = guardrail(_guardrail_probe_output())
    assert ok2 is True


def test_each_task_loads_shared_rules_and_its_role_prompt(ctx) -> None:
    mapping = ctx.placeholder_map()
    bundle = build_analyst_crew(ctx, llm=None)
    for role_key in _ROLES:
        spec = ANALYST_SPECS[role_key]
        filled = prompt_loader.fill_placeholders(spec.task_prompt, mapping)
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


def test_write_tool_derives_status_from_evidence(ctx) -> None:
    import json

    tools = {t.name: t for group in build_analyst_tools(ctx).values() for t in group}
    # The tool builds the handoff from the trusted on-disk evidence and ignores any
    # status/handoff_ready the agent tries to inject — so a *false PASS* (or here, a
    # false BLOCKED) cannot be forced, and a weak/empty LLM answer cannot crash it.
    out = tools["write_source_quality_review"]._run(
        record={"status": "BLOCKED", "handoff_ready": False, "summary": "agent note"}
    )
    assert json.loads(out)["status"] == "PASS"  # evidence-derived, not the injected value
    written = json.loads((ctx.run_dir / "source_quality_review.json").read_text(encoding="utf-8"))
    assert written["status"] == "PASS"
    assert written["handoff_ready"] is True
    assert written["summary"] == "agent note"  # only the interpretive summary is honored
    assert written["review_sha256"] and written["review_sha256"] != "pending"

    # With no agent record at all, the tool still writes a valid, evidence-backed handoff.
    assert json.loads(tools["write_source_quality_review"]._run())["status"] == "PASS"


def test_write_data_engineering_handoff_reports_missing_prerequisite(ctx) -> None:
    """If an agent jumps to the terminal tool before run_cleaning_pipeline, it gets a
    clear message — not a raw exception — and no handoff is written."""
    tools = {t.name: t for group in build_analyst_tools(ctx).values() for t in group}
    out = json.loads(tools["write_data_engineering_handoff"]._run())
    assert "error" in out
    assert "run_cleaning_pipeline" in out["error"]
    assert not (ctx.run_dir / "data_engineering_handoff.json").exists()


def test_write_analyst_handoff_reports_missing_prerequisite(ctx) -> None:
    tools = {t.name: t for group in build_analyst_tools(ctx).values() for t in group}
    out = json.loads(tools["write_analyst_handoff"]._run())
    assert "error" in out
    assert "run_eda_pipeline" in out["error"]
    assert not (ctx.run_dir / "analyst_crew_handoff.json").exists()


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
