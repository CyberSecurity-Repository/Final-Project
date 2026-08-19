"""CrewAI Scientist crew: wiring, prompt loading, offline tool runs.

All tests are offline: they build and inspect the crew, drive the deterministic
tools directly (network blocked), and exercise the run path with the LLM factory
and ``kickoff`` mocked — never a real OpenAI call.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from crewai import TaskOutput

from retail_clickstream_ai.crews import prompts as prompt_loader
from retail_clickstream_ai.crews.context import ScientistRunContext
from retail_clickstream_ai.crews.scientist.crew import build_scientist_crew, run_scientist_crew
from retail_clickstream_ai.crews.scientist.specs import SCIENTIST_SPECS
from retail_clickstream_ai.crews.scientist.tools import build_scientist_tools

_ROLES = ("contract_feature_engineer", "model_trainer", "evaluation_governance")


def _guardrail_probe_output() -> TaskOutput:
    """A minimal, valid TaskOutput — only description/agent are required."""
    return TaskOutput(description="task", agent="agent", raw="agent's raw final answer")


@pytest.fixture
def ctx(scientist_env: Any) -> ScientistRunContext:
    context = ScientistRunContext.build(run_id="test", pin_hash=False)
    context.scientist_dir.mkdir(parents=True, exist_ok=True)
    context.run_dir.mkdir(parents=True, exist_ok=True)
    return context


def _flat_tools(context: ScientistRunContext) -> dict[str, Any]:
    return {t.name: t for group in build_scientist_tools(context).values() for t in group}


# --- prompt specs ---------------------------------------------------------- #
def test_all_scientist_specs_populated() -> None:
    assert prompt_loader.SHARED_RUNTIME_RULES.strip()
    assert tuple(SCIENTIST_SPECS) == _ROLES
    for spec in SCIENTIST_SPECS.values():
        assert spec.role and spec.goal and spec.backstory
        assert spec.task_prompt and spec.expected_output


# --- crew wiring ----------------------------------------------------------- #
def test_crew_has_three_distinct_agents_sequential(ctx: ScientistRunContext) -> None:
    bundle = build_scientist_crew(ctx, llm=None)
    agents = bundle.crew.agents
    assert len(agents) == 3
    assert len({id(a) for a in agents}) == 3
    assert str(bundle.crew.process) == "Process.sequential"
    for role_key in _ROLES:
        spec = SCIENTIST_SPECS[role_key]
        assert bundle.agents[role_key].role == spec.role
        assert bundle.agents[role_key].goal == spec.goal
        assert bundle.agents[role_key].backstory == spec.backstory


def test_tasks_are_ordered_with_growing_context(ctx: ScientistRunContext) -> None:
    bundle = build_scientist_crew(ctx, llm=None)
    ctx_lengths = [len(bundle.tasks[r].context) for r in _ROLES]
    assert ctx_lengths == [0, 1, 2]
    for role_key in _ROLES:
        # Tasks intentionally set no output_pydantic (weak models can't author the
        # machine-stamped handoff fields); the write_* tools persist the validated
        # handoff to disk and the deterministic gates are the authority.
        assert bundle.tasks[role_key].output_pydantic is None


def test_tasks_are_wired_with_handoff_guardrails(ctx: ScientistRunContext) -> None:
    """Every task has a completion guardrail — the backstop for a weak model that
    stops calling tools before reaching its terminal write_* tool."""
    bundle = build_scientist_crew(ctx, llm=None)
    expected_retries = {
        "contract_feature_engineer": 4,
        "model_trainer": 3,
        "evaluation_governance": 4,
    }
    for role_key in _ROLES:
        task = bundle.tasks[role_key]
        assert task.guardrail is not None
        assert task.guardrail_max_retries == expected_retries[role_key]


def test_contract_feature_engineer_guardrail_blocks_until_handoff_written(
    ctx: ScientistRunContext,
) -> None:
    bundle = build_scientist_crew(ctx, llm=None)
    guardrail = bundle.tasks["contract_feature_engineer"].guardrail

    # No tool has run yet in this run — the guardrail must refuse.
    ok, message = guardrail(_guardrail_probe_output())
    assert ok is False
    assert "write_feature_engineering_handoff" in message

    # Drive the real tool chain, then the same stored guardrail must pass.
    tools = _flat_tools(ctx)
    tools["run_feature_pipeline"]._run()
    tools["run_leakage_audit"]._run()
    tools["write_feature_engineering_handoff"]._run()
    ok2, _ = guardrail(_guardrail_probe_output())
    assert ok2 is True


def test_each_task_loads_shared_rules_and_its_role_prompt(ctx: ScientistRunContext) -> None:
    mapping = ctx.placeholder_map()
    bundle = build_scientist_crew(ctx, llm=None)
    for role_key in _ROLES:
        spec = SCIENTIST_SPECS[role_key]
        filled = prompt_loader.fill_placeholders(spec.task_prompt, mapping)
        desc = bundle.tasks[role_key].description
        assert "Shared Runtime Rules" in desc
        assert filled in desc
        assert ctx.run_id in desc


# --- offline tool invocation (the full deterministic chain) ---------------- #
def test_tool_chain_runs_offline_without_network(ctx: ScientistRunContext, no_network: Any) -> None:
    tools = _flat_tools(ctx)

    assert json.loads(tools["validate_analyst_handoff"].run())["passed"] is True

    fp = json.loads(tools["run_feature_pipeline"].run())
    assert fp["features"]["predictor_count"] == 12
    assert fp["features"]["row_count"] > 0

    assert json.loads(tools["run_leakage_audit"].run())["passed"] is True
    assert json.loads(tools["validate_feature_handoff"].run())["passed"] is True

    ec = json.loads(tools["read_experiment_config"].run())
    assert ec["primary_metric"] == "macro_f1"

    ce = json.loads(tools["run_candidate_experiments"].run())
    assert ce["candidate_results"]["candidate_count"] >= 3
    assert ce["test_accessed"] is False
    assert json.loads(tools["validate_training_outputs"].run())["passed"] is True

    # Trainer writes its handoff; governance validates it.
    record = {
        "status": "PASS",
        "summary": "trained",
        "experiment_config": {
            "path": ec["path"],
            "sha256": ec["sha256"],
            "seed": ec["seed"],
            "primary_metric": "macro_f1",
        },
        "candidate_results": ce["candidate_results"],
        "candidate_artifact_manifest": ce["candidate_artifact_manifest"],
        "validation_ranking": ce["validation_ranking"],
        "test_accessed": ce["test_accessed"],
        "validation": {"passed": True, "evidence_ref": "tool:validate_training_outputs#passed"},
        "handoff_ready": True,
    }
    assert json.loads(tools["write_training_handoff"]._run(record=record))["status"] == "PASS"
    vth = json.loads(tools["validate_training_handoff"].run())
    assert vth["passed"] is True and vth["test_accessed"] is False

    lk = json.loads(tools["lock_winner_and_evaluate_test"].run())
    assert lk["performed_once"] is True
    assert lk["model_metadata"]["round_trip_validated"] is True

    rb = json.loads(tools["read_evaluation_bundle"].run())
    assert "test" in rb and "validation" in rb

    assert "evaluation_report" in json.loads(tools["render_scientist_reports"].run())
    assert json.loads(tools["validate_scientist_artifacts"].run())["passed"] is True

    for name in ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md"):
        assert (ctx.scientist_dir / name).exists()


def test_write_tool_derives_status_from_evidence(ctx: ScientistRunContext) -> None:
    tools = _flat_tools(ctx)
    # Drive agent 1's real tool chain to lay down the on-disk evidence.
    tools["run_feature_pipeline"]._run()
    tools["run_leakage_audit"]._run()

    # The write tool builds the handoff from that evidence and ignores any injected
    # status/handoff_ready — a false PASS/BLOCKED cannot be forced, and a weak/empty
    # LLM answer cannot crash it on machine-only fields.
    out = tools["write_feature_engineering_handoff"]._run(
        record={"status": "BLOCKED", "handoff_ready": False, "summary": "agent note"}
    )
    assert json.loads(out)["status"] == "PASS"  # evidence-derived, not the injected value
    written = json.loads(
        (ctx.run_dir / "feature_engineering_handoff.json").read_text(encoding="utf-8")
    )
    assert written["status"] == "PASS"
    assert written["handoff_ready"] is True
    assert written["summary"] == "agent note"  # only the interpretive summary is honored
    assert written["features"]["sha256"]  # machine field filled from disk, not the LLM
    assert written["leakage_audit"]["passed"] is True

    # With no agent record at all, the tool still writes a valid, evidence-backed handoff.
    assert json.loads(tools["write_feature_engineering_handoff"]._run())["status"] == "PASS"


def test_write_feature_engineering_handoff_reports_missing_prerequisite(
    ctx: ScientistRunContext,
) -> None:
    """If an agent jumps to the terminal tool before run_feature_pipeline, it gets a
    clear message — not a raw exception — and no handoff is written."""
    tools = _flat_tools(ctx)
    out = json.loads(tools["write_feature_engineering_handoff"]._run())
    assert "error" in out
    assert "run_feature_pipeline" in out["error"]
    assert not (ctx.run_dir / "feature_engineering_handoff.json").exists()


def test_write_training_handoff_reports_missing_prerequisite(ctx: ScientistRunContext) -> None:
    tools = _flat_tools(ctx)
    out = json.loads(tools["write_training_handoff"]._run())
    assert "error" in out
    assert "run_candidate_experiments" in out["error"]
    assert not (ctx.run_dir / "training_handoff.json").exists()


def test_write_scientist_handoff_reports_missing_prerequisite(ctx: ScientistRunContext) -> None:
    tools = _flat_tools(ctx)
    out = json.loads(tools["write_scientist_handoff"]._run())
    assert "error" in out
    assert "lock_winner_and_evaluate_test" in out["error"]
    assert not (ctx.run_dir / "scientist_crew_handoff.json").exists()


# --- mocked run path (no real OpenAI call) --------------------------------- #
def test_run_scientist_crew_uses_mocked_llm_and_kickoff(
    scientist_env: Any, monkeypatch: pytest.MonkeyPatch, no_network: Any
) -> None:
    from crewai import Crew

    import retail_clickstream_ai.crews.llm as llm_mod

    calls: dict[str, object] = {}

    def fake_make_llm(settings: Any = None) -> None:
        calls["llm"] = True
        return None

    def fake_kickoff(self: Any, inputs: Any = None) -> str:
        calls["kickoff_inputs"] = inputs
        return "SENTINEL_RESULT"

    monkeypatch.setattr(llm_mod, "make_llm", fake_make_llm)
    monkeypatch.setattr(Crew, "kickoff", fake_kickoff)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-model")

    result, context = run_scientist_crew(run_id="mocked")
    assert result == "SENTINEL_RESULT"
    assert calls["llm"] is True
    assert calls["kickoff_inputs"]["run_id"] == context.run_id
