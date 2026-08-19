"""``crews.guardrails``: the completion-guardrail backstop.

Fully offline: no crew/LLM machinery, no network. Exercises the guardrail
closure directly against a real :class:`AnalystRunContext` and a real
``TaskOutput``, proving the three documented guarantees: it fails until the
terminal ``write_*`` tool has run in *this* run, and it never inspects
PASS/BLOCKED status — only whether the handoff record exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from crewai import TaskOutput

from retail_clickstream_ai.crews.context import AnalystRunContext
from retail_clickstream_ai.crews.guardrails import build_handoff_guardrail
from retail_clickstream_ai.pipeline.analyst_pipeline import prepare_run
from retail_clickstream_ai.validation.contract import DatasetContract

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "clickstream_sample.csv"


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AnalystRunContext:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    context = AnalystRunContext.build(
        FIXTURE, run_id="guardrail-test", pin_hash=False, require_all_months=True
    )
    context.run_dir.mkdir(parents=True, exist_ok=True)
    prepare_run(FIXTURE, DatasetContract.build(), context.run_dir, require_all_months=True)
    return context


def _output() -> TaskOutput:
    """A minimal, valid TaskOutput — only description/agent are required."""
    return TaskOutput(description="task", agent="agent", raw="agent's raw final answer")


def test_fails_when_handoff_file_is_missing(ctx: AnalystRunContext, no_network: None) -> None:
    guardrail = build_handoff_guardrail(
        ctx, handoff_filename="source_quality_review.json", tool_name="write_source_quality_review"
    )
    ok, message = guardrail(_output())
    assert ok is False
    assert "write_source_quality_review" in message
    assert "source_quality_review.json" in message


def test_fails_when_handoff_file_is_empty(ctx: AnalystRunContext, no_network: None) -> None:
    (ctx.run_dir / "source_quality_review.json").write_text("", encoding="utf-8")
    guardrail = build_handoff_guardrail(
        ctx, handoff_filename="source_quality_review.json", tool_name="write_source_quality_review"
    )
    ok, message = guardrail(_output())
    assert ok is False
    assert "write_source_quality_review" in message


def test_fails_when_handoff_file_is_not_valid_json(
    ctx: AnalystRunContext, no_network: None
) -> None:
    (ctx.run_dir / "source_quality_review.json").write_text("{not json", encoding="utf-8")
    guardrail = build_handoff_guardrail(
        ctx, handoff_filename="source_quality_review.json", tool_name="write_source_quality_review"
    )
    ok, message = guardrail(_output())
    assert ok is False
    assert "not valid JSON" in message
    assert "write_source_quality_review" in message


def test_fails_when_handoff_belongs_to_a_different_run(
    ctx: AnalystRunContext, no_network: None
) -> None:
    (ctx.run_dir / "source_quality_review.json").write_text(
        json.dumps({"run_id": "some-other-run", "status": "PASS"}), encoding="utf-8"
    )
    guardrail = build_handoff_guardrail(
        ctx, handoff_filename="source_quality_review.json", tool_name="write_source_quality_review"
    )
    ok, message = guardrail(_output())
    assert ok is False
    assert "some-other-run" in message
    assert ctx.run_id in message


def test_passes_when_handoff_matches_this_run_regardless_of_status(
    ctx: AnalystRunContext, no_network: None
) -> None:
    """The guardrail's only job is 'did the terminal tool run' — never 'did it PASS'."""
    (ctx.run_dir / "source_quality_review.json").write_text(
        json.dumps({"run_id": ctx.run_id, "status": "BLOCKED"}), encoding="utf-8"
    )
    guardrail = build_handoff_guardrail(
        ctx, handoff_filename="source_quality_review.json", tool_name="write_source_quality_review"
    )
    output = _output()
    ok, returned = guardrail(output)
    assert ok is True
    assert returned is output  # unchanged pass-through, never mutated or re-judged
