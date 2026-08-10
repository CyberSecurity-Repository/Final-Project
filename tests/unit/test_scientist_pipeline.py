"""Stage 4 — the deterministic Scientist pipeline (end-to-end + fail-closed).

Runs the whole stage on synthetic-but-contract-valid Analyst artifacts and proves
the four required artifacts are produced and validate, the run is repeatable, and
the pipeline fails closed (writing a failure report, never repairing Analyst
output) when the Analyst handoff is invalid.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from retail_clickstream_ai import paths
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline.scientist_pipeline import (
    ScientistPipelineError,
    run_scientist_pipeline,
)
from retail_clickstream_ai.validation import artifacts as artifact_validation

_REQUIRED = ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md")


def test_pipeline_produces_and_validates_four_artifacts(scientist_env: Any) -> None:
    result = run_scientist_pipeline(pin_analyst_hash=False, run_id="scientist-synth")
    assert result.ok
    for name in _REQUIRED:
        assert result.artifacts[name].exists()

    report = artifact_validation.validate_scientist_artifacts(
        features_path=result.scientist_dir / "features.csv",
        model_path=result.scientist_dir / "model.joblib",
        eval_report_path=result.scientist_dir / "evaluation_report.md",
        model_card_path=result.scientist_dir / "model_card.md",
        metrics_path=result.scientist_dir / "metrics.json",
        metadata_path=result.scientist_dir / "model_metadata.json",
    )
    assert report.ok, report.render()

    for handoff in result.handoffs.values():
        assert handoff.exists()
    final = json.loads(result.handoffs["scientist_crew_handoff"].read_text(encoding="utf-8"))
    assert final["status"] == "PASS"
    assert final["test_evaluation"]["performed_once"] is True
    assert len(final["required_artifacts"]) == 4


def test_pipeline_is_repeatable(scientist_env: Any) -> None:
    r1 = run_scientist_pipeline(pin_analyst_hash=False, run_id="run-a", render_figures=False)
    sha1 = d.sha256_file(r1.scientist_dir / "features.csv")
    macro1 = r1.metrics["test"]["macro_f1"]

    r2 = run_scientist_pipeline(pin_analyst_hash=False, run_id="run-b", render_figures=False)
    sha2 = d.sha256_file(r2.scientist_dir / "features.csv")
    macro2 = r2.metrics["test"]["macro_f1"]

    assert sha1 == sha2  # features.csv is byte-reproducible
    assert abs(macro1 - macro2) < 1e-9  # metrics reproduce under the fixed seed


def test_pipeline_fails_closed_on_invalid_analyst_handoff(scientist_env: Any) -> None:
    # Corrupt an Analyst artifact after setup; the Scientist gate must refuse.
    (paths.analyst_artifacts() / "insights.md").write_text("", encoding="utf-8")
    with pytest.raises(ScientistPipelineError) as exc:
        run_scientist_pipeline(pin_analyst_hash=False, run_id="scientist-bad")
    failure_report = exc.value.failure_report
    assert failure_report.exists()
    payload = json.loads(failure_report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["stage"] == "analyst_handoff"
    assert payload["downstream"].startswith("blocked")
