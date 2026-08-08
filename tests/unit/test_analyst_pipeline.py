"""Stage 3 — the deterministic Analyst pipeline end-to-end (offline, fixture).

Proves the happy path produces four non-empty artifacts and is repeatable, and
that the failure path stops, writes a structured failure report, and never emits
a misleading success artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retail_clickstream_ai.pipeline.analyst_pipeline import (
    AnalystPipelineError,
    run_analyst_pipeline,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "clickstream_sample.csv"

REQUIRED = ("clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json")


def test_pipeline_produces_four_nonempty_artifacts(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    result = run_analyst_pipeline(
        FIXTURE, pin_contract_hash=False, require_all_months=True, run_id="fix"
    )
    assert result.status == "PASS"
    assert result.ok
    for name in REQUIRED:
        path = result.artifacts[name]
        assert path.exists() and path.stat().st_size > 0
    assert result.figure_count >= 5
    assert result.validation is not None and result.validation.ok
    # All three handoff records were written.
    for handoff in result.handoffs.values():
        assert handoff.exists()
    handoff = json.loads(result.handoffs["analyst_crew_handoff"].read_text())
    assert handoff["status"] == "PASS"
    assert {a["name"] for a in handoff["required_artifacts"]} == set(REQUIRED)


def test_pipeline_is_repeatable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    r1 = run_analyst_pipeline(FIXTURE, pin_contract_hash=False, run_id="a")
    clean1 = r1.artifacts["clean_data.csv"].read_bytes()
    insights1 = r1.artifacts["insights.md"].read_text()
    metrics1 = json.dumps(r1.metrics, sort_keys=True)

    r2 = run_analyst_pipeline(FIXTURE, pin_contract_hash=False, run_id="b")
    assert r2.artifacts["clean_data.csv"].read_bytes() == clean1
    assert r2.artifacts["insights.md"].read_text() == insights1
    assert json.dumps(r2.metrics, sort_keys=True) == metrics1


def test_pipeline_fails_closed_and_writes_failure_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    # Corrupt one main-category value to an out-of-domain code.
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(";")
    parts[6] = "9"  # page 1 (main category) -> invalid
    lines[1] = ";".join(parts)
    broken = tmp_path / "broken.csv"
    broken.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(AnalystPipelineError) as excinfo:
        run_analyst_pipeline(broken, pin_contract_hash=False, require_all_months=True, run_id="bad")

    failure = excinfo.value.failure_report
    assert failure.exists()
    payload = json.loads(failure.read_text())
    assert payload["status"] == "FAIL"
    assert payload["remediation"]
    assert payload["downstream"].startswith("blocked")
    # No misleading success artifact was written.
    assert not (tmp_path / "analyst" / "clean_data.csv").exists()
