"""Stage 3 — structured-output models: valid, malformed, and false-PASS cases.

Proves the three Analyst handoff models accept well-formed records, reject
malformed ones (missing fields, bad enums, unknown keys), and — critically —
reject a *false PASS*: a ``status=PASS`` that the evidence does not support.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from retail_clickstream_ai.crews.analyst.models import (
    AnalystCrewHandoff,
    ArtifactRef,
    AuditRef,
    CleanDataRef,
    ContractRef,
    DataEngineeringHandoff,
    EdaEvidence,
    Issue,
    SourceInfo,
    SourceQualityReview,
    ValidationResult,
)


def _source_review(**over):
    base = {
        "status": "PASS",
        "run_id": "r1",
        "summary": "ok",
        "source": SourceInfo(input_sha256="abc"),
        "handoff_ready": True,
        "review_path": "artifacts/runs/r1/source_quality_review.json",
        "review_sha256": "deadbeef",
    }
    base.update(over)
    return SourceQualityReview(**base)


def _de_handoff(**over):
    base = {
        "status": "PASS",
        "run_id": "r1",
        "summary": "ok",
        "input_sha256": "abc",
        "clean_data": CleanDataRef(path="a.csv", sha256="h", row_count=10, column_count=15),
        "dataset_contract": ContractRef(path="c.json", sha256="h", contract_version="1.0.0"),
        "transformation_audit": AuditRef(
            path="au.json", sha256="h", rule_count=6, fatal_issue_count=0
        ),
        "validation": ValidationResult(passed=True, evidence_ref="tool:x#passed"),
        "handoff_ready": True,
        "handoff_path": "h.json",
        "handoff_sha256": "h",
    }
    base.update(over)
    return DataEngineeringHandoff(**base)


def _artifacts(names):
    return [
        ArtifactRef(name=n, path=f"artifacts/analyst/{n}", sha256="h", size_bytes=1) for n in names
    ]


def _analyst_handoff(**over):
    base = {
        "status": "PASS",
        "run_id": "r1",
        "summary": "ok",
        "input_sha256": "abc",
        "required_artifacts": _artifacts(
            ["clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json"]
        ),
        "eda_evidence": EdaEvidence(metrics_path="m.json", figure_count=6),
        "validation": ValidationResult(passed=True, evidence_ref="tool:x#passed"),
        "handoff_ready": True,
        "handoff_path": "h.json",
        "handoff_sha256": "h",
    }
    base.update(over)
    return AnalystCrewHandoff(**base)


# --- valid ----------------------------------------------------------------- #
def test_valid_records_construct() -> None:
    assert _source_review().status == "PASS"
    assert _de_handoff().status == "PASS"
    assert len(_analyst_handoff().required_artifacts) == 4


def test_fail_and_blocked_allow_nulls() -> None:
    r = SourceQualityReview(
        status="FAIL", run_id="r1", summary="bad", source=SourceInfo(), handoff_ready=False
    )
    assert r.status == "FAIL"
    assert r.review_path is None


# --- malformed ------------------------------------------------------------- #
def test_missing_required_field_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceQualityReview(
            status="PASS", run_id="r1", summary="x", handoff_ready=True
        )  # no source


def test_bad_status_enum_rejected() -> None:
    with pytest.raises(ValidationError):
        _source_review(status="OK")


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _source_review(unexpected="nope")


# --- false PASS ------------------------------------------------------------ #
def test_source_review_pass_requires_handoff_ready() -> None:
    with pytest.raises(ValidationError):
        _source_review(handoff_ready=False)


def test_source_review_pass_requires_written_review() -> None:
    with pytest.raises(ValidationError):
        _source_review(review_path=None)


def test_source_review_pass_rejects_fatal_issue() -> None:
    fatal = Issue(severity="fatal", rule="x", observed="y", evidence_ref="z", remediation="fix")
    with pytest.raises(ValidationError):
        _source_review(issues=[fatal])


def test_de_handoff_pass_requires_validation_passed() -> None:
    with pytest.raises(ValidationError):
        _de_handoff(validation=ValidationResult(passed=False, evidence_ref="tool:x#passed"))


def test_de_handoff_pass_requires_clean_hash() -> None:
    with pytest.raises(ValidationError):
        _de_handoff(clean_data=CleanDataRef(path="a.csv", sha256=None))


def test_analyst_handoff_pass_requires_four_artifacts() -> None:
    with pytest.raises(ValidationError):
        _analyst_handoff(required_artifacts=_artifacts(["clean_data.csv", "insights.md"]))


def test_analyst_handoff_pass_rejects_duplicate_artifacts() -> None:
    with pytest.raises(ValidationError):
        _analyst_handoff(
            required_artifacts=_artifacts(
                ["clean_data.csv", "clean_data.csv", "insights.md", "dataset_contract.json"]
            )
        )
