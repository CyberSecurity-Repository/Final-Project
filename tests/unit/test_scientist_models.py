"""Stage 4 — Scientist structured-output models.

Proves the three handoff models accept valid evidence, reject malformed payloads,
and — critically — reject a *false PASS*: a PASS that the evidence does not
support (missing hashes, a failed leakage audit, prior test access, fewer than
three candidates, or not exactly four artifacts).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from retail_clickstream_ai.crews.scientist import models as m


def _feature_handoff() -> dict:
    return {
        "status": "PASS",
        "run_id": "r",
        "summary": "ok",
        "input_sha256": "h",
        "features": {
            "path": "artifacts/scientist/features.csv",
            "sha256": "x",
            "row_count": 10,
            "predictor_count": 12,
            "target_name": "next_main_category",
        },
        "split_manifest": {
            "path": "artifacts/scientist/split_manifest.json",
            "sha256": "y",
            "train_rows": 6,
            "validation_rows": 2,
            "test_rows": 2,
        },
        "feature_schema_path": "artifacts/scientist/feature_schema.json",
        "leakage_audit": {
            "path": "artifacts/scientist/leakage_audit.json",
            "sha256": "z",
            "passed": True,
        },
        "validation": {"passed": True, "evidence_ref": "tool:#p"},
        "handoff_ready": True,
        "handoff_path": "artifacts/runs/r/feature_engineering_handoff.json",
        "handoff_sha256": "hh",
    }


def _training_handoff() -> dict:
    return {
        "status": "PASS",
        "run_id": "r",
        "summary": "ok",
        "experiment_config": {
            "path": "artifacts/scientist/experiment_config.json",
            "sha256": "x",
            "seed": 42,
            "primary_metric": "macro_f1",
        },
        "candidate_results": {
            "path": "artifacts/scientist/candidate_results.json",
            "sha256": "y",
            "candidate_count": 3,
            "all_required_candidates_present": True,
        },
        "candidate_artifact_manifest": {"path": "artifacts/runs/r/m.json", "sha256": "z"},
        "validation_ranking": [
            {"rank": 1, "candidate_id": "baseline_transition", "macro_f1": 0.8, "evidence_ref": "e"}
        ],
        "test_accessed": False,
        "validation": {"passed": True, "evidence_ref": "tool:#p"},
        "handoff_ready": True,
        "handoff_path": "artifacts/runs/r/training_handoff.json",
        "handoff_sha256": "hh",
    }


def _scientist_handoff() -> dict:
    return {
        "status": "PASS",
        "run_id": "r",
        "summary": "ok",
        "required_artifacts": [
            {"name": "features.csv", "path": "a/features.csv", "sha256": "1", "size_bytes": 10},
            {"name": "model.joblib", "path": "a/model.joblib", "sha256": "2", "size_bytes": 10},
            {
                "name": "evaluation_report.md",
                "path": "a/evaluation_report.md",
                "sha256": "3",
                "size_bytes": 10,
            },
            {"name": "model_card.md", "path": "a/model_card.md", "sha256": "4", "size_bytes": 10},
        ],
        "selection": {
            "primary_metric": "macro_f1",
            "selected_candidate_id": "baseline_transition",
            "validation_metric_value": 0.8,
            "selection_evidence_ref": "metrics.json#validation.macro_f1",
        },
        "test_evaluation": {
            "performed_once": True,
            "macro_f1": 0.82,
            "evidence_ref": "metrics.json#test.macro_f1",
        },
        "model_metadata": {
            "path": "artifacts/scientist/model_metadata.json",
            "sha256": "mm",
            "round_trip_validated": True,
        },
        "metrics": {"path": "artifacts/scientist/metrics.json", "sha256": "mx"},
        "validation": {"passed": True, "evidence_ref": "tool:#p"},
        "handoff_ready": True,
        "handoff_path": "artifacts/runs/r/scientist_crew_handoff.json",
        "handoff_sha256": "hh",
    }


# --- valid ----------------------------------------------------------------- #
def test_valid_handoffs_construct() -> None:
    assert m.FeatureEngineeringHandoff.model_validate(_feature_handoff()).status == "PASS"
    assert m.TrainingRunHandoff.model_validate(_training_handoff()).status == "PASS"
    assert m.ScientistCrewHandoff.model_validate(_scientist_handoff()).status == "PASS"


# --- malformed ------------------------------------------------------------- #
def test_unknown_field_rejected() -> None:
    rec = _feature_handoff() | {"surprise": 1}
    with pytest.raises(ValidationError):
        m.FeatureEngineeringHandoff.model_validate(rec)


def test_bad_status_enum_rejected() -> None:
    rec = _training_handoff() | {"status": "OK"}
    with pytest.raises(ValidationError):
        m.TrainingRunHandoff.model_validate(rec)


# --- false PASS ------------------------------------------------------------ #
def test_feature_pass_requires_passing_leakage_audit() -> None:
    rec = _feature_handoff()
    rec["leakage_audit"]["passed"] = False
    with pytest.raises(ValidationError):
        m.FeatureEngineeringHandoff.model_validate(rec)


def test_feature_pass_requires_hashes() -> None:
    rec = _feature_handoff()
    rec["features"]["sha256"] = None
    with pytest.raises(ValidationError):
        m.FeatureEngineeringHandoff.model_validate(rec)


def test_training_pass_rejects_prior_test_access() -> None:
    rec = _training_handoff()
    rec["test_accessed"] = True
    with pytest.raises(ValidationError):
        m.TrainingRunHandoff.model_validate(rec)


def test_training_pass_requires_three_candidates() -> None:
    rec = _training_handoff()
    rec["candidate_results"]["candidate_count"] = 2
    with pytest.raises(ValidationError):
        m.TrainingRunHandoff.model_validate(rec)


def test_scientist_pass_requires_four_unique_artifacts() -> None:
    rec = _scientist_handoff()
    rec["required_artifacts"] = rec["required_artifacts"][:3]
    with pytest.raises(ValidationError):
        m.ScientistCrewHandoff.model_validate(rec)


def test_scientist_pass_requires_test_performed_once() -> None:
    rec = _scientist_handoff()
    rec["test_evaluation"]["performed_once"] = False
    with pytest.raises(ValidationError):
        m.ScientistCrewHandoff.model_validate(rec)


def test_scientist_pass_requires_metrics_evidence() -> None:
    rec = _scientist_handoff()
    rec["metrics"]["sha256"] = None
    with pytest.raises(ValidationError):
        m.ScientistCrewHandoff.model_validate(rec)
