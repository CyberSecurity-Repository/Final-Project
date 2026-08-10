"""Stage 4 — Scientist report renderers.

Proves the reports contain their required sections and that a fabricated number
is rejected (the deterministic equivalent of the renderer refusing an
unsupported claim), so no figure can reach the report without machine evidence.
"""

from __future__ import annotations

import pytest

from retail_clickstream_ai.reporting.evidence import Claim, UnsupportedClaimError, validate_claims
from retail_clickstream_ai.reporting.model_report import (
    REQUIRED_EVAL_REPORT_MARKERS,
    REQUIRED_MODEL_CARD_MARKERS,
    render_evaluation_report,
    render_model_card,
)

_METRICS = {
    "selected_candidate_id": "baseline_transition",
    "selected_model": "current_category_transition_baseline",
    "primary_metric": "macro_f1",
    "seed": 42,
    "classes": [1, 2, 3, 4],
    "class_labels": {"1": "trousers", "2": "skirts", "3": "blouses", "4": "sale"},
    "selection_rule": "highest validation macro_f1; tie-break ...",
    "split_counts": {
        "train_rows": 100,
        "validation_rows": 30,
        "test_rows": 20,
        "train_sessions": 20,
        "validation_sessions": 6,
        "test_sessions": 4,
    },
    "validation": {
        "macro_f1": 0.81,
        "weighted_f1": 0.81,
        "accuracy": 0.82,
        "log_loss": 0.67,
        "per_class": {
            str(c): {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 25}
            for c in (1, 2, 3, 4)
        },
        "confusion_matrix": [[1, 0, 0, 0]] * 4,
        "class_distribution": {str(c): 25 for c in (1, 2, 3, 4)},
    },
    "test": {
        "macro_f1": 0.82,
        "weighted_f1": 0.82,
        "accuracy": 0.83,
        "log_loss": 0.66,
        "per_class": {
            str(c): {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 5} for c in (1, 2, 3, 4)
        },
        "confusion_matrix": [[1, 0, 0, 0]] * 4,
        "class_distribution": {str(c): 5 for c in (1, 2, 3, 4)},
    },
    "candidates": {
        "baseline_transition": {
            "family": "current_category_transition_baseline",
            "best_params": {},
            "macro_f1": 0.81,
            "weighted_f1": 0.81,
            "accuracy": 0.82,
            "log_loss": 0.67,
        },
        "logistic_regression": {
            "family": "multinomial_logistic_regression",
            "best_params": {"C": 1.0},
            "macro_f1": 0.80,
            "weighted_f1": 0.80,
            "accuracy": 0.81,
            "log_loss": 0.66,
        },
        "random_forest": {
            "family": "random_forest",
            "best_params": {"n_estimators": 200, "max_depth": None},
            "macro_f1": 0.79,
            "weighted_f1": 0.79,
            "accuracy": 0.80,
            "log_loss": 0.68,
        },
    },
    "validation_ranking": [
        {"rank": 1, "candidate_id": "baseline_transition", "macro_f1": 0.81, "evidence_ref": "e"}
    ],
    "timing": {"winner_train_seconds": 0.1, "winner_inference_seconds": 0.01},
}

_METADATA = {
    "training_data_sha256": "abc123",
    "artifact_sha256": "def456",
    "seed": 42,
    "python_version": "3.13.9",
    "round_trip_validated": True,
}


def test_evaluation_report_has_required_sections() -> None:
    md, _ = render_evaluation_report(_METRICS)
    for marker in REQUIRED_EVAL_REPORT_MARKERS:
        assert marker in md, marker


def test_model_card_has_required_sections() -> None:
    md, _ = render_model_card(_METRICS, _METADATA)
    for marker in REQUIRED_MODEL_CARD_MARKERS:
        assert marker in md, marker


def test_renderers_are_deterministic() -> None:
    assert render_evaluation_report(_METRICS)[0] == render_evaluation_report(_METRICS)[0]
    assert render_model_card(_METRICS, _METADATA)[0] == render_model_card(_METRICS, _METADATA)[0]


def test_every_evaluation_number_is_backed_by_metrics() -> None:
    _, ev = render_evaluation_report(_METRICS)
    ev.validate()  # would raise if any claim did not match its metric


def test_fabricated_claim_is_rejected() -> None:
    # A number that does not match its cited metric must be refused.
    bad = Claim(ref=f"{'artifacts/scientist/metrics.json'}#test.macro_f1", stated=0.99, text="99%")
    with pytest.raises(UnsupportedClaimError):
        validate_claims([bad], _METRICS)
