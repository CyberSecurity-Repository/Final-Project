"""Structured output models for the Scientist crew handoffs.

The three validated control records the Scientist agents return —
``FeatureEngineeringHandoff``, ``TrainingRunHandoff``, and
``ScientistCrewHandoff``. They mirror the JSON schemas in the runtime prompt
files (``scientist/specs.py``) exactly, and each carries an ``after`` validator
that enforces the shared-rules invariant: a ``PASS`` status is only legal when
the deterministic evidence supports it. That makes a *false PASS* — including a
training handoff that admits prior test access, or a crew handoff that never had
its one-time test evaluation tool-confirmed — a hard validation error rather
than a silently accepted claim.

Imports only Pydantic (never CrewAI), so the deterministic pipeline can build
and validate these records without any LLM dependency. Shared primitives are
reused from the Analyst models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from retail_clickstream_ai.crews.analyst.models import (
    Issue,
    Limitation,
    Status,
    ValidationResult,
    _has_fatal,
    _Strict,
)

ScientistArtifactName = Literal[
    "features.csv", "model.joblib", "evaluation_report.md", "model_card.md"
]


# --------------------------------------------------------------------------- #
# Agent 1 — Contract & Feature Engineer
# --------------------------------------------------------------------------- #
class FeaturesRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    row_count: int | None = None
    predictor_count: int | None = None
    target_name: str | None = None


class SplitManifestRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    # Fixed policy string; Optional so an LLM final answer that nulls it still parses.
    policy: str | None = "April-June train; July validation; August test"
    train_rows: int | None = None
    validation_rows: int | None = None
    test_rows: int | None = None


class LeakageAuditRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    passed: bool = False


class FeatureEngineeringHandoff(_Strict):
    """Agent 1 — leakage-safe feature dataset handoff."""

    status: Status
    run_id: str
    summary: str
    input_sha256: str
    features: FeaturesRef
    split_manifest: SplitManifestRef
    feature_schema_path: str | None = None
    leakage_audit: LeakageAuditRef
    validation: ValidationResult
    issues: list[Issue] = []
    handoff_ready: bool
    handoff_path: str | None = None
    handoff_sha256: str | None = None
    evidence_refs: list[str] = []

    @model_validator(mode="after")
    def _pass_requires_evidence(self) -> FeatureEngineeringHandoff:
        if self.status == "PASS":
            problems = []
            if not self.handoff_ready:
                problems.append("handoff_ready must be true")
            if not self.validation.passed:
                problems.append("validation.passed must be true")
            if not self.leakage_audit.passed:
                problems.append("leakage_audit.passed must be true")
            if _has_fatal(self.issues):
                problems.append("no fatal issue may be present")
            if self.features.path is None or self.features.sha256 is None:
                problems.append("features path and sha256 must be set")
            if self.split_manifest.path is None or self.split_manifest.sha256 is None:
                problems.append("split_manifest path and sha256 must be set")
            if self.feature_schema_path is None:
                problems.append("feature_schema_path must be set")
            if self.handoff_path is None or self.handoff_sha256 is None:
                problems.append("handoff_path and handoff_sha256 must be set")
            if problems:
                raise ValueError("PASS is unsupported: " + "; ".join(problems))
        return self


# --------------------------------------------------------------------------- #
# Agent 2 — Model Trainer
# --------------------------------------------------------------------------- #
class ExperimentConfigRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    seed: int | None = None
    primary_metric: Literal["macro_f1"] = "macro_f1"


class CandidateResultsRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    candidate_count: int | None = None
    all_required_candidates_present: bool = False


class CandidateArtifactManifestRef(_Strict):
    path: str | None = None
    sha256: str | None = None


class RankingItem(_Strict):
    rank: int
    candidate_id: str
    macro_f1: float
    evidence_ref: str


class TrainingRunHandoff(_Strict):
    """Agent 2 — candidate training comparison handoff (no test access)."""

    status: Status
    run_id: str
    summary: str
    experiment_config: ExperimentConfigRef
    candidate_results: CandidateResultsRef
    candidate_artifact_manifest: CandidateArtifactManifestRef
    validation_ranking: list[RankingItem] = []
    test_accessed: bool = False
    validation: ValidationResult
    issues: list[Issue] = []
    handoff_ready: bool
    handoff_path: str | None = None
    handoff_sha256: str | None = None
    evidence_refs: list[str] = []

    @model_validator(mode="after")
    def _pass_requires_evidence(self) -> TrainingRunHandoff:
        if self.status == "PASS":
            problems = []
            if not self.handoff_ready:
                problems.append("handoff_ready must be true")
            if not self.validation.passed:
                problems.append("validation.passed must be true")
            if self.test_accessed is not False:
                problems.append("test_accessed must be false during training")
            if _has_fatal(self.issues):
                problems.append("no fatal issue may be present")
            count = self.candidate_results.candidate_count
            if count is None or count < 3:
                problems.append("candidate_count must be at least three")
            if not self.candidate_results.all_required_candidates_present:
                problems.append("all_required_candidates_present must be true")
            if self.candidate_results.path is None or self.candidate_results.sha256 is None:
                problems.append("candidate_results path and sha256 must be set")
            if self.handoff_path is None or self.handoff_sha256 is None:
                problems.append("handoff_path and handoff_sha256 must be set")
            if problems:
                raise ValueError("PASS is unsupported: " + "; ".join(problems))
        return self


# --------------------------------------------------------------------------- #
# Agent 3 — Evaluation & Governance Reviewer
# --------------------------------------------------------------------------- #
class ScientistArtifactRef(_Strict):
    name: ScientistArtifactName
    path: str
    sha256: str
    size_bytes: int


class Selection(_Strict):
    primary_metric: Literal["macro_f1"] = "macro_f1"
    selected_candidate_id: str | None = None
    validation_metric_value: float | None = None
    selection_evidence_ref: str | None = None


class TestEvaluation(_Strict):
    performed_once: bool = False
    macro_f1: float | None = None
    evidence_ref: str | None = None


class ModelMetadataRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    round_trip_validated: bool = False


class MetricsRef(_Strict):
    path: str | None = None
    sha256: str | None = None


class ScientistCrewHandoff(_Strict):
    """Agent 3 — final Scientist crew handoff."""

    status: Status
    run_id: str
    summary: str
    required_artifacts: list[ScientistArtifactRef] = []
    selection: Selection
    test_evaluation: TestEvaluation
    model_metadata: ModelMetadataRef
    metrics: MetricsRef
    validation: ValidationResult
    limitations: list[Limitation] = []
    issues: list[Issue] = []
    handoff_ready: bool
    handoff_path: str | None = None
    handoff_sha256: str | None = None
    evidence_refs: list[str] = []

    @model_validator(mode="after")
    def _pass_requires_evidence(self) -> ScientistCrewHandoff:
        if self.status == "PASS":
            problems = []
            if not self.handoff_ready:
                problems.append("handoff_ready must be true")
            if not self.validation.passed:
                problems.append("validation.passed must be true")
            if _has_fatal(self.issues):
                problems.append("no fatal issue may be present")
            names = [a.name for a in self.required_artifacts]
            if len(names) != 4 or len(set(names)) != 4:
                problems.append("required_artifacts must list exactly four unique artifacts")
            if not self.test_evaluation.performed_once:
                problems.append("test_evaluation.performed_once must be true")
            if self.metrics.path is None or self.metrics.sha256 is None:
                problems.append("metrics path and sha256 must be set")
            if self.model_metadata.path is None or self.model_metadata.sha256 is None:
                problems.append("model_metadata path and sha256 must be set")
            if self.handoff_path is None or self.handoff_sha256 is None:
                problems.append("handoff_path and handoff_sha256 must be set")
            if problems:
                raise ValueError("PASS is unsupported: " + "; ".join(problems))
        return self
