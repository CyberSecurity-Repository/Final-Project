"""Structured output models for the Analyst crew handoffs.

These Pydantic models are the validated control records the three Analyst agents
return — ``SourceQualityReview``, ``DataEngineeringHandoff``, and
``AnalystCrewHandoff``. They mirror the JSON schemas in the runtime prompt files
exactly, and each carries an ``after`` validator that enforces the shared-rules
invariant: a ``PASS`` status is only legal when the deterministic evidence
actually supports it (``handoff_ready`` true, no fatal issue, required artifacts
and hashes present). That makes a *false PASS* a hard validation error rather
than a silently accepted lie.

This module imports only Pydantic — never CrewAI — so the deterministic pipeline
can build and validate these records without any LLM dependency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

Status = Literal["PASS", "FAIL", "BLOCKED"]
Severity = Literal["fatal", "warning", "info"]
ArtifactName = Literal["clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json"]


class _Strict(BaseModel):
    """Base config: reject unknown fields so malformed payloads fail loudly."""

    model_config = ConfigDict(extra="forbid")


class Issue(_Strict):
    severity: Severity
    rule: str
    observed: str
    evidence_ref: str
    remediation: str


class CheckItem(_Strict):
    name: str
    passed: bool
    evidence_ref: str


class ValidationResult(_Strict):
    passed: bool
    # Descriptive citation. Optional so an LLM agent's final structured answer that
    # omits it still parses; the deterministic validators are the real authority
    # for `passed`, and PASS invariants never depend on this field.
    evidence_ref: str | None = None


class SourceInfo(_Strict):
    dataset_name: str | None = None
    source_url: str | None = None
    primary_documentation_url: str | None = None
    publisher: str | None = None
    license: str | None = None
    documented_time_period: str | None = None
    observed_time_period: str | None = None
    input_sha256: str | None = None


class CleanDataRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    row_count: int | None = None
    column_count: int | None = None


class ContractRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    contract_version: str | None = None


class AuditRef(_Strict):
    path: str | None = None
    sha256: str | None = None
    rule_count: int | None = None
    fatal_issue_count: int | None = None


class EdaEvidence(_Strict):
    metrics_path: str | None = None
    metrics_sha256: str | None = None
    figure_manifest_path: str | None = None
    figure_count: int | None = None


class ArtifactRef(_Strict):
    name: ArtifactName
    path: str
    sha256: str
    size_bytes: int


class Limitation(_Strict):
    statement: str
    evidence_ref: str


def _has_fatal(issues: list[Issue]) -> bool:
    return any(i.severity == "fatal" for i in issues)


class SourceQualityReview(_Strict):
    """Agent 1 — source & quality readiness decision."""

    status: Status
    run_id: str
    summary: str
    source: SourceInfo
    checks: list[CheckItem] = []
    issues: list[Issue] = []
    handoff_ready: bool
    review_path: str | None = None
    review_sha256: str | None = None
    evidence_refs: list[str] = []

    @model_validator(mode="after")
    def _pass_requires_evidence(self) -> SourceQualityReview:
        if self.status == "PASS":
            problems = []
            if not self.handoff_ready:
                problems.append("handoff_ready must be true")
            if _has_fatal(self.issues):
                problems.append("no fatal issue may be present")
            if self.review_path is None or self.review_sha256 is None:
                problems.append("review_path and review_sha256 must be set")
            if problems:
                raise ValueError("PASS is unsupported: " + "; ".join(problems))
        return self


class DataEngineeringHandoff(_Strict):
    """Agent 2 — validated cleaned-data handoff."""

    status: Status
    run_id: str
    summary: str
    input_sha256: str
    clean_data: CleanDataRef
    dataset_contract: ContractRef
    transformation_audit: AuditRef
    validation: ValidationResult
    issues: list[Issue] = []
    handoff_ready: bool
    handoff_path: str | None = None
    handoff_sha256: str | None = None
    evidence_refs: list[str] = []

    @model_validator(mode="after")
    def _pass_requires_evidence(self) -> DataEngineeringHandoff:
        if self.status == "PASS":
            problems = []
            if not self.handoff_ready:
                problems.append("handoff_ready must be true")
            if not self.validation.passed:
                problems.append("validation.passed must be true")
            if _has_fatal(self.issues):
                problems.append("no fatal issue may be present")
            if self.clean_data.path is None or self.clean_data.sha256 is None:
                problems.append("clean_data path and sha256 must be set")
            if self.dataset_contract.path is None or self.dataset_contract.sha256 is None:
                problems.append("dataset_contract path and sha256 must be set")
            if self.transformation_audit.path is None or self.transformation_audit.sha256 is None:
                problems.append("transformation_audit path and sha256 must be set")
            if self.handoff_path is None or self.handoff_sha256 is None:
                problems.append("handoff_path and handoff_sha256 must be set")
            if problems:
                raise ValueError("PASS is unsupported: " + "; ".join(problems))
        return self


class AnalystCrewHandoff(_Strict):
    """Agent 3 — final Analyst crew handoff."""

    status: Status
    run_id: str
    summary: str
    input_sha256: str
    required_artifacts: list[ArtifactRef] = []
    eda_evidence: EdaEvidence
    validation: ValidationResult
    limitations: list[Limitation] = []
    issues: list[Issue] = []
    handoff_ready: bool
    handoff_path: str | None = None
    handoff_sha256: str | None = None
    evidence_refs: list[str] = []

    @model_validator(mode="after")
    def _pass_requires_evidence(self) -> AnalystCrewHandoff:
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
            if self.handoff_path is None or self.handoff_sha256 is None:
                problems.append("handoff_path and handoff_sha256 must be set")
            if problems:
                raise ValueError("PASS is unsupported: " + "; ".join(problems))
        return self
