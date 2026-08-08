"""The deterministic Analyst pipeline.

This module runs the entire Analyst stage as tested Python, with no LLM in the
loop: it validates the raw input, cleans it, writes the four required artifacts,
computes the EDA metrics/figures, renders the reports, validates everything, and
emits the three structured handoff records. The CrewAI Analyst crew
(:mod:`retail_clickstream_ai.crews.analyst`) drives these *same* deterministic
functions through tools, so a real agent run and this offline run produce
identical artifacts — the agents interpret, they never compute.

Failure is graceful and loud: on invalid raw data or an artifact mismatch the run
stops, writes a structured ``failure_report.json`` under ``artifacts/runs/`` with
cause and remediation, and never emits a PASS handoff. Nothing downstream may run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retail_clickstream_ai import paths
from retail_clickstream_ai.crews.analyst import models as m
from retail_clickstream_ai.crews.io_helpers import (
    rel as _rel,
)
from retail_clickstream_ai.crews.io_helpers import (
    stamp_and_write as _persist_handoff,
)
from retail_clickstream_ai.crews.io_helpers import (
    write_json as _write_json,
)
from retail_clickstream_ai.pipeline import cleaning, eda
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.reporting.eda_report import render_eda_report
from retail_clickstream_ai.reporting.insights import render_insights
from retail_clickstream_ai.validation import artifacts as artifact_validation
from retail_clickstream_ai.validation.contract import DatasetContract
from retail_clickstream_ai.validation.errors import ContractValidationError, ValidationReport


class AnalystPipelineError(RuntimeError):
    """Raised when the Analyst pipeline stops on invalid data or a bad artifact.

    Carries the path of the structured failure report so callers can surface it.
    """

    def __init__(self, message: str, failure_report: Path) -> None:
        self.failure_report = failure_report
        super().__init__(f"{message} (see {failure_report})")


@dataclass
class AnalystPipelineResult:
    """Everything a caller needs to know about one deterministic Analyst run."""

    status: str
    run_id: str
    input_sha256: str
    run_dir: Path
    analyst_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)
    handoffs: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    figure_count: int = 0
    validation: ValidationReport | None = None
    failure_report: Path | None = None

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


# --------------------------------------------------------------------------- #
# Run preparation (the inputs the Source & Quality agent reads)
# --------------------------------------------------------------------------- #
@dataclass
class PreparedRun:
    input_path: Path
    input_sha256: str
    sha_match: bool
    raw_shape: tuple[int, int]
    report: ValidationReport
    profile: dict[str, Any]
    source_metadata_path: Path
    raw_validation_report_path: Path
    raw_profile_path: Path


def _source_metadata(contract: DatasetContract) -> dict[str, Any]:
    ds = contract.data["dataset"]
    return {
        "dataset_name": ds["name"],
        "source_url": ds["source_urls"]["uci"],
        "primary_documentation_url": ds["source_urls"]["uci"],
        "kaggle_url": ds["source_urls"]["kaggle"],
        "publisher": ds["publisher"],
        "license": ds["license"],
        "documented_time_period": ds["documented_time_period"],
        "data_description_file": ds["data_description_file"],
        "acquisition_method": "Kaggle CLI (slug " + ds["kaggle_slug"] + "); not committed.",
        "evidence_status": "documented (publisher) — confirm against observed validation",
        "expected_input_sha256": contract.raw_sha256,
    }


def prepare_run(
    input_path: str | Path,
    contract: DatasetContract,
    run_dir: Path,
    *,
    require_all_months: bool = True,
) -> PreparedRun:
    """Validate + profile the raw input and write the three agent-input files."""
    from retail_clickstream_ai.validation.raw import validate_dataframe

    path = Path(input_path)
    input_sha = d.sha256_file(path)
    raw = d.read_raw_csv(path)
    report = validate_dataframe(raw, contract, require_all_months=require_all_months)
    profile = d.profile_dataframe(raw)
    sha_match = input_sha == contract.raw_sha256

    source_meta = _source_metadata(contract)
    source_meta["observed_input_sha256"] = input_sha
    source_meta["input_sha256_matches_contract"] = sha_match

    smp = _write_json(run_dir / "source_metadata.json", source_meta)
    rvp = _write_json(
        run_dir / "raw_validation_report.json",
        {
            "input_sha256": input_sha,
            "expected_sha256": contract.raw_sha256,
            "sha256_match": sha_match,
            "row_count": int(raw.shape[0]),
            "column_count": int(raw.shape[1]),
            "passed": report.ok and sha_match,
            "report": report.to_dict(),
        },
    )
    rpp = _write_json(run_dir / "raw_profile.json", profile)

    return PreparedRun(
        input_path=path,
        input_sha256=input_sha,
        sha_match=sha_match,
        raw_shape=(int(raw.shape[0]), int(raw.shape[1])),
        report=report,
        profile=profile,
        source_metadata_path=smp,
        raw_validation_report_path=rvp,
        raw_profile_path=rpp,
    )


# --------------------------------------------------------------------------- #
# EDA table intermediates (traceability)
# --------------------------------------------------------------------------- #
def _write_eda_tables(metrics: dict[str, Any], tables_dir: Path) -> list[Path]:
    return eda.write_tables(metrics, tables_dir)


# --------------------------------------------------------------------------- #
# Handoff-record builders (deterministic, from computed evidence)
# --------------------------------------------------------------------------- #
def _issues_from_report(report: ValidationReport) -> list[m.Issue]:
    return [
        m.Issue(
            severity=i.severity.value,
            rule=i.rule,
            observed=str(i.observed),
            evidence_ref="artifacts/runs#raw_validation_report.json",
            remediation="Fix the raw input to satisfy the dataset contract, then rerun.",
        )
        for i in report.issues
    ]


def _failure_report(
    run_dir: Path,
    *,
    run_id: str,
    input_sha256: str,
    stage: str,
    cause: str,
    report: ValidationReport | None,
    remediation: str,
) -> Path:
    payload: dict[str, Any] = {
        "status": "FAIL",
        "run_id": run_id,
        "stage": stage,
        "input_sha256": input_sha256,
        "cause": cause,
        "remediation": remediation,
        "downstream": "blocked — the Scientist crew must not run after an Analyst failure.",
    }
    if report is not None:
        payload["validation"] = report.to_dict()
    return _write_json(run_dir / "failure_report.json", payload)


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def run_analyst_pipeline(
    input_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    run_id: str | None = None,
    pin_contract_hash: bool = True,
    require_all_months: bool = True,
    render_figures: bool = True,
) -> AnalystPipelineResult:
    """Run the full deterministic Analyst stage; return a structured result.

    Parameters mirror the safety knobs the tests need: ``pin_contract_hash``
    (and ``require_all_months``) are relaxed only for the synthetic fixture,
    which intentionally differs from the pinned production data.

    Raises
    ------
    AnalystPipelineError
        On invalid raw data or an artifact mismatch. A ``failure_report.json``
        is written first and no PASS handoff is produced.
    """
    if artifact_root is not None:
        import os

        os.environ["ARTIFACT_ROOT"] = str(artifact_root)

    contract = DatasetContract.build()
    analyst_dir = paths.analyst_artifacts()
    eda_dir = analyst_dir / "eda"
    analyst_dir.mkdir(parents=True, exist_ok=True)

    input_sha_probe = d.sha256_file(Path(input_path))
    rid = run_id or f"analyst-{input_sha_probe[:12]}"
    run_dir = paths.runs_artifacts() / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    result = AnalystPipelineResult(
        status="FAIL",
        run_id=rid,
        input_sha256=input_sha_probe,
        run_dir=run_dir,
        analyst_dir=analyst_dir,
    )

    # -- Prepare + source/quality gate ------------------------------------ #
    prepared = prepare_run(input_path, contract, run_dir, require_all_months=require_all_months)
    result.input_sha256 = prepared.input_sha256
    raw_valid = prepared.report.ok and (prepared.sha_match or not pin_contract_hash)
    if not raw_valid:
        cause = "raw validation failed" if not prepared.report.ok else "input SHA-256 mismatch"
        fr = _failure_report(
            run_dir,
            run_id=rid,
            input_sha256=prepared.input_sha256,
            stage="source_quality",
            cause=cause,
            report=prepared.report,
            remediation="Re-acquire the exact source file (see data/README.md) and rerun.",
        )
        result.failure_report = fr
        result.validation = prepared.report
        raise AnalystPipelineError("Analyst source/quality gate failed", fr)

    observed_period = "2008 months " + ",".join(
        str(m_) for m_ in sorted(prepared.profile.get("month_distribution", {}))
    )
    source_review = m.SourceQualityReview(
        status="PASS",
        run_id=rid,
        summary="Source authentic and structurally usable; raw validation passed.",
        source=m.SourceInfo(
            dataset_name=contract.data["dataset"]["name"],
            source_url=contract.data["dataset"]["source_urls"]["uci"],
            primary_documentation_url=contract.data["dataset"]["source_urls"]["uci"],
            publisher=contract.data["dataset"]["publisher"],
            license=contract.data["dataset"]["license"],
            documented_time_period=contract.data["dataset"]["documented_time_period"],
            observed_time_period=observed_period,
            input_sha256=prepared.input_sha256,
        ),
        checks=[
            m.CheckItem(
                name="input_sha256_match",
                passed=prepared.sha_match,
                evidence_ref="artifacts/runs#raw_validation_report.json",
            ),
            m.CheckItem(
                name="schema_and_ranges",
                passed=prepared.report.ok,
                evidence_ref="artifacts/runs#raw_validation_report.json",
            ),
            m.CheckItem(
                name="session_click_order",
                passed=prepared.report.ok,
                evidence_ref="artifacts/runs#raw_validation_report.json",
            ),
        ],
        issues=_issues_from_report(prepared.report),
        handoff_ready=True,
        review_path=_rel(run_dir / "source_quality_review.json"),
        review_sha256="pending",
        evidence_refs=[_rel(prepared.raw_validation_report_path)],
    )

    # -- Data engineering: clean + contract + audit ----------------------- #
    raw = d.read_raw_csv(prepared.input_path)
    try:
        clean_df, audit = cleaning.clean_dataframe(
            raw, contract, require_all_months=require_all_months
        )
    except ContractValidationError as exc:
        fr = _failure_report(
            run_dir,
            run_id=rid,
            input_sha256=prepared.input_sha256,
            stage="cleaning",
            cause="cleaning refused invalid data (fail-closed)",
            report=exc.report,
            remediation="Correct the flagged columns/rows in the source, then rerun.",
        )
        result.failure_report = fr
        result.validation = exc.report
        raise AnalystPipelineError("Analyst cleaning gate failed", fr) from exc

    clean_path = analyst_dir / "clean_data.csv"
    clean_stat = cleaning.write_clean_csv(clean_df, clean_path)
    audit_path = _write_json(run_dir / "transformation_audit.json", audit.to_dict())
    contract_path = contract.write(analyst_dir / "dataset_contract.json")

    result.artifacts["clean_data.csv"] = clean_path
    result.artifacts["dataset_contract.json"] = contract_path

    # Gate: the produced clean file must match the contract.
    clean_report = artifact_validation.validate_clean_file_against_contract(
        clean_path, contract, pin_hash=pin_contract_hash, require_all_months=require_all_months
    )
    if not clean_report.ok:
        fr = _failure_report(
            run_dir,
            run_id=rid,
            input_sha256=prepared.input_sha256,
            stage="clean_contract_match",
            cause="cleaned data does not match the dataset contract",
            report=clean_report,
            remediation="Investigate the cleaning pipeline; do not hand off to modeling.",
        )
        result.failure_report = fr
        result.validation = clean_report
        raise AnalystPipelineError("Analyst clean/contract gate failed", fr)

    de_handoff = m.DataEngineeringHandoff(
        status="PASS",
        run_id=rid,
        summary="Cleaned data produced and validated against the contract.",
        input_sha256=prepared.input_sha256,
        clean_data=m.CleanDataRef(
            path=_rel(clean_path),
            sha256=clean_stat["sha256"],
            row_count=clean_stat["row_count"],
            column_count=clean_stat["column_count"],
        ),
        dataset_contract=m.ContractRef(
            path=_rel(contract_path),
            sha256=d.sha256_file(contract_path),
            contract_version=contract.data["contract_version"],
        ),
        transformation_audit=m.AuditRef(
            path=_rel(audit_path),
            sha256=d.sha256_file(audit_path),
            rule_count=len(audit.rules),
            fatal_issue_count=audit.fatal_issue_count,
        ),
        validation=m.ValidationResult(
            passed=True, evidence_ref="tool:validate_cleaned_handoff#passed"
        ),
        issues=[],
        handoff_ready=True,
        handoff_path=_rel(run_dir / "data_engineering_handoff.json"),
        handoff_sha256="pending",
        evidence_refs=[_rel(clean_path), _rel(contract_path), _rel(audit_path)],
    )

    # -- EDA + reports ----------------------------------------------------- #
    metrics = eda.compute_metrics(
        clean_df,
        input_sha256=prepared.input_sha256,
        clean_sha256=clean_stat["sha256"],
        contract_version=contract.data["contract_version"],
    )
    metrics_path = _write_json(eda_dir / "eda_metrics.json", metrics)
    _write_eda_tables(metrics, eda_dir / "tables")

    figures = eda.make_figures(clean_df, metrics) if render_figures else []
    manifest = eda.figure_manifest(figures)
    manifest_path = _write_json(eda_dir / "figure_manifest.json", manifest)

    insights_md, _ = render_insights(metrics)
    insights_path = analyst_dir / "insights.md"
    insights_path.write_text(insights_md, encoding="utf-8")

    report_html, _ = render_eda_report(metrics, figures, contract=contract)
    report_path = analyst_dir / "eda_report.html"
    report_path.write_text(report_html, encoding="utf-8")

    result.artifacts["insights.md"] = insights_path
    result.artifacts["eda_report.html"] = report_path
    result.metrics = metrics
    result.figure_count = len(figures)

    # -- Final artifact gate ---------------------------------------------- #
    final_report = artifact_validation.validate_analyst_artifacts(
        clean_path=clean_path,
        eda_report_path=report_path,
        insights_path=insights_path,
        contract_path=contract_path,
        contract=contract,
        pin_hash=pin_contract_hash,
        require_all_months=require_all_months,
    )
    result.validation = final_report
    if not final_report.ok:
        fr = _failure_report(
            run_dir,
            run_id=rid,
            input_sha256=prepared.input_sha256,
            stage="artifact_validation",
            cause="one or more required Analyst artifacts failed validation",
            report=final_report,
            remediation="Inspect the failing artifact; do not proceed to modeling.",
        )
        result.failure_report = fr
        raise AnalystPipelineError("Analyst artifact gate failed", fr)

    def _artifact_ref(name: m.ArtifactName, path: Path) -> m.ArtifactRef:
        import os

        return m.ArtifactRef(
            name=name,
            path=_rel(path),
            sha256=d.sha256_file(path),
            size_bytes=int(os.path.getsize(path)),
        )

    analyst_handoff = m.AnalystCrewHandoff(
        status="PASS",
        run_id=rid,
        summary="Analyst crew complete: four artifacts produced and validated.",
        input_sha256=prepared.input_sha256,
        required_artifacts=[
            _artifact_ref("clean_data.csv", clean_path),
            _artifact_ref("eda_report.html", report_path),
            _artifact_ref("insights.md", insights_path),
            _artifact_ref("dataset_contract.json", contract_path),
        ],
        eda_evidence=m.EdaEvidence(
            metrics_path=_rel(metrics_path),
            metrics_sha256=d.sha256_file(metrics_path),
            figure_manifest_path=_rel(manifest_path),
            figure_count=len(figures),
        ),
        validation=m.ValidationResult(
            passed=True, evidence_ref="tool:validate_analyst_artifacts#passed"
        ),
        limitations=[
            m.Limitation(
                statement="Data is from 2008; not current-market evidence.",
                evidence_ref=_rel(metrics_path) + "#provenance.time_period",
            ),
            m.Limitation(
                statement="EDA is descriptive, not causal.",
                evidence_ref=_rel(insights_path) + "#limitations",
            ),
            m.Limitation(
                statement="One shop, one dominant country.",
                evidence_ref=_rel(metrics_path) + "#country.top",
            ),
        ],
        issues=[],
        handoff_ready=True,
        handoff_path=_rel(run_dir / "analyst_crew_handoff.json"),
        handoff_sha256="pending",
        evidence_refs=[_rel(p) for p in result.artifacts.values()],
    )

    # -- Persist the three handoff records -------------------------------- #
    sq, _ = _persist_handoff(
        source_review, run_dir / "source_quality_review.json", "review_path", "review_sha256"
    )
    de, _ = _persist_handoff(
        de_handoff, run_dir / "data_engineering_handoff.json", "handoff_path", "handoff_sha256"
    )
    an, _ = _persist_handoff(
        analyst_handoff, run_dir / "analyst_crew_handoff.json", "handoff_path", "handoff_sha256"
    )
    result.handoffs = {
        "source_quality_review": run_dir / "source_quality_review.json",
        "data_engineering_handoff": run_dir / "data_engineering_handoff.json",
        "analyst_crew_handoff": run_dir / "analyst_crew_handoff.json",
    }
    result.status = "PASS"
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Run the deterministic Analyst pipeline from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m retail_clickstream_ai.pipeline.analyst_pipeline",
        description="Run the deterministic Analyst pipeline (no LLM).",
    )
    parser.add_argument("--input", required=True, help="Path to the raw CSV.")
    parser.add_argument("--run-id", default=None, help="Optional run id.")
    args = parser.parse_args(argv)

    try:
        result = run_analyst_pipeline(args.input, run_id=args.run_id)
    except AnalystPipelineError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS: run {result.run_id}")
    for name, path in result.artifacts.items():
        print(f"  {name}: {_rel(path)}")
    print(f"  figures: {result.figure_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
