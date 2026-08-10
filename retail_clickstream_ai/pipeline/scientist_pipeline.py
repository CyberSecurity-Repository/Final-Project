"""The deterministic Scientist pipeline.

This module runs the entire Scientist stage as tested Python, with no LLM in the
loop: it validates the Analyst handoff, builds leakage-safe features, audits for
leakage, trains and compares the candidate models on train/validation, locks the
winner and evaluates it once on the held-out test month, renders the two reports,
validates the four required artifacts, and emits the three structured handoff
records. The CrewAI Scientist crew drives these *same* deterministic functions
through tools, so a real agent run and this offline run produce identical
artifacts — the agents interpret, they never compute.

Failure is graceful and loud: on an invalid Analyst handoff, a failed leakage
audit, or an artifact mismatch the run stops, writes a structured
``failure_report.json`` under ``artifacts/runs/``, and never emits a PASS handoff.
It never repairs Analyst output. Nothing downstream may run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retail_clickstream_ai import paths
from retail_clickstream_ai.crews.io_helpers import rel as _rel
from retail_clickstream_ai.crews.io_helpers import stamp_and_write as _persist_handoff
from retail_clickstream_ai.crews.io_helpers import write_json as _write_json
from retail_clickstream_ai.crews.scientist import models as m
from retail_clickstream_ai.pipeline import cleaning
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline import features as F
from retail_clickstream_ai.pipeline import modeling as M
from retail_clickstream_ai.reporting.model_report import (
    render_evaluation_report,
    render_model_card,
)
from retail_clickstream_ai.validation import artifacts as artifact_validation
from retail_clickstream_ai.validation.contract import DatasetContract
from retail_clickstream_ai.validation.errors import ValidationReport


class ScientistPipelineError(RuntimeError):
    """Raised when the Scientist pipeline stops on invalid input or a bad artifact."""

    def __init__(self, message: str, failure_report: Path) -> None:
        self.failure_report = failure_report
        super().__init__(f"{message} (see {failure_report})")


@dataclass
class ScientistPipelineResult:
    """Everything a caller needs to know about one deterministic Scientist run."""

    status: str
    run_id: str
    input_sha256: str
    run_dir: Path
    scientist_dir: Path
    selected_candidate_id: str | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    handoffs: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    validation: ValidationReport | None = None
    failure_report: Path | None = None

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


def _failure_report(
    run_dir: Path,
    *,
    run_id: str,
    input_sha256: str,
    stage: str,
    cause: str,
    remediation: str,
    report: ValidationReport | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "status": "FAIL",
        "run_id": run_id,
        "stage": stage,
        "input_sha256": input_sha256,
        "cause": cause,
        "remediation": remediation,
        "downstream": "blocked — the Flow must not proceed after a Scientist failure.",
    }
    if report is not None:
        payload["validation"] = report.to_dict()
    if extra is not None:
        payload["detail"] = extra
    return _write_json(run_dir / "failure_report.json", payload)


def _artifact_ref(name: m.ScientistArtifactName, path: Path) -> m.ScientistArtifactRef:
    import os

    return m.ScientistArtifactRef(
        name=name,
        path=_rel(path),
        sha256=d.sha256_file(path),
        size_bytes=int(os.path.getsize(path)),
    )


def run_scientist_pipeline(
    *,
    artifact_root: str | Path | None = None,
    run_id: str | None = None,
    pin_analyst_hash: bool = True,
    render_figures: bool = True,
) -> ScientistPipelineResult:
    """Run the full deterministic Scientist stage; return a structured result.

    ``pin_analyst_hash`` is relaxed only for a synthetic fixture whose cleaned
    data intentionally differs from the pinned production file.

    Raises
    ------
    ScientistPipelineError
        On an invalid Analyst handoff, a failed leakage audit, or an artifact
        mismatch. A ``failure_report.json`` is written first and no PASS handoff
        is produced.
    """
    if artifact_root is not None:
        import os

        os.environ["ARTIFACT_ROOT"] = str(artifact_root)

    contract = DatasetContract.build()
    analyst_dir = paths.analyst_artifacts()
    scientist_dir = paths.scientist_artifacts()
    scientist_dir.mkdir(parents=True, exist_ok=True)

    clean_path = analyst_dir / "clean_data.csv"
    input_sha = d.sha256_file(clean_path)
    rid = run_id or f"scientist-{input_sha[:12]}"
    run_dir = paths.runs_artifacts() / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    result = ScientistPipelineResult(
        status="FAIL",
        run_id=rid,
        input_sha256=input_sha,
        run_dir=run_dir,
        scientist_dir=scientist_dir,
    )

    # -- Gate 1: Analyst handoff (never repair Analyst output) ------------- #
    analyst_report = artifact_validation.validate_analyst_artifacts(
        clean_path=clean_path,
        eda_report_path=analyst_dir / "eda_report.html",
        insights_path=analyst_dir / "insights.md",
        contract_path=analyst_dir / "dataset_contract.json",
        contract=contract,
        pin_hash=pin_analyst_hash,
        require_all_months=True,
    )
    if not analyst_report.ok:
        fr = _failure_report(
            run_dir,
            run_id=rid,
            input_sha256=input_sha,
            stage="analyst_handoff",
            cause="Analyst artifacts failed validation",
            remediation="Re-run the Analyst stage; do not repair its output in Scientist code.",
            report=analyst_report,
        )
        result.failure_report = fr
        result.validation = analyst_report
        raise ScientistPipelineError("Scientist analyst-handoff gate failed", fr)

    # -- Feature engineering + leakage audit ------------------------------- #
    clean_df = cleaning.read_clean_csv(clean_path)
    features_df, schema = F.build_features(clean_df, contract)
    features_path = scientist_dir / "features.csv"
    feat_stat = F.write_features_csv(features_df, features_path)
    schema_path = _write_json(scientist_dir / "feature_schema.json", schema)
    manifest = F.build_split_manifest(features_df, contract)
    manifest_path = _write_json(scientist_dir / "split_manifest.json", manifest)
    audit = F.run_leakage_audit(clean_df, features_df, contract)
    audit_path = _write_json(scientist_dir / "leakage_audit.json", audit)
    result.artifacts["features.csv"] = features_path

    if not audit["passed"]:
        failed = [c["name"] for c in audit["checks"] if not c["passed"]]
        fr = _failure_report(
            run_dir,
            run_id=rid,
            input_sha256=input_sha,
            stage="leakage_audit",
            cause="the leakage audit did not pass",
            remediation="Fix the feature pipeline; never fall back to a random split.",
            extra={"failed_checks": failed},
        )
        result.failure_report = fr
        raise ScientistPipelineError("Scientist leakage-audit gate failed", fr)

    feature_handoff = m.FeatureEngineeringHandoff(
        status="PASS",
        run_id=rid,
        summary="Leakage-safe features built; audit passed; chronological split applied.",
        input_sha256=input_sha,
        features=m.FeaturesRef(
            path=_rel(features_path),
            sha256=feat_stat["sha256"],
            row_count=feat_stat["row_count"],
            predictor_count=schema["predictor_count"],
            target_name=F.TARGET,
        ),
        split_manifest=m.SplitManifestRef(
            path=_rel(manifest_path),
            sha256=d.sha256_file(manifest_path),
            train_rows=manifest["train_rows"],
            validation_rows=manifest["validation_rows"],
            test_rows=manifest["test_rows"],
        ),
        feature_schema_path=_rel(schema_path),
        leakage_audit=m.LeakageAuditRef(
            path=_rel(audit_path), sha256=d.sha256_file(audit_path), passed=True
        ),
        validation=m.ValidationResult(
            passed=True, evidence_ref="artifacts/scientist/leakage_audit.json#passed"
        ),
        issues=[],
        handoff_ready=True,
        handoff_path=_rel(run_dir / "feature_engineering_handoff.json"),
        handoff_sha256="pending",
        evidence_refs=[_rel(features_path), _rel(manifest_path), _rel(audit_path)],
    )

    # -- Candidate training (train/validation only; NO test access) -------- #
    config_path = M.write_experiment_config(scientist_dir / "experiment_config.json")
    results = M.run_candidate_experiments(features_df, M.experiment_config())
    results_path = _write_json(scientist_dir / "candidate_results.json", results)
    candidate_manifest_path = _write_json(
        run_dir / "candidate_artifact_manifest.json",
        {"candidates": list(results["candidates"].keys()), "results_file": _rel(results_path)},
    )
    training_check = M.validate_training_outputs(results)
    if not training_check["passed"]:
        fr = _failure_report(
            run_dir,
            run_id=rid,
            input_sha256=input_sha,
            stage="training",
            cause="candidate training outputs failed validation",
            remediation="Inspect the experiment configuration and candidate results.",
            extra={"issues": training_check["issues"]},
        )
        result.failure_report = fr
        raise ScientistPipelineError("Scientist training gate failed", fr)

    training_handoff = m.TrainingRunHandoff(
        status="PASS",
        run_id=rid,
        summary="Baseline + two model variations compared on validation; test untouched.",
        experiment_config=m.ExperimentConfigRef(
            path=_rel(config_path),
            sha256=d.sha256_file(config_path),
            seed=results["seed"],
            primary_metric="macro_f1",
        ),
        candidate_results=m.CandidateResultsRef(
            path=_rel(results_path),
            sha256=d.sha256_file(results_path),
            candidate_count=results["candidate_count"],
            all_required_candidates_present=results["all_required_candidates_present"],
        ),
        candidate_artifact_manifest=m.CandidateArtifactManifestRef(
            path=_rel(candidate_manifest_path),
            sha256=d.sha256_file(candidate_manifest_path),
        ),
        validation_ranking=[m.RankingItem(**r) for r in results["validation_ranking"]],
        test_accessed=False,
        validation=m.ValidationResult(
            passed=True, evidence_ref="tool:validate_training_outputs#passed"
        ),
        issues=[],
        handoff_ready=True,
        handoff_path=_rel(run_dir / "training_handoff.json"),
        handoff_sha256="pending",
        evidence_refs=[_rel(results_path), _rel(config_path)],
    )

    # -- Lock winner + one-time held-out test evaluation ------------------- #
    bundle = M.lock_winner_and_evaluate_test(
        features_df,
        scientist_dir,
        candidate_results=results,
        training_data_sha256=input_sha,
        features_sha256=feat_stat["sha256"],
        render_figures=render_figures,
    )
    result.selected_candidate_id = bundle.selected_candidate_id
    result.metrics = bundle.metrics
    result.artifacts["model.joblib"] = bundle.model_path

    # -- Reports ----------------------------------------------------------- #
    eval_md, _ = render_evaluation_report(bundle.metrics)
    eval_path = scientist_dir / "evaluation_report.md"
    eval_path.write_text(eval_md, encoding="utf-8")
    card_md, _ = render_model_card(bundle.metrics, bundle.metadata)
    card_path = scientist_dir / "model_card.md"
    card_path.write_text(card_md, encoding="utf-8")
    result.artifacts["evaluation_report.md"] = eval_path
    result.artifacts["model_card.md"] = card_path

    # -- Final artifact gate ----------------------------------------------- #
    final_report = artifact_validation.validate_scientist_artifacts(
        features_path=features_path,
        model_path=bundle.model_path,
        eval_report_path=eval_path,
        model_card_path=card_path,
        metrics_path=bundle.metrics_path,
        metadata_path=bundle.metadata_path,
    )
    result.validation = final_report
    if not final_report.ok:
        fr = _failure_report(
            run_dir,
            run_id=rid,
            input_sha256=input_sha,
            stage="artifact_validation",
            cause="one or more required Scientist artifacts failed validation",
            remediation="Inspect the failing artifact; do not proceed downstream.",
            report=final_report,
        )
        result.failure_report = fr
        raise ScientistPipelineError("Scientist artifact gate failed", fr)

    scientist_handoff = m.ScientistCrewHandoff(
        status="PASS",
        run_id=rid,
        summary=(
            f"Scientist crew complete: winner '{bundle.selected_candidate_id}' selected on "
            "validation macro F1 and evaluated once on the held-out test month."
        ),
        required_artifacts=[
            _artifact_ref("features.csv", features_path),
            _artifact_ref("model.joblib", bundle.model_path),
            _artifact_ref("evaluation_report.md", eval_path),
            _artifact_ref("model_card.md", card_path),
        ],
        selection=m.Selection(
            primary_metric="macro_f1",
            selected_candidate_id=bundle.selected_candidate_id,
            validation_metric_value=bundle.metrics["validation"]["macro_f1"],
            selection_evidence_ref="artifacts/scientist/metrics.json#validation.macro_f1",
        ),
        test_evaluation=m.TestEvaluation(
            performed_once=True,
            macro_f1=bundle.metrics["test"]["macro_f1"],
            evidence_ref="artifacts/scientist/metrics.json#test.macro_f1",
        ),
        model_metadata=m.ModelMetadataRef(
            path=_rel(bundle.metadata_path),
            sha256=d.sha256_file(bundle.metadata_path),
            round_trip_validated=bundle.round_trip_validated,
        ),
        metrics=m.MetricsRef(
            path=_rel(bundle.metrics_path), sha256=d.sha256_file(bundle.metrics_path)
        ),
        validation=m.ValidationResult(
            passed=True, evidence_ref="tool:validate_scientist_artifacts#passed"
        ),
        limitations=[
            m.Limitation(
                statement="Data is from 2008; not current-market evidence.",
                evidence_ref="artifacts/scientist/model_card.md#limitations",
            ),
            m.Limitation(
                statement=(
                    "A strong same-category baseline is hard to beat; no significance claimed."
                ),
                evidence_ref="artifacts/scientist/metrics.json#validation_ranking",
            ),
            m.Limitation(
                statement=(
                    "model.joblib is pickle-based; load only the trusted hash-verified artifact."
                ),
                evidence_ref="artifacts/scientist/model_metadata.json#security_note",
            ),
        ],
        issues=[],
        handoff_ready=True,
        handoff_path=_rel(run_dir / "scientist_crew_handoff.json"),
        handoff_sha256="pending",
        evidence_refs=[_rel(p) for p in result.artifacts.values()],
    )

    # -- Persist the three handoff records --------------------------------- #
    fe, _ = _persist_handoff(
        feature_handoff,
        run_dir / "feature_engineering_handoff.json",
        "handoff_path",
        "handoff_sha256",
    )
    tr, _ = _persist_handoff(
        training_handoff, run_dir / "training_handoff.json", "handoff_path", "handoff_sha256"
    )
    sc, _ = _persist_handoff(
        scientist_handoff, run_dir / "scientist_crew_handoff.json", "handoff_path", "handoff_sha256"
    )
    result.handoffs = {
        "feature_engineering_handoff": run_dir / "feature_engineering_handoff.json",
        "training_handoff": run_dir / "training_handoff.json",
        "scientist_crew_handoff": run_dir / "scientist_crew_handoff.json",
    }
    result.status = "PASS"
    return result


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Run the deterministic Scientist pipeline from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m retail_clickstream_ai.pipeline.scientist_pipeline",
        description="Run the deterministic Scientist pipeline (no LLM).",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id.")
    args = parser.parse_args(argv)

    try:
        result = run_scientist_pipeline(run_id=args.run_id)
    except ScientistPipelineError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS: run {result.run_id} — winner {result.selected_candidate_id}")
    for name, path in result.artifacts.items():
        print(f"  {name}: {_rel(path)}")
    print(f"  test macro_f1: {result.metrics['test']['macro_f1']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
