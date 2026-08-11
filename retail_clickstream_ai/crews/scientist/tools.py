"""CrewAI tools for the Scientist crew.

Every tool is a thin, typed wrapper around a *tested* deterministic function —
agents never fit a model, compute a metric, or edit a binary. Tools return
compact JSON strings (never raw rows). The four ``write_*`` tools validate the
agent-supplied record through its Pydantic model and stamp it with a content
hash before persisting, so an agent response cannot become an unvalidated
artifact. The one-time held-out test evaluation lives entirely inside
``lock_winner_and_evaluate_test`` — the training tools never touch the test split.

All tools are bound to a :class:`ScientistRunContext` at construction, so the run
id, trusted paths, and input hash are fixed for the run.
"""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict

from retail_clickstream_ai.crews.context import ScientistRunContext
from retail_clickstream_ai.crews.io_helpers import rel, stamp_and_write, write_json
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


def _compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _WriteRecordArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Optional: the tool builds the machine-truth record from the trusted on-disk
    # artifacts; the agent need only supply an interpretive ``summary`` (any other
    # keys are ignored). This makes the handoff robust to any/empty LLM output and
    # makes a false PASS or a malformed record impossible.
    record: dict[str, Any] | None = None


def _opt_summary(record: dict[str, Any] | None, default: str) -> str:
    """Take a non-empty ``summary`` from the agent's record, else use ``default``."""
    if isinstance(record, dict):
        text = record.get("summary")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return default


class _ContextTool(BaseTool):
    """Base for tools bound to a Scientist run context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    context: ScientistRunContext


# --------------------------------------------------------------------------- #
# Agent 1 — Contract & Feature Engineer
# --------------------------------------------------------------------------- #
class ValidateAnalystHandoffTool(_ContextTool):
    name: str = "validate_analyst_handoff"
    description: str = (
        "Verify the Analyst artifacts, hashes, contract-to-data agreement, and any "
        "Analyst crew handoff status. Never repair Analyst output."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        contract = DatasetContract.build()
        report = artifact_validation.validate_analyst_artifacts(
            clean_path=self.context.clean_data_path,
            eda_report_path=self.context.analyst_dir / "eda_report.html",
            insights_path=self.context.analyst_dir / "insights.md",
            contract_path=self.context.dataset_contract_path,
            contract=contract,
            pin_hash=self.context.pin_hash,
            require_all_months=True,
        )
        handoff_status: str | None = None
        handoff = self.context.analyst_handoff_path
        if handoff is not None and handoff.exists():
            handoff_status = json.loads(handoff.read_text(encoding="utf-8")).get("status")
        passed = report.ok and (handoff_status in (None, "PASS"))
        return _compact(
            {
                "passed": passed,
                "analyst_handoff_status": handoff_status,
                "input_sha256": self.context.input_sha256,
                "evidence_ref": "tool:validate_analyst_handoff#passed",
                "issues": [i.to_dict() for i in report.issues],
            }
        )


class RunFeaturePipelineTool(_ContextTool):
    name: str = "run_feature_pipeline"
    description: str = (
        "Deterministically build the leakage-safe next-main-category features, split "
        "labels, feature schema, and split manifest from the cleaned data."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        contract = DatasetContract.build()
        clean = cleaning.read_clean_csv(self.context.clean_data_path)
        features_df, schema = F.build_features(clean, contract)
        features_path = self.context.scientist_dir / "features.csv"
        stat = F.write_features_csv(features_df, features_path)
        schema_path = write_json(self.context.scientist_dir / "feature_schema.json", schema)
        manifest = F.build_split_manifest(features_df, contract)
        manifest_path = write_json(self.context.scientist_dir / "split_manifest.json", manifest)
        return _compact(
            {
                "features": {
                    "path": rel(features_path),
                    "sha256": stat["sha256"],
                    "row_count": stat["row_count"],
                    "predictor_count": schema["predictor_count"],
                    "target_name": F.TARGET,
                },
                "split_manifest": {
                    "path": rel(manifest_path),
                    "sha256": d.sha256_file(manifest_path),
                    "policy": manifest["policy"],
                    "train_rows": manifest["train_rows"],
                    "validation_rows": manifest["validation_rows"],
                    "test_rows": manifest["test_rows"],
                },
                "feature_schema_path": rel(schema_path),
            }
        )


class RunLeakageAuditTool(_ContextTool):
    name: str = "run_leakage_audit"
    description: str = (
        "Run the deterministic leakage audit: session boundaries, target shift, "
        "last-click removal, forbidden columns, split boundaries, class coverage."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        contract = DatasetContract.build()
        clean = cleaning.read_clean_csv(self.context.clean_data_path)
        features_df = F.read_features_csv(self.context.scientist_dir / "features.csv")
        audit = F.run_leakage_audit(clean, features_df, contract)
        audit_path = write_json(self.context.scientist_dir / "leakage_audit.json", audit)
        return _compact(
            {
                "path": rel(audit_path),
                "sha256": d.sha256_file(audit_path),
                "passed": audit["passed"],
                "checks": [{"name": c["name"], "passed": c["passed"]} for c in audit["checks"]],
                "evidence_ref": "artifacts/scientist/leakage_audit.json#passed",
            }
        )


class ValidateFeatureHandoffTool(_ContextTool):
    name: str = "validate_feature_handoff"
    description: str = (
        "Validate features.csv schema, split manifest, class coverage, and that the "
        "leakage audit passed."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        contract = DatasetContract.build()
        issues: list[str] = []
        features_path = self.context.scientist_dir / "features.csv"
        if not features_path.exists():
            issues.append("features.csv is missing")
        else:
            features_df = F.read_features_csv(features_path)
            cols = set(features_df.columns)
            need = {*F.PREDICTOR_COLUMNS, F.SPLIT_COLUMN, F.TARGET}
            if not need <= cols:
                issues.append(f"features missing columns: {sorted(need - cols)}")
            manifest = F.build_split_manifest(features_df, contract)
            if not manifest["all_classes_present_each_split"]:
                issues.append("not all target classes are present in every split")
        audit_path = self.context.scientist_dir / "leakage_audit.json"
        if not audit_path.exists():
            issues.append("leakage_audit.json is missing")
        elif not json.loads(audit_path.read_text(encoding="utf-8")).get("passed"):
            issues.append("leakage audit did not pass")
        return _compact(
            {
                "passed": not issues,
                "evidence_ref": "tool:validate_feature_handoff#passed",
                "issues": issues,
            }
        )


class WriteFeatureEngineeringHandoffTool(_ContextTool):
    name: str = "write_feature_engineering_handoff"
    description: str = (
        "Persist the feature-engineering handoff. The tool builds the machine record "
        "(paths, hashes, counts, pass/fail) from the trusted on-disk artifacts; you may "
        "supply an interpretive `summary` and nothing else."
    )
    args_schema: type[BaseModel] = _WriteRecordArgs

    def _run(self, record: dict[str, Any] | None = None) -> str:
        sci = self.context.scientist_dir
        features_path = sci / "features.csv"
        schema_path = sci / "feature_schema.json"
        manifest_path = sci / "split_manifest.json"
        audit_path = sci / "leakage_audit.json"

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        passed = bool(audit.get("passed"))
        row_count = int(
            manifest.get("train_rows", 0)
            + manifest.get("validation_rows", 0)
            + manifest.get("test_rows", 0)
        )

        handoff = m.FeatureEngineeringHandoff(
            status="PASS" if passed else "BLOCKED",
            run_id=self.context.run_id,
            summary=_opt_summary(
                record, "Leakage-safe features built; leakage audit checked; split applied."
            ),
            input_sha256=self.context.input_sha256,
            features=m.FeaturesRef(
                path=rel(features_path),
                sha256=d.sha256_file(features_path),
                row_count=row_count,
                predictor_count=schema.get("predictor_count"),
                target_name=F.TARGET,
            ),
            split_manifest=m.SplitManifestRef(
                path=rel(manifest_path),
                sha256=d.sha256_file(manifest_path),
                policy=manifest.get("policy"),
                train_rows=manifest.get("train_rows"),
                validation_rows=manifest.get("validation_rows"),
                test_rows=manifest.get("test_rows"),
            ),
            feature_schema_path=rel(schema_path),
            leakage_audit=m.LeakageAuditRef(
                path=rel(audit_path), sha256=d.sha256_file(audit_path), passed=passed
            ),
            validation=m.ValidationResult(
                passed=passed, evidence_ref="artifacts/scientist/leakage_audit.json#passed"
            ),
            issues=[],
            handoff_ready=passed,
            handoff_path=rel(self.context.run_dir / "feature_engineering_handoff.json"),
            handoff_sha256="pending",
            evidence_refs=[rel(features_path), rel(manifest_path), rel(audit_path)],
        )
        stamped, sha = stamp_and_write(
            handoff,
            self.context.run_dir / "feature_engineering_handoff.json",
            "handoff_path",
            "handoff_sha256",
        )
        return _compact(
            {"status": stamped.status, "handoff_path": stamped.handoff_path, "handoff_sha256": sha}
        )


# --------------------------------------------------------------------------- #
# Agent 2 — Model Trainer
# --------------------------------------------------------------------------- #
class ReadExperimentConfigTool(_ContextTool):
    name: str = "read_experiment_config"
    description: str = (
        "Write and return the immutable experiment configuration: candidates, seed, "
        "metrics, tuning grid, and selection rule."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        path = M.write_experiment_config(self.context.experiment_config_path)
        cfg = M.experiment_config()
        return _compact(
            {
                "path": rel(path),
                "sha256": d.sha256_file(path),
                "seed": cfg["seed"],
                "primary_metric": cfg["primary_metric"],
                "candidate_families": [c["family"] for c in cfg["candidates"]],
                "tie_break": cfg["tie_break"],
                "selection_rule": cfg["selection_rule"],
            }
        )


class RunCandidateExperimentsTool(_ContextTool):
    name: str = "run_candidate_experiments"
    description: str = (
        "Fit each candidate on training rows and score on validation rows. Test labels "
        "are inaccessible to this tool."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        features_df = F.read_features_csv(self.context.scientist_dir / "features.csv")
        results = M.run_candidate_experiments(features_df)
        results_path = write_json(self.context.scientist_dir / "candidate_results.json", results)
        manifest = {
            "candidates": list(results["candidates"].keys()),
            "results_file": rel(results_path),
            "test_accessed": results["test_accessed"],
        }
        manifest_path = write_json(
            self.context.run_dir / "candidate_artifact_manifest.json", manifest
        )
        return _compact(
            {
                "candidate_results": {
                    "path": rel(results_path),
                    "sha256": d.sha256_file(results_path),
                    "candidate_count": results["candidate_count"],
                    "all_required_candidates_present": results["all_required_candidates_present"],
                },
                "candidate_artifact_manifest": {
                    "path": rel(manifest_path),
                    "sha256": d.sha256_file(manifest_path),
                },
                "validation_ranking": results["validation_ranking"],
                "test_accessed": results["test_accessed"],
            }
        )


class ValidateTrainingOutputsTool(_ContextTool):
    name: str = "validate_training_outputs"
    description: str = (
        "Check candidate completeness, metric schema, reproducibility metadata, and "
        "that the test set was not accessed."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        results = json.loads(
            (self.context.scientist_dir / "candidate_results.json").read_text(encoding="utf-8")
        )
        v = M.validate_training_outputs(results)
        return _compact(
            {
                "passed": v["passed"],
                "test_accessed": v["test_accessed"],
                "candidate_count": v["candidate_count"],
                "all_required_candidates_present": v["all_required_candidates_present"],
                "issues": v["issues"],
                "evidence_ref": "tool:validate_training_outputs#passed",
            }
        )


class WriteTrainingHandoffTool(_ContextTool):
    name: str = "write_training_handoff"
    description: str = (
        "Persist the training-comparison handoff. The tool builds the machine record "
        "(candidate results, ranking, hashes, pass/fail) from the trusted on-disk "
        "artifacts; you may supply an interpretive `summary` and nothing else."
    )
    args_schema: type[BaseModel] = _WriteRecordArgs

    def _run(self, record: dict[str, Any] | None = None) -> str:
        sci = self.context.scientist_dir
        config_path = M.write_experiment_config(self.context.experiment_config_path)
        cfg = M.experiment_config()
        results_path = sci / "candidate_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        check = M.validate_training_outputs(results)
        passed = bool(check["passed"]) and check["test_accessed"] is False

        manifest_path = self.context.run_dir / "candidate_artifact_manifest.json"
        if not manifest_path.exists():
            manifest_path = write_json(
                manifest_path,
                {
                    "candidates": list(results.get("candidates", {}).keys()),
                    "results_file": rel(results_path),
                    "test_accessed": results.get("test_accessed", False),
                },
            )

        handoff = m.TrainingRunHandoff(
            status="PASS" if passed else "BLOCKED",
            run_id=self.context.run_id,
            summary=_opt_summary(
                record, "Baseline + two models compared on validation; test untouched."
            ),
            experiment_config=m.ExperimentConfigRef(
                path=rel(config_path),
                sha256=d.sha256_file(config_path),
                seed=cfg["seed"],
                primary_metric=cfg["primary_metric"],
            ),
            candidate_results=m.CandidateResultsRef(
                path=rel(results_path),
                sha256=d.sha256_file(results_path),
                candidate_count=results.get("candidate_count"),
                all_required_candidates_present=bool(
                    results.get("all_required_candidates_present")
                ),
            ),
            candidate_artifact_manifest=m.CandidateArtifactManifestRef(
                path=rel(manifest_path), sha256=d.sha256_file(manifest_path)
            ),
            validation_ranking=[m.RankingItem(**r) for r in results.get("validation_ranking", [])],
            test_accessed=False,
            validation=m.ValidationResult(
                passed=passed, evidence_ref="tool:validate_training_outputs#passed"
            ),
            issues=[],
            handoff_ready=passed,
            handoff_path=rel(self.context.run_dir / "training_handoff.json"),
            handoff_sha256="pending",
            evidence_refs=[rel(results_path), rel(config_path)],
        )
        stamped, sha = stamp_and_write(
            handoff,
            self.context.run_dir / "training_handoff.json",
            "handoff_path",
            "handoff_sha256",
        )
        return _compact(
            {"status": stamped.status, "handoff_path": stamped.handoff_path, "handoff_sha256": sha}
        )


# --------------------------------------------------------------------------- #
# Agent 3 — Evaluation & Governance Reviewer
# --------------------------------------------------------------------------- #
class ValidateTrainingHandoffTool(_ContextTool):
    name: str = "validate_training_handoff"
    description: str = (
        "Verify the training handoff status, required candidates, selection metric, and "
        "that test was not accessed."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        handoff_path = self.context.run_dir / "training_handoff.json"
        issues: list[str] = []
        test_accessed: Any = None
        if not handoff_path.exists():
            issues.append("training_handoff.json is missing")
        else:
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            test_accessed = handoff.get("test_accessed")
            if handoff.get("status") != "PASS":
                issues.append(f"training handoff status is {handoff.get('status')}")
            if test_accessed is not False:
                issues.append("training handoff reports prior test access")
            count = (handoff.get("candidate_results") or {}).get("candidate_count") or 0
            if count < 3:
                issues.append("fewer than three candidates were compared")
        return _compact(
            {
                "passed": not issues,
                "test_accessed": test_accessed,
                "evidence_ref": "tool:validate_training_handoff#passed",
                "issues": issues,
            }
        )


class LockWinnerAndEvaluateTestTool(_ContextTool):
    name: str = "lock_winner_and_evaluate_test"
    description: str = (
        "Select the winner by validation macro F1 + tie-break, evaluate the test set "
        "exactly once, and write model.joblib, metrics, metadata, and figures."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        features_df = F.read_features_csv(self.context.scientist_dir / "features.csv")
        results = json.loads(
            (self.context.scientist_dir / "candidate_results.json").read_text(encoding="utf-8")
        )
        bundle = M.lock_winner_and_evaluate_test(
            features_df,
            self.context.scientist_dir,
            candidate_results=results,
            training_data_sha256=self.context.input_sha256,
            features_sha256=d.sha256_file(self.context.scientist_dir / "features.csv"),
        )
        return _compact(
            {
                "selected_candidate_id": bundle.selected_candidate_id,
                "performed_once": True,
                "model": {
                    "path": rel(bundle.model_path),
                    "sha256": d.sha256_file(bundle.model_path),
                },
                "metrics": {
                    "path": rel(bundle.metrics_path),
                    "sha256": d.sha256_file(bundle.metrics_path),
                },
                "model_metadata": {
                    "path": rel(bundle.metadata_path),
                    "sha256": d.sha256_file(bundle.metadata_path),
                    "round_trip_validated": bundle.round_trip_validated,
                },
                "validation_macro_f1": bundle.metrics["validation"]["macro_f1"],
                "test_macro_f1": bundle.metrics["test"]["macro_f1"],
                "selection_evidence_ref": "artifacts/scientist/metrics.json#validation.macro_f1",
                "test_evidence_ref": "artifacts/scientist/metrics.json#test.macro_f1",
            }
        )


class ReadEvaluationBundleTool(_ContextTool):
    name: str = "read_evaluation_bundle"
    description: str = (
        "Return compact computed validation/test metrics, per-class results, confusion "
        "summaries, timing, and provenance (never raw rows)."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        metrics = json.loads(
            (self.context.scientist_dir / "metrics.json").read_text(encoding="utf-8")
        )
        keys = (
            "selected_candidate_id",
            "selected_model",
            "primary_metric",
            "seed",
            "classes",
            "class_labels",
            "selection_rule",
            "split_counts",
            "validation",
            "test",
            "candidates",
            "validation_ranking",
            "timing",
        )
        return _compact({k: metrics[k] for k in keys if k in metrics})


class RenderScientistReportsTool(_ContextTool):
    name: str = "render_scientist_reports"
    description: str = (
        "Render evaluation_report.md and model_card.md from the computed metrics; every "
        "number is validated against machine evidence before writing."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        metrics = json.loads(
            (self.context.scientist_dir / "metrics.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (self.context.scientist_dir / "model_metadata.json").read_text(encoding="utf-8")
        )
        eval_md, _ = render_evaluation_report(metrics)
        card_md, _ = render_model_card(metrics, metadata)
        eval_path = self.context.scientist_dir / "evaluation_report.md"
        card_path = self.context.scientist_dir / "model_card.md"
        eval_path.write_text(eval_md, encoding="utf-8")
        card_path.write_text(card_md, encoding="utf-8")
        return _compact(
            {
                "evaluation_report": {
                    "path": rel(eval_path),
                    "sha256": d.sha256_file(eval_path),
                },
                "model_card": {"path": rel(card_path), "sha256": d.sha256_file(card_path)},
            }
        )


class ValidateScientistArtifactsTool(_ContextTool):
    name: str = "validate_scientist_artifacts"
    description: str = (
        "Validate the four required Scientist artifacts, hashes, trusted-model round "
        "trip, metadata, and report/metric consistency."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        report = artifact_validation.validate_scientist_artifacts(
            features_path=self.context.scientist_dir / "features.csv",
            model_path=self.context.scientist_dir / "model.joblib",
            eval_report_path=self.context.scientist_dir / "evaluation_report.md",
            model_card_path=self.context.scientist_dir / "model_card.md",
            metrics_path=self.context.scientist_dir / "metrics.json",
            metadata_path=self.context.scientist_dir / "model_metadata.json",
        )
        # Surface the exact per-artifact refs so the governance agent can populate
        # `required_artifacts` (name/path/sha256/size_bytes) without guessing.
        required_artifacts = [
            {
                "name": name,
                "path": rel(self.context.scientist_dir / name),
                "sha256": d.sha256_file(self.context.scientist_dir / name),
                "size_bytes": (self.context.scientist_dir / name).stat().st_size,
            }
            for name in (
                "features.csv",
                "model.joblib",
                "evaluation_report.md",
                "model_card.md",
            )
            if (self.context.scientist_dir / name).exists()
        ]
        return _compact(
            {
                "passed": report.ok,
                "evidence_ref": "tool:validate_scientist_artifacts#passed",
                "required_artifacts": required_artifacts,
                "issues": [i.to_dict() for i in report.issues],
            }
        )


class WriteScientistHandoffTool(_ContextTool):
    name: str = "write_scientist_handoff"
    description: str = (
        "Persist the final Scientist crew handoff. The tool builds the machine record "
        "(required artifacts, selection, metrics, hashes, pass/fail) from the trusted "
        "on-disk artifacts; you may supply an interpretive `summary` and nothing else."
    )
    args_schema: type[BaseModel] = _WriteRecordArgs

    def _run(self, record: dict[str, Any] | None = None) -> str:
        sci = self.context.scientist_dir
        metrics_path = sci / "metrics.json"
        metadata_path = sci / "model_metadata.json"
        model_path = sci / "model.joblib"

        report = artifact_validation.validate_scientist_artifacts(
            features_path=sci / "features.csv",
            model_path=model_path,
            eval_report_path=sci / "evaluation_report.md",
            model_card_path=sci / "model_card.md",
            metrics_path=metrics_path,
            metadata_path=metadata_path,
        )
        passed = report.ok
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        required_artifacts = [
            m.ScientistArtifactRef(
                name=name,
                path=rel(sci / name),
                sha256=d.sha256_file(sci / name),
                size_bytes=(sci / name).stat().st_size,
            )
            for name in ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md")
        ]

        handoff = m.ScientistCrewHandoff(
            status="PASS" if passed else "BLOCKED",
            run_id=self.context.run_id,
            summary=_opt_summary(
                record,
                "Winner selected on validation macro F1 and evaluated once on the test month.",
            ),
            required_artifacts=required_artifacts,
            selection=m.Selection(
                primary_metric="macro_f1",
                selected_candidate_id=metrics.get("selected_candidate_id"),
                validation_metric_value=metrics.get("validation", {}).get("macro_f1"),
                selection_evidence_ref="artifacts/scientist/metrics.json#validation.macro_f1",
            ),
            test_evaluation=m.TestEvaluation(
                performed_once=True,
                macro_f1=metrics.get("test", {}).get("macro_f1"),
                evidence_ref="artifacts/scientist/metrics.json#test.macro_f1",
            ),
            model_metadata=m.ModelMetadataRef(
                path=rel(metadata_path),
                sha256=d.sha256_file(metadata_path),
                round_trip_validated=bool(metadata.get("round_trip_validated")),
            ),
            metrics=m.MetricsRef(path=rel(metrics_path), sha256=d.sha256_file(metrics_path)),
            validation=m.ValidationResult(
                passed=passed, evidence_ref="tool:validate_scientist_artifacts#passed"
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
                        "model.joblib is pickle-based; load only the trusted, "
                        "hash-verified artifact."
                    ),
                    evidence_ref="artifacts/scientist/model_metadata.json#security_note",
                ),
            ],
            issues=[],
            handoff_ready=passed,
            handoff_path=rel(self.context.run_dir / "scientist_crew_handoff.json"),
            handoff_sha256="pending",
            evidence_refs=[rel(sci / a.name) for a in required_artifacts],
        )
        stamped, sha = stamp_and_write(
            handoff,
            self.context.run_dir / "scientist_crew_handoff.json",
            "handoff_path",
            "handoff_sha256",
        )
        return _compact(
            {"status": stamped.status, "handoff_path": stamped.handoff_path, "handoff_sha256": sha}
        )


# --------------------------------------------------------------------------- #
# Tool assembly per agent (mirrors the allowed-tools lists in the prompts)
# --------------------------------------------------------------------------- #
def build_scientist_tools(context: ScientistRunContext) -> dict[str, list[BaseTool]]:
    """Return the tool list for each Scientist role, bound to ``context``."""
    return {
        "contract_feature_engineer": [
            ValidateAnalystHandoffTool(context=context),
            RunFeaturePipelineTool(context=context),
            RunLeakageAuditTool(context=context),
            ValidateFeatureHandoffTool(context=context),
            WriteFeatureEngineeringHandoffTool(context=context),
        ],
        "model_trainer": [
            ReadExperimentConfigTool(context=context),
            RunCandidateExperimentsTool(context=context),
            ValidateTrainingOutputsTool(context=context),
            WriteTrainingHandoffTool(context=context),
        ],
        "evaluation_governance": [
            ValidateTrainingHandoffTool(context=context),
            LockWinnerAndEvaluateTestTool(context=context),
            ReadEvaluationBundleTool(context=context),
            RenderScientistReportsTool(context=context),
            ValidateScientistArtifactsTool(context=context),
            WriteScientistHandoffTool(context=context),
        ],
    }


__all__ = ["build_scientist_tools"]
