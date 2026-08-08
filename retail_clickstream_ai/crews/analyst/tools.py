"""CrewAI tools for the Analyst crew.

Every tool is a thin, typed wrapper around a *tested* deterministic function —
agents never parse raw CSV text or compute a metric themselves. Tools return
compact JSON strings (never raw rows). The three ``write_*`` tools validate the
agent-supplied record through its Pydantic model and stamp it with a
content hash before persisting, so an agent response cannot become an
unvalidated artifact.

All tools are bound to an :class:`AnalystRunContext` at construction, so the run
id, trusted paths, and input hash are fixed for the run.
"""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict

from retail_clickstream_ai.crews.analyst import models as m
from retail_clickstream_ai.crews.context import AnalystRunContext
from retail_clickstream_ai.crews.io_helpers import rel, stamp_and_write
from retail_clickstream_ai.pipeline import cleaning, eda
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.reporting.eda_report import render_eda_report
from retail_clickstream_ai.reporting.insights import render_insights
from retail_clickstream_ai.validation import artifacts as artifact_validation
from retail_clickstream_ai.validation.contract import DatasetContract


def _compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _WriteRecordArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record: dict[str, Any]


class _ReadMetricsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keys: list[str] = []


class _ContextTool(BaseTool):
    """Base for tools bound to a run context (arbitrary type allowed as a field)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    context: AnalystRunContext


# --------------------------------------------------------------------------- #
# Agent 1 — Source & Quality Analyst
# --------------------------------------------------------------------------- #
class ReadSourceMetadataTool(_ContextTool):
    name: str = "read_source_metadata"
    description: str = "Return normalized source provenance/license metadata for the run."
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        return self.context.source_metadata_path.read_text(encoding="utf-8")


class ReadRawValidationReportTool(_ContextTool):
    name: str = "read_raw_validation_report"
    description: str = (
        "Return the deterministic raw-data validation report (hash, schema, ranges, session order)."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        return self.context.raw_validation_report_path.read_text(encoding="utf-8")


class ReadRawProfileTool(_ContextTool):
    name: str = "read_raw_profile"
    description: str = (
        "Return a compact observed profile of the raw data (counts/distributions, no rows)."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        return self.context.raw_profile_path.read_text(encoding="utf-8")


class WriteSourceQualityReviewTool(_ContextTool):
    name: str = "write_source_quality_review"
    description: str = "Validate and persist the structured SourceQualityReview record."
    args_schema: type[BaseModel] = _WriteRecordArgs

    def _run(self, record: dict[str, Any]) -> str:
        record.setdefault("run_id", self.context.run_id)
        record.setdefault("review_path", rel(self.context.run_dir / "source_quality_review.json"))
        record.setdefault("review_sha256", "pending")
        review = m.SourceQualityReview.model_validate(record)
        stamped, sha = stamp_and_write(
            review,
            self.context.run_dir / "source_quality_review.json",
            "review_path",
            "review_sha256",
        )
        return _compact(
            {"status": stamped.status, "review_path": stamped.review_path, "review_sha256": sha}
        )


# --------------------------------------------------------------------------- #
# Agent 2 — Data Engineer
# --------------------------------------------------------------------------- #
class RunCleaningPipelineTool(_ContextTool):
    name: str = "run_cleaning_pipeline"
    description: str = (
        "Run deterministic cleaning; write clean_data.csv, the contract, and the audit."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        from retail_clickstream_ai.crews.io_helpers import write_json

        contract = DatasetContract.build()
        raw = d.read_raw_csv(self.context.input_path)
        clean_df, audit = cleaning.clean_dataframe(
            raw, contract, require_all_months=self.context.require_all_months
        )
        clean_path = self.context.analyst_dir / "clean_data.csv"
        stat = cleaning.write_clean_csv(clean_df, clean_path)
        contract_path = contract.write(self.context.analyst_dir / "dataset_contract.json")
        audit_path = write_json(self.context.run_dir / "transformation_audit.json", audit.to_dict())
        return _compact(
            {
                "clean_data": {
                    "path": rel(clean_path),
                    **{k: stat[k] for k in ("sha256", "row_count", "column_count")},
                },
                "dataset_contract": {
                    "path": rel(contract_path),
                    "sha256": d.sha256_file(contract_path),
                    "contract_version": contract.data["contract_version"],
                },
                "transformation_audit": {
                    "path": rel(audit_path),
                    "sha256": d.sha256_file(audit_path),
                    "rule_count": len(audit.rules),
                    "fatal_issue_count": audit.fatal_issue_count,
                    "rules": [r.name for r in audit.rules],
                },
                "rows_preserved": audit.rows_preserved,
            }
        )


class ValidateCleanedHandoffTool(_ContextTool):
    name: str = "validate_cleaned_handoff"
    description: str = (
        "Validate the produced clean_data.csv against the dataset contract (hash/schema/ordering)."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        contract = DatasetContract.build()
        report = artifact_validation.validate_clean_file_against_contract(
            self.context.analyst_dir / "clean_data.csv",
            contract,
            pin_hash=self.context.pin_hash,
            require_all_months=self.context.require_all_months,
        )
        return _compact(
            {
                "passed": report.ok,
                "evidence_ref": "tool:validate_cleaned_handoff#passed",
                "issues": [i.to_dict() for i in report.issues],
            }
        )


class WriteDataEngineeringHandoffTool(_ContextTool):
    name: str = "write_data_engineering_handoff"
    description: str = "Validate and persist the structured DataEngineeringHandoff record."
    args_schema: type[BaseModel] = _WriteRecordArgs

    def _run(self, record: dict[str, Any]) -> str:
        record.setdefault("run_id", self.context.run_id)
        record.setdefault("input_sha256", self.context.input_sha256)
        record.setdefault(
            "handoff_path", rel(self.context.run_dir / "data_engineering_handoff.json")
        )
        record.setdefault("handoff_sha256", "pending")
        handoff = m.DataEngineeringHandoff.model_validate(record)
        stamped, sha = stamp_and_write(
            handoff,
            self.context.run_dir / "data_engineering_handoff.json",
            "handoff_path",
            "handoff_sha256",
        )
        return _compact(
            {"status": stamped.status, "handoff_path": stamped.handoff_path, "handoff_sha256": sha}
        )


# --------------------------------------------------------------------------- #
# Agent 3 — EDA & Business Analyst
# --------------------------------------------------------------------------- #
class RunEdaPipelineTool(_ContextTool):
    name: str = "run_eda_pipeline"
    description: str = (
        "Compute descriptive EDA metrics and at least five figures from the validated cleaned data."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        from retail_clickstream_ai.crews.io_helpers import write_json

        contract = DatasetContract.build()
        clean_df = cleaning.read_clean_csv(self.context.analyst_dir / "clean_data.csv")
        metrics = eda.compute_metrics(
            clean_df,
            input_sha256=self.context.input_sha256,
            clean_sha256=d.sha256_file(self.context.analyst_dir / "clean_data.csv"),
            contract_version=contract.data["contract_version"],
        )
        metrics_path = write_json(self.context.eda_dir / "eda_metrics.json", metrics)
        eda.write_tables(metrics, self.context.eda_dir / "tables")
        figures = eda.make_figures(clean_df, metrics)
        manifest_path = write_json(
            self.context.eda_dir / "figure_manifest.json", eda.figure_manifest(figures)
        )
        return _compact(
            {
                "metrics_path": rel(metrics_path),
                "metrics_sha256": d.sha256_file(metrics_path),
                "figure_manifest_path": rel(manifest_path),
                "figure_count": len(figures),
                "metric_keys": sorted(metrics.keys()),
                "headline_keys": sorted(metrics["headline"].keys()),
            }
        )


class ReadEdaMetricsTool(_ContextTool):
    name: str = "read_eda_metrics"
    description: str = (
        "Return selected machine-computed EDA metrics by top-level key (no raw rows)."
    )
    args_schema: type[BaseModel] = _ReadMetricsArgs

    def _run(self, keys: list[str] | None = None) -> str:
        metrics = json.loads(
            (self.context.eda_dir / "eda_metrics.json").read_text(encoding="utf-8")
        )
        if not keys:
            return _compact({"available_keys": sorted(metrics.keys())})
        return _compact({k: metrics.get(k) for k in keys})


class RenderAnalystReportsTool(_ContextTool):
    name: str = "render_analyst_reports"
    description: str = (
        "Render insights.md and the self-contained eda_report.html from validated metrics; "
        "unsupported numbers are rejected."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        contract = DatasetContract.build()
        metrics = json.loads(
            (self.context.eda_dir / "eda_metrics.json").read_text(encoding="utf-8")
        )
        clean_df = cleaning.read_clean_csv(self.context.analyst_dir / "clean_data.csv")
        figures = eda.make_figures(clean_df, metrics)

        insights_md, _ = render_insights(metrics)
        insights_path = self.context.analyst_dir / "insights.md"
        insights_path.write_text(insights_md, encoding="utf-8")

        report_html, _ = render_eda_report(metrics, figures, contract=contract)
        report_path = self.context.analyst_dir / "eda_report.html"
        report_path.write_text(report_html, encoding="utf-8")
        return _compact(
            {
                "insights_md": {"path": rel(insights_path), "sha256": d.sha256_file(insights_path)},
                "eda_report_html": {"path": rel(report_path), "sha256": d.sha256_file(report_path)},
            }
        )


class ValidateAnalystArtifactsTool(_ContextTool):
    name: str = "validate_analyst_artifacts"
    description: str = (
        "Validate the four required Analyst artifacts, contract agreement, and report sections."
    )
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> str:
        contract = DatasetContract.build()
        report = artifact_validation.validate_analyst_artifacts(
            clean_path=self.context.analyst_dir / "clean_data.csv",
            eda_report_path=self.context.analyst_dir / "eda_report.html",
            insights_path=self.context.analyst_dir / "insights.md",
            contract_path=self.context.analyst_dir / "dataset_contract.json",
            contract=contract,
            pin_hash=self.context.pin_hash,
            require_all_months=self.context.require_all_months,
        )
        return _compact(
            {
                "passed": report.ok,
                "evidence_ref": "tool:validate_analyst_artifacts#passed",
                "issues": [i.to_dict() for i in report.issues],
            }
        )


class WriteAnalystHandoffTool(_ContextTool):
    name: str = "write_analyst_handoff"
    description: str = "Validate and persist the final structured AnalystCrewHandoff record."
    args_schema: type[BaseModel] = _WriteRecordArgs

    def _run(self, record: dict[str, Any]) -> str:
        record.setdefault("run_id", self.context.run_id)
        record.setdefault("input_sha256", self.context.input_sha256)
        record.setdefault("handoff_path", rel(self.context.run_dir / "analyst_crew_handoff.json"))
        record.setdefault("handoff_sha256", "pending")
        handoff = m.AnalystCrewHandoff.model_validate(record)
        stamped, sha = stamp_and_write(
            handoff,
            self.context.run_dir / "analyst_crew_handoff.json",
            "handoff_path",
            "handoff_sha256",
        )
        return _compact(
            {"status": stamped.status, "handoff_path": stamped.handoff_path, "handoff_sha256": sha}
        )


# --------------------------------------------------------------------------- #
# Tool assembly per agent (mirrors the allowed-tools lists in the prompts)
# --------------------------------------------------------------------------- #
def build_analyst_tools(context: AnalystRunContext) -> dict[str, list[BaseTool]]:
    """Return the tool list for each Analyst role, bound to ``context``."""
    return {
        "source_quality": [
            ReadSourceMetadataTool(context=context),
            ReadRawValidationReportTool(context=context),
            ReadRawProfileTool(context=context),
            WriteSourceQualityReviewTool(context=context),
        ],
        "data_engineer": [
            RunCleaningPipelineTool(context=context),
            ValidateCleanedHandoffTool(context=context),
            WriteDataEngineeringHandoffTool(context=context),
        ],
        "eda_business": [
            RunEdaPipelineTool(context=context),
            ReadEdaMetricsTool(context=context),
            RenderAnalystReportsTool(context=context),
            ValidateAnalystArtifactsTool(context=context),
            WriteAnalystHandoffTool(context=context),
        ],
    }


__all__ = [
    "build_analyst_tools",
    "ReadSourceMetadataTool",
    "ReadRawValidationReportTool",
    "ReadRawProfileTool",
    "WriteSourceQualityReviewTool",
    "RunCleaningPipelineTool",
    "ValidateCleanedHandoffTool",
    "WriteDataEngineeringHandoffTool",
    "RunEdaPipelineTool",
    "ReadEdaMetricsTool",
    "RenderAnalystReportsTool",
    "ValidateAnalystArtifactsTool",
    "WriteAnalystHandoffTool",
]
