"""CrewAI Flow orchestration — the run conductor.

One :class:`RetailClickstreamFlow` automatically and visibly coordinates the two
crews behind deterministic gates::

    prepare -> Analyst crew -> Analyst gate -> Scientist crew -> model gate -> manifest

The **deterministic validators own every pass/fail decision**, never the LLM. Any
failed gate routes to a single terminal failure handler that writes an actionable
``failure_report.json`` and *blocks* downstream execution — the Scientist crew is
unreachable unless the Analyst gate passes, and no manifest is written on failure.

Design guarantees:

* Typed Pydantic state (:class:`FlowState`) carries the run id, status, timestamps,
  input hash, per-artifact paths/hashes, the selected model/primary metric, and the
  accumulated validation errors.
* Heavy computation stays in the existing tested crews/pipelines and validators; the
  Flow only orchestrates, gates, and records.
* Unexpected exceptions are *converted* (not suppressed) into a sanitized failure
  state, with the full traceback kept in a local ``flow.log`` (never secrets).
* A repeated run id never silently overwrites a prior run, and earlier successful
  artifacts are never deleted.

Importing this module runs nothing (safe for the Streamlit app to import for reading
manifests). A real run is started only through :func:`run_flow` / :func:`main`; the
crews require OpenAI credentials only when they actually start.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from crewai import Flow
from crewai.flow import listen, router, start
from pydantic import BaseModel, Field

from retail_clickstream_ai import paths
from retail_clickstream_ai.config import ConfigurationError, load_settings
from retail_clickstream_ai.crews.analyst.crew import run_analyst_crew
from retail_clickstream_ai.crews.io_helpers import rel, write_json
from retail_clickstream_ai.crews.scientist.crew import run_scientist_crew
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline.analyst_pipeline import (
    AnalystPipelineError,
    run_analyst_pipeline,
)
from retail_clickstream_ai.pipeline.scientist_pipeline import (
    ScientistPipelineError,
    run_scientist_pipeline,
)
from retail_clickstream_ai.validation import artifacts as artifact_validation
from retail_clickstream_ai.validation.contract import DatasetContract
from retail_clickstream_ai.validation.errors import ValidationReport
from retail_clickstream_ai.validation.raw import validate_dataframe

APP_NAME = "retail-clickstream-ai"
DEFAULT_INPUT = "data/raw/e-shop clothing 2008.csv"

# Route labels emitted by the routers (deterministic gate decisions).
_ROUTE_FAILED = "failed"
_ROUTE_PREPARE_OK = "prepare_ok"
_ROUTE_ANALYST_OK = "analyst_ok"
_ROUTE_MODEL_OK = "model_ok"

# The four/six required artifacts each gate must find on disk (exact names).
_ANALYST_REQUIRED = ("clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json")
_SCIENTIST_REQUIRED = (
    "features.csv",
    "model.joblib",
    "evaluation_report.md",
    "model_card.md",
    "metrics.json",
    "model_metadata.json",
)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _now() -> str:
    """UTC timestamp in ISO-8601."""
    return datetime.now(UTC).isoformat()


def _package_version() -> str:
    """Installed distribution version, falling back to the package's ``__version__``."""
    try:
        from importlib.metadata import version

        return version(APP_NAME)
    except Exception:  # noqa: BLE001 - fall back to the in-package constant
        from retail_clickstream_ai import __version__

        return __version__


def _artifact_info(path: Path) -> ArtifactInfo:
    """Record an artifact's repo-relative path, content hash, and size."""
    return ArtifactInfo(
        path=rel(path),
        sha256=d.sha256_file(path),
        size_bytes=int(os.path.getsize(path)),
    )


class _FlowStepError(Exception):
    """An *expected*, structured step failure (raw invalid, run-id reuse, crew error).

    Carries the actionable cause/remediation and an optional validation report so the
    step wrapper can record a clean, non-suppressing failure state.
    """

    def __init__(
        self,
        cause: str,
        remediation: str,
        *,
        report: ValidationReport | None = None,
        detail: str = "",
    ) -> None:
        self.cause = cause
        self.remediation = remediation
        self.report = report
        self.detail = detail or cause
        super().__init__(cause)


# --------------------------------------------------------------------------- #
# Typed Flow state (Pydantic)
# --------------------------------------------------------------------------- #
class RunStatus(StrEnum):
    """Lifecycle status of one Flow run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Stage(StrEnum):
    """Clear event names for each Flow step (recorded in ``stage_events``)."""

    PREPARE = "prepare"
    ANALYST_CREW = "analyst_crew"
    ANALYST_GATE = "analyst_gate"
    SCIENTIST_CREW = "scientist_crew"
    MODEL_GATE = "model_gate"
    MANIFEST = "manifest"


class ArtifactInfo(BaseModel):
    """A produced artifact's path + content hash + size."""

    path: str = ""
    sha256: str = ""
    size_bytes: int = 0


class StageEvent(BaseModel):
    """A recorded step: name, outcome, timestamps, and duration (seconds)."""

    stage: str
    status: str = "running"
    started_at: str = ""
    ended_at: str = ""
    duration_seconds: float = 0.0


class FlowState(BaseModel):
    """Typed state for one Flow run (all fields defaulted; seeded via ``kickoff``)."""

    run_id: str = ""
    status: RunStatus = RunStatus.PENDING
    engine: str = "crew"  # "crew" (LLM) or "deterministic" (no LLM)
    pin_hash: bool = True
    require_all_months: bool = True

    started_at: str = ""
    ended_at: str = ""

    input_path: str = ""
    input_sha256: str = ""

    analyst_run_id: str = ""
    analyst_run_dir: str = ""
    scientist_run_id: str = ""
    scientist_run_dir: str = ""

    analyst_artifacts: dict[str, ArtifactInfo] = Field(default_factory=dict)
    scientist_artifacts: dict[str, ArtifactInfo] = Field(default_factory=dict)

    selected_model: str = ""
    selected_model_family: str = ""
    primary_metric: str = ""
    primary_metric_value: float | None = None

    stage_events: list[StageEvent] = Field(default_factory=list)
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)

    failed_stage: str = ""
    failure_cause: str = ""
    failure_remediation: str = ""
    failure_detail: str = ""

    package_version: str = ""
    manifest_path: str = ""
    failure_report_path: str = ""


# --------------------------------------------------------------------------- #
# The Flow
# --------------------------------------------------------------------------- #
class RetailClickstreamFlow(Flow[FlowState]):
    """Sequential CrewAI Flow: prepare -> Analyst -> gate -> Scientist -> gate -> manifest."""

    # -- steps & routers --------------------------------------------------- #
    @start()
    def prepare(self) -> None:
        """Validate configuration + raw input and create the isolated run context."""
        ev, t0 = self._begin(Stage.PREPARE)
        try:
            self._prepare_impl()
            self._finish(ev, t0, "passed")
        except _FlowStepError as exc:
            self._record_step_error(Stage.PREPARE, exc)
            self._finish(ev, t0, "failed")
        except Exception as exc:  # noqa: BLE001 - converted (not suppressed) to a sanitized state
            self._record_unexpected(Stage.PREPARE, exc)
            self._finish(ev, t0, "failed")

    @router(prepare)
    def route_prepare(self) -> str:
        return _ROUTE_FAILED if self.state.status == RunStatus.FAILED else _ROUTE_PREPARE_OK

    @listen(_ROUTE_PREPARE_OK)
    def run_analyst(self) -> None:
        """Run the Analyst crew (LLM) — reachable only after prepare passes."""
        ev, t0 = self._begin(Stage.ANALYST_CREW)
        try:
            self._run_analyst_impl()
            self._finish(ev, t0, "passed")
        except (AnalystPipelineError, ConfigurationError) as exc:
            self._record_step_error(
                Stage.ANALYST_CREW,
                _FlowStepError(
                    "the Analyst crew did not complete successfully",
                    "Confirm OpenAI credentials and the raw input, then re-run the Analyst crew.",
                    detail=str(exc),
                ),
            )
            self._finish(ev, t0, "failed")
        except Exception as exc:  # noqa: BLE001 - converted to a sanitized failure state
            self._record_unexpected(Stage.ANALYST_CREW, exc)
            self._finish(ev, t0, "failed")

    @router(run_analyst)
    def analyst_gate(self) -> str:
        """Deterministic Analyst gate — owns the pass/fail decision (never the LLM)."""
        if self.state.status == RunStatus.FAILED:
            return _ROUTE_FAILED  # the crew step already failed; do not run the gate
        ev, t0 = self._begin(Stage.ANALYST_GATE)
        try:
            ok = self._analyst_gate_impl()
            self._finish(ev, t0, "passed" if ok else "failed")
            return _ROUTE_ANALYST_OK if ok else _ROUTE_FAILED
        except Exception as exc:  # noqa: BLE001 - converted to a sanitized failure state
            self._record_unexpected(Stage.ANALYST_GATE, exc)
            self._finish(ev, t0, "failed")
            return _ROUTE_FAILED

    @listen(_ROUTE_ANALYST_OK)
    def run_scientist(self) -> None:
        """Run the Scientist crew (LLM) — impossible unless the Analyst gate passed."""
        ev, t0 = self._begin(Stage.SCIENTIST_CREW)
        try:
            self._run_scientist_impl()
            self._finish(ev, t0, "passed")
        except (ScientistPipelineError, ConfigurationError) as exc:
            self._record_step_error(
                Stage.SCIENTIST_CREW,
                _FlowStepError(
                    "the Scientist crew did not complete successfully",
                    "Inspect the Scientist run; never repair Analyst output in Scientist code.",
                    detail=str(exc),
                ),
            )
            self._finish(ev, t0, "failed")
        except Exception as exc:  # noqa: BLE001 - converted to a sanitized failure state
            self._record_unexpected(Stage.SCIENTIST_CREW, exc)
            self._finish(ev, t0, "failed")

    @router(run_scientist)
    def model_gate(self) -> str:
        """Deterministic model gate — trusted-hash pre-check + round-trip smoke test."""
        if self.state.status == RunStatus.FAILED:
            return _ROUTE_FAILED
        ev, t0 = self._begin(Stage.MODEL_GATE)
        try:
            ok = self._model_gate_impl()
            self._finish(ev, t0, "passed" if ok else "failed")
            return _ROUTE_MODEL_OK if ok else _ROUTE_FAILED
        except Exception as exc:  # noqa: BLE001 - converted to a sanitized failure state
            self._record_unexpected(Stage.MODEL_GATE, exc)
            self._finish(ev, t0, "failed")
            return _ROUTE_FAILED

    @listen(_ROUTE_MODEL_OK)
    def write_manifest(self) -> None:
        """Terminal success step — write the run manifest (only reached when all gates pass)."""
        ev, t0 = self._begin(Stage.MANIFEST)
        # Finalize the run + this stage event *before* writing, so the manifest is a
        # self-consistent snapshot (it records its own step as passed).
        self.state.ended_at = _now()
        self.state.status = RunStatus.SUCCESS
        self._finish(ev, t0, "passed")
        try:
            path = self._write_manifest()
            self.state.manifest_path = rel(path)
            self._log(f"SUCCESS run {self.state.run_id} manifest={self.state.manifest_path}")
        except Exception as exc:  # noqa: BLE001 - converted; no manifest on failure
            ev.status = "failed"
            self.state.manifest_path = ""
            self._record_unexpected(Stage.MANIFEST, exc)
            self.state.failure_report_path = rel(self._write_failure_report())

    @listen(_ROUTE_FAILED)
    def handle_failure(self) -> None:
        """Single terminal failure handler — write the actionable failure report."""
        self.state.status = RunStatus.FAILED
        if not self.state.ended_at:
            self.state.ended_at = _now()
        path = self._write_failure_report()
        self.state.failure_report_path = rel(path)

    # -- step implementations --------------------------------------------- #
    def _prepare_impl(self) -> None:
        st = self.state
        st.status = RunStatus.RUNNING
        st.started_at = _now()
        st.package_version = _package_version()

        # Configuration (references only — never secrets).
        settings = load_settings()

        # Raw input must exist and be non-empty before we resolve identity.
        if not st.input_path:
            raise _FlowStepError(
                "no raw input path was provided",
                f"Pass an input CSV (e.g. --input '{DEFAULT_INPUT}').",
            )
        input_path = Path(st.input_path)
        if not input_path.exists() or os.path.getsize(input_path) == 0:
            raise _FlowStepError(
                "the raw input file is missing or empty",
                "Provide a readable, non-empty raw CSV (see data/README.md).",
                detail=f"input_path={st.input_path}",
            )
        st.input_sha256 = d.sha256_file(input_path)

        # Resolve a run id; default is unique per run so re-runs never collide.
        if not st.run_id:
            stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            st.run_id = f"flow-{st.input_sha256[:12]}-{stamp}"

        # Run-id reuse guard: never overwrite a prior run's manifest (checked before
        # creating or writing anything in the target directory).
        if (self._run_dir() / "run_manifest.json").exists():
            raise _FlowStepError(
                f"run id '{st.run_id}' already has a completed run_manifest.json",
                "Use a new run id; refusing to overwrite or delete a prior successful run.",
                detail="run_id_reuse",
            )

        # Isolated run context + log (created only for a fresh run id).
        self._run_dir().mkdir(parents=True, exist_ok=True)
        self._log(
            "prepared run "
            f"run_id={st.run_id} engine={st.engine} "
            f"model={settings.openai_model_name or '(unset)'} "
            f"artifact_root={settings.artifact_root} input_sha256={st.input_sha256}"
        )

        # Validate the raw input against the dataset contract (deterministic).
        contract = DatasetContract.build()
        raw = d.read_raw_csv(input_path)
        report = validate_dataframe(raw, contract, require_all_months=st.require_all_months)
        if not report.ok:
            raise _FlowStepError(
                "the raw input failed dataset-contract validation",
                "Fix the flagged raw columns/rows to satisfy the contract, then re-run.",
                report=report,
            )

    def _run_analyst_impl(self) -> None:
        st = self.state
        ctx: Any
        if st.engine == "crew":
            _output, ctx = run_analyst_crew(
                st.input_path,
                run_id=None,
                pin_hash=st.pin_hash,
                require_all_months=st.require_all_months,
            )
        else:
            ctx = run_analyst_pipeline(
                st.input_path,
                run_id=None,
                pin_contract_hash=st.pin_hash,
                require_all_months=st.require_all_months,
            )
        st.analyst_run_id = str(getattr(ctx, "run_id", ""))
        run_dir = getattr(ctx, "run_dir", None)
        st.analyst_run_dir = rel(run_dir) if run_dir is not None else ""
        self._log(f"analyst crew finished analyst_run_id={st.analyst_run_id}")

    def _analyst_gate_impl(self) -> bool:
        st = self.state
        an = paths.analyst_artifacts()
        report = artifact_validation.validate_analyst_artifacts(
            clean_path=an / "clean_data.csv",
            eda_report_path=an / "eda_report.html",
            insights_path=an / "insights.md",
            contract_path=an / "dataset_contract.json",
            contract=DatasetContract.build(),
            pin_hash=st.pin_hash,
            require_all_months=st.require_all_months,
        )
        if not report.ok:
            self._record_failure(
                Stage.ANALYST_GATE,
                "Analyst artifacts failed deterministic validation",
                "Re-run the Analyst crew; do not repair its output in downstream code.",
                report=report,
            )
            return False

        # Absence of reported fatal errors: the crew must not have written a failure report.
        if (
            st.analyst_run_id
            and (paths.runs_artifacts() / st.analyst_run_id / "failure_report.json").exists()
        ):
            self._record_failure(
                Stage.ANALYST_GATE,
                "the Analyst crew reported a fatal error (failure_report.json present)",
                "Inspect the Analyst run's failure report and re-run the Analyst crew.",
            )
            return False

        for name in _ANALYST_REQUIRED:
            st.analyst_artifacts[name] = _artifact_info(an / name)
        self._log("analyst gate PASSED")
        return True

    def _run_scientist_impl(self) -> None:
        st = self.state
        ctx: Any
        if st.engine == "crew":
            _output, ctx = run_scientist_crew(run_id=None)
        else:
            ctx = run_scientist_pipeline(run_id=None, pin_analyst_hash=st.pin_hash)
        st.scientist_run_id = str(getattr(ctx, "run_id", ""))
        run_dir = getattr(ctx, "run_dir", None)
        st.scientist_run_dir = rel(run_dir) if run_dir is not None else ""
        self._log(f"scientist crew finished scientist_run_id={st.scientist_run_id}")

    def _model_gate_impl(self) -> bool:
        st = self.state
        sc = paths.scientist_artifacts()
        files = {name: sc / name for name in _SCIENTIST_REQUIRED}

        missing = [n for n, p in files.items() if not p.exists() or os.path.getsize(p) == 0]
        if missing:
            self._record_failure(
                Stage.MODEL_GATE,
                f"required model artifacts are missing or empty: {missing}",
                "Re-run the Scientist crew; never publish without all required artifacts.",
                detail=str(missing),
            )
            return False

        # Trusted-hash pre-check BEFORE loading the pickle: refuse to load an artifact
        # whose bytes do not match the recorded metadata hash (no untrusted loading).
        metadata = json.loads(files["model_metadata.json"].read_text(encoding="utf-8"))
        model_sha = d.sha256_file(files["model.joblib"])
        if metadata.get("artifact_sha256") != model_sha:
            self.state.validation_errors.append(
                {
                    "rule": "model_artifact_untrusted",
                    "message": "model.joblib hash does not match model_metadata.json",
                    "observed": model_sha,
                    "expected": str(metadata.get("artifact_sha256")),
                    "severity": "fatal",
                }
            )
            self._record_failure(
                Stage.MODEL_GATE,
                "the model artifact hash does not match its metadata; refused to load an "
                "untrusted model",
                "Re-produce model.joblib from the trusted Scientist run; never load an "
                "unverified pickle.",
                detail="artifact_sha256 mismatch",
            )
            return False

        # Deterministic gate: schema, metrics/report consistency, and the trusted-model
        # round-trip inference smoke test (loads model.joblib, predicts 4-class proba).
        report = artifact_validation.validate_scientist_artifacts(
            features_path=files["features.csv"],
            model_path=files["model.joblib"],
            eval_report_path=files["evaluation_report.md"],
            model_card_path=files["model_card.md"],
            metrics_path=files["metrics.json"],
            metadata_path=files["model_metadata.json"],
        )
        if not report.ok:
            self._record_failure(
                Stage.MODEL_GATE,
                "Scientist artifacts failed deterministic validation",
                "Inspect the failing artifact; do not publish a manifest.",
                report=report,
            )
            return False

        for name in _SCIENTIST_REQUIRED:
            st.scientist_artifacts[name] = _artifact_info(files[name])
        metrics = json.loads(files["metrics.json"].read_text(encoding="utf-8"))
        st.selected_model = str(metadata.get("selected_candidate_id", ""))
        st.selected_model_family = str(metadata.get("selected_model", ""))
        st.primary_metric = str(metrics.get("primary_metric", "macro_f1"))
        with contextlib.suppress(Exception):
            st.primary_metric_value = float(metrics["test"]["macro_f1"])
        self._log(
            f"model gate PASSED selected={st.selected_model} "
            f"{st.primary_metric}={st.primary_metric_value}"
        )
        return True

    # -- stage-event + failure recording ---------------------------------- #
    def _begin(self, stage: Stage) -> tuple[StageEvent, float]:
        ev = StageEvent(stage=stage.value, status="running", started_at=_now())
        self.state.stage_events.append(ev)
        return ev, time.perf_counter()

    def _finish(self, ev: StageEvent, t0: float, status: str) -> None:
        ev.status = status
        ev.ended_at = _now()
        ev.duration_seconds = round(time.perf_counter() - t0, 6)

    def _record_failure(
        self,
        stage: Stage,
        cause: str,
        remediation: str,
        *,
        report: ValidationReport | None = None,
        detail: str = "",
    ) -> None:
        st = self.state
        st.status = RunStatus.FAILED
        if not st.failed_stage:  # keep the first (root-cause) failure
            st.failed_stage = stage.value
            st.failure_cause = cause
            st.failure_remediation = remediation
            st.failure_detail = detail or cause
        if report is not None:
            st.validation_errors.extend(i.to_dict() for i in report.fatal_issues)

    def _record_step_error(self, stage: Stage, exc: _FlowStepError) -> None:
        self._record_failure(
            stage, exc.cause, exc.remediation, report=exc.report, detail=exc.detail
        )

    def _record_unexpected(self, stage: Stage, exc: Exception) -> None:
        """Convert an unexpected exception into a sanitized failure state + local log."""
        self._log(f"UNEXPECTED error in {stage.value}:\n{traceback.format_exc()}")
        self._record_failure(
            stage,
            f"an unexpected error occurred during '{stage.value}'",
            "Inspect flow.log for the full local traceback, fix the underlying error, and re-run.",
            detail=f"{type(exc).__name__}: {exc}",
        )

    # -- paths, logging, and writers -------------------------------------- #
    def _run_dir(self) -> Path:
        return paths.runs_artifacts() / self.state.run_id

    def _failure_dir(self) -> Path:
        """Where to write a failure report — never a directory holding a prior manifest."""
        target = self._run_dir()
        if (target / "run_manifest.json").exists():
            stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            target = paths.runs_artifacts() / f"{self.state.run_id}__rejected-{stamp}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _log(self, message: str) -> None:
        """Append a line to the run's flow.log; logging must never break the flow."""
        with contextlib.suppress(Exception):
            run_dir = self._run_dir()
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "flow.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{_now()} {message}\n")

    def _write_manifest(self) -> Path:
        st = self.state
        total = round(sum(e.duration_seconds for e in st.stage_events), 6)
        payload = {
            "app_name": APP_NAME,
            "package_version": st.package_version,
            "run_id": st.run_id,
            "status": RunStatus.SUCCESS.value,
            "engine": st.engine,
            "started_at": st.started_at,
            "ended_at": st.ended_at,
            "total_duration_seconds": total,
            "input": {"path": st.input_path, "sha256": st.input_sha256},
            "stages": [e.model_dump() for e in st.stage_events],
            "gates": {"analyst_gate": "passed", "model_gate": "passed"},
            "analyst": {
                "run_id": st.analyst_run_id,
                "run_dir": st.analyst_run_dir,
                "artifacts": {k: v.model_dump() for k, v in st.analyst_artifacts.items()},
            },
            "scientist": {
                "run_id": st.scientist_run_id,
                "run_dir": st.scientist_run_dir,
                "artifacts": {k: v.model_dump() for k, v in st.scientist_artifacts.items()},
            },
            "selection": {
                "selected_candidate_id": st.selected_model,
                "selected_model": st.selected_model_family,
                "primary_metric": st.primary_metric,
                "primary_metric_value": st.primary_metric_value,
            },
        }
        return write_json(self._run_dir() / "run_manifest.json", payload)

    def _write_failure_report(self) -> Path:
        st = self.state
        rules = [
            e.get("rule") for e in st.validation_errors if isinstance(e, dict) and e.get("rule")
        ]
        message = (
            f"Flow run '{st.run_id}' FAILED at step '{st.failed_stage}'.\n"
            f"Problem: {st.failure_cause}\n"
            f"Observed: {st.failure_detail}\n"
            f"Validation rule(s): {', '.join(str(r) for r in rules) or 'n/a'}\n"
            f"Remediation: {st.failure_remediation}\n"
            "Downstream steps did not run — no manifest was written."
        )
        payload = {
            "status": "FAIL",
            "run_id": st.run_id,
            "failed_step": st.failed_stage,
            "stage": st.failed_stage,
            "input_sha256": st.input_sha256,
            "cause": st.failure_cause,
            "observed_problem": st.failure_detail,
            "validation_rules": rules,
            "remediation": st.failure_remediation,
            "downstream_blocked": True,
            "downstream": "blocked — no downstream step ran after the failing gate.",
            "validation": {"errors": st.validation_errors},
            "message": message,
        }
        path = write_json(self._failure_dir() / "failure_report.json", payload)
        with contextlib.suppress(Exception):
            sys.stderr.write(message + "\n")
        return path


# --------------------------------------------------------------------------- #
# Programmatic + command-line entry points
# --------------------------------------------------------------------------- #
def run_flow(
    input_path: str | Path,
    *,
    run_id: str | None = None,
    engine: str = "crew",
    pin_hash: bool = True,
    require_all_months: bool = True,
    artifact_root: str | Path | None = None,
) -> FlowState:
    """Run one Flow end to end and return its typed final state.

    ``engine="crew"`` runs the real LLM-backed crews (credentials required only when a
    crew starts); ``engine="deterministic"`` runs the tested pipelines with no LLM.
    """
    if artifact_root is not None:
        os.environ["ARTIFACT_ROOT"] = str(artifact_root)
    flow = RetailClickstreamFlow()
    inputs: dict[str, Any] = {
        "input_path": str(input_path),
        "engine": engine,
        "pin_hash": pin_hash,
        "require_all_months": require_all_months,
    }
    if run_id:
        inputs["run_id"] = run_id
    flow.kickoff(inputs=inputs)
    return flow.state


def main(argv: list[str] | None = None) -> int:
    """Run the full Flow from the command line.

    Examples::

        python -m retail_clickstream_ai.flow --engine deterministic
        python -m retail_clickstream_ai.flow --input "data/raw/e-shop clothing 2008.csv"
    """
    parser = argparse.ArgumentParser(
        prog="python -m retail_clickstream_ai.flow",
        description=(
            "Run the CrewAI Flow: prepare -> Analyst crew -> gate -> Scientist crew -> "
            "gate -> manifest. Deterministic validators own every gate."
        ),
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the raw CSV.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    parser.add_argument(
        "--engine",
        choices=("crew", "deterministic"),
        default="crew",
        help="'crew' runs the real LLM crews (needs OpenAI creds); 'deterministic' runs "
        "the tested pipelines with no LLM.",
    )
    parser.add_argument(
        "--no-pin-hash",
        action="store_true",
        help="Relax the pinned-hash checks (for data that intentionally differs from the "
        "committed production file).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save the CrewAI Flow plot (HTML) into the run directory.",
    )
    args = parser.parse_args(argv)

    # Quiet CrewAI's network-touching version check for the run.
    os.environ.setdefault("CREWAI_DISABLE_VERSION_CHECK", "true")

    flow = RetailClickstreamFlow()
    inputs: dict[str, Any] = {
        "input_path": args.input,
        "engine": args.engine,
        "pin_hash": not args.no_pin_hash,
        "require_all_months": True,
    }
    if args.run_id:
        inputs["run_id"] = args.run_id
    flow.kickoff(inputs=inputs)
    state = flow.state

    if args.plot:
        with contextlib.suppress(Exception):
            plot_path = flow.plot(
                filename=str(paths.runs_artifacts() / state.run_id / "flow_plot"),
                show=False,
            )
            sys.stderr.write(f"Flow plot saved: {plot_path}\n")

    if state.status == RunStatus.SUCCESS:
        print(f"SUCCESS: run {state.run_id}")
        print(f"  selected model: {state.selected_model} ({state.selected_model_family})")
        print(f"  {state.primary_metric}: {state.primary_metric_value}")
        print(f"  manifest: {state.manifest_path}")
        return 0

    print(f"FAIL: run {state.run_id} failed at '{state.failed_stage}'", file=sys.stderr)
    print(f"  cause: {state.failure_cause}", file=sys.stderr)
    print(f"  remediation: {state.failure_remediation}", file=sys.stderr)
    print(f"  failure report: {state.failure_report_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
