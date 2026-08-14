"""Streamlit-free dashboard data services.

Every function here reads only committed/produced artifacts (never the raw
dataset), performs no network access, and never starts the CrewAI Flow. This
module is the "tested service" layer the UI (`dashboard/sections.py`, via
`dashboard/cache.py`) imports — kept import-clean of `streamlit` so it stays
unit-testable and reusable outside a Streamlit runtime.

Boundaries mirrored from the rest of the codebase:

* Numbers shown in the EDA/evaluation sections are read verbatim from
  ``eda_metrics.json`` / ``metrics.json`` — never recomputed from raw data, so
  they can never disagree with the committed reports.
* The model is loaded only after a trusted-hash pre-check against
  ``model_metadata.json`` (the same check ``flow.py``'s model gate performs) —
  never an uploaded or otherwise unverified pickle.
* The prediction form's fields come from ``model_metadata.json``'s
  ``feature_list`` (the trained pipeline's actual contract), with per-field
  domains resolved from ``dataset_contract.json`` for raw dataset columns and
  documented explicitly here for the two engineered session-history columns
  that have no raw-column counterpart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from retail_clickstream_ai import paths
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline import features as feat
from retail_clickstream_ai.reference import codebook
from retail_clickstream_ai.validation.contract import DatasetContract
from retail_clickstream_ai.validation.errors import (
    ContractValidationError,
    ValidationIssue,
    ValidationReport,
)

ANALYST_ARTIFACT_NAMES: tuple[str, ...] = (
    "clean_data.csv",
    "eda_report.html",
    "insights.md",
    "dataset_contract.json",
)
SCIENTIST_ARTIFACT_NAMES: tuple[str, ...] = (
    "features.csv",
    "model.joblib",
    "evaluation_report.md",
    "model_card.md",
    "metrics.json",
    "model_metadata.json",
)


class ArtifactMissingError(RuntimeError):
    """A required artifact file is absent or empty."""


class ModelIntegrityError(RuntimeError):
    """``model.joblib``'s hash does not match ``model_metadata.json``."""


# --------------------------------------------------------------------------- #
# Small file-read helpers
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise ArtifactMissingError(f"missing or empty artifact: {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_text(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        raise ArtifactMissingError(f"missing or empty artifact: {path}")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Artifact presence / freshness (Overview section)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactStatus:
    """One artifact's presence, size, timestamp, and content hash."""

    name: str
    path: str
    exists: bool
    size_bytes: int
    modified_at: str | None
    sha256: str | None


def artifact_status(path: Path, *, label: str | None = None) -> ArtifactStatus:
    """Stat + hash one artifact file (hash is `None` when it doesn't exist)."""
    exists = path.exists() and path.stat().st_size > 0
    if not exists:
        return ArtifactStatus(
            name=label or path.name,
            path=str(path),
            exists=False,
            size_bytes=0,
            modified_at=None,
            sha256=None,
        )
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return ArtifactStatus(
        name=label or path.name,
        path=str(path),
        exists=True,
        size_bytes=int(stat.st_size),
        modified_at=modified,
        sha256=d.sha256_file(path),
    )


def analyst_artifact_statuses() -> list[ArtifactStatus]:
    base = paths.analyst_artifacts()
    return [artifact_status(base / name) for name in ANALYST_ARTIFACT_NAMES]


def scientist_artifact_statuses() -> list[ArtifactStatus]:
    base = paths.scientist_artifacts()
    return [artifact_status(base / name) for name in SCIENTIST_ARTIFACT_NAMES]


# --------------------------------------------------------------------------- #
# Latest Flow run (manifest / failure report)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatestRun:
    """The most recently written run manifest or failure report, if any."""

    kind: str  # "success" | "failed" | "corrupt" | "none"
    run_id: str | None
    path: str | None
    payload: dict[str, Any] | None
    modified_at: str | None


def find_latest_run(runs_dir: Path | None = None) -> LatestRun:
    """Return the newest `run_manifest.json` / `failure_report.json`, by mtime.

    Never raises: a missing runs directory, an empty one, or a corrupt JSON
    file all degrade to a well-typed result the UI can render a concise
    message for.
    """
    base = runs_dir if runs_dir is not None else paths.runs_artifacts()
    if not base.exists():
        return LatestRun(kind="none", run_id=None, path=None, payload=None, modified_at=None)

    candidates: list[tuple[float, str, Path]] = []
    for pattern, kind in (("*/run_manifest.json", "success"), ("*/failure_report.json", "failed")):
        for p in base.glob(pattern):
            if p.is_file() and p.stat().st_size > 0:
                candidates.append((p.stat().st_mtime, kind, p))

    if not candidates:
        return LatestRun(kind="none", run_id=None, path=None, payload=None, modified_at=None)

    mtime, kind, path = max(candidates, key=lambda c: c[0])
    modified_at = datetime.fromtimestamp(mtime, tz=UTC).isoformat()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return LatestRun(
            kind="corrupt",
            run_id=path.parent.name,
            path=str(path),
            payload=None,
            modified_at=modified_at,
        )
    run_id = str(payload.get("run_id") or path.parent.name)
    return LatestRun(
        kind=kind, run_id=run_id, path=str(path), payload=payload, modified_at=modified_at
    )


# --------------------------------------------------------------------------- #
# Analyst artifact readers (EDA section)
# --------------------------------------------------------------------------- #
def read_dataset_contract() -> dict[str, Any]:
    return _read_json(paths.analyst_artifacts() / "dataset_contract.json")


def read_insights() -> str:
    return _read_text(paths.analyst_artifacts() / "insights.md")


def read_eda_report_html() -> str:
    return _read_text(paths.analyst_artifacts() / "eda_report.html")


def read_eda_metrics() -> dict[str, Any]:
    return _read_json(paths.analyst_artifacts() / "eda" / "eda_metrics.json")


def read_eda_figure_manifest() -> dict[str, Any]:
    return _read_json(paths.analyst_artifacts() / "eda" / "figure_manifest.json")


_EDA_TABLE_NAMES: tuple[str, ...] = (
    "main_category_frequency",
    "transition_matrix_pct",
    "clicks_per_session_buckets",
    "top_countries",
    "month_coverage",
)


def read_eda_table(name: str) -> pd.DataFrame:
    if name not in _EDA_TABLE_NAMES:
        raise ValueError(f"unknown EDA table '{name}'; expected one of {_EDA_TABLE_NAMES}")
    path = paths.analyst_artifacts() / "eda" / "tables" / f"{name}.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise ArtifactMissingError(f"missing or empty artifact: {path}")
    return pd.read_csv(path)


def extract_markdown_section(markdown: str, heading: str) -> str:
    """Return the body of one `## {heading}` section (up to the next `## `)."""
    marker = f"## {heading}"
    start = markdown.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = markdown.find("\n## ", start)
    body = markdown[start:] if end == -1 else markdown[start:end]
    return body.strip()


# --------------------------------------------------------------------------- #
# Scientist artifact readers (Model evaluation section)
# --------------------------------------------------------------------------- #
def read_scientist_metrics() -> dict[str, Any]:
    return _read_json(paths.scientist_artifacts() / "metrics.json")


def read_model_metadata() -> dict[str, Any]:
    return _read_json(paths.scientist_artifacts() / "model_metadata.json")


def read_model_card() -> str:
    return _read_text(paths.scientist_artifacts() / "model_card.md")


def read_evaluation_report() -> str:
    return _read_text(paths.scientist_artifacts() / "evaluation_report.md")


def read_feature_schema() -> dict[str, Any]:
    return _read_json(paths.scientist_artifacts() / "feature_schema.json")


def weakest_test_class(metrics: dict[str, Any]) -> str:
    """The class code with the lowest test F1 — reuses the report's own selection."""
    from retail_clickstream_ai.reporting.model_report import _weakest_test_class

    return str(_weakest_test_class(metrics))


# --------------------------------------------------------------------------- #
# Trusted model loading (never an uploaded/unverified pickle)
# --------------------------------------------------------------------------- #
def verify_model_artifact() -> tuple[bool, str, str]:
    """Hash-check `model.joblib` against `model_metadata.json` WITHOUT loading it."""
    metadata = read_model_metadata()
    model_path = paths.scientist_artifacts() / "model.joblib"
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise ArtifactMissingError(f"missing or empty artifact: {model_path}")
    observed = d.sha256_file(model_path)
    expected = str(metadata.get("artifact_sha256", ""))
    return observed == expected, observed, expected


def load_verified_model() -> Any:
    """Verify `model.joblib`'s hash, then load it. Never loads an unverified file."""
    ok, observed, expected = verify_model_artifact()
    if not ok:
        raise ModelIntegrityError(
            f"model.joblib hash {observed} does not match model_metadata.json "
            f"artifact_sha256 {expected}; refusing to load an untrusted artifact"
        )
    # Safe: the hash pre-check above proves this is the trusted, repo-produced
    # artifact — the same guarantee `flow.py`'s model gate enforces.
    return joblib.load(paths.scientist_artifacts() / "model.joblib")


# --------------------------------------------------------------------------- #
# Prediction form contract (built from model_metadata.json + dataset_contract.json)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldSpec:
    """One prediction-form field, derived from the trained pipeline's contract."""

    name: str
    label: str
    kind: str  # "categorical" | "numeric"
    group: str  # "current" | "history"
    help: str
    options: tuple[tuple[int, str], ...] | None = None  # (code, display label)
    min_value: int | None = None
    max_value: int | None = None
    default: int = 0


def _categorical_field(
    name: str,
    label: str,
    group: str,
    contract: DatasetContract,
    *,
    extra: tuple[tuple[int, str], ...] = (),
) -> FieldSpec:
    columns = {c.name: c for c in contract.columns}
    col = columns[name]
    lo, hi = col.value_range if col.value_range is not None else (None, None)
    if col.allowed_values is not None:
        values: tuple[int, ...] = col.allowed_values
    elif lo is not None and hi is not None:
        # Some contract columns (e.g. country) encode a discrete domain as a
        # closed range rather than an explicit allowed-values list.
        values = tuple(range(lo, hi + 1))
    else:
        raise ValueError(f"column '{name}' has no bounded domain to build a dropdown from")
    options = extra + tuple(
        (v, codebook.label_with_code(name, v) if codebook.has_labels(name) else str(v))
        for v in values
    )
    default = options[0][0] if options else 0
    return FieldSpec(
        name=name,
        label=label,
        kind="categorical",
        group=group,
        help=col.description,
        options=options,
        default=default,
    )


def prediction_field_specs() -> list[FieldSpec]:
    """Build the prediction form's fields from the model's own feature list.

    Iterates ``model_metadata.json#feature_list`` (the exact predictor set the
    trained pipeline expects) rather than a hard-coded list, so the form always
    matches the deployed model. Raw dataset columns get their domain from
    ``dataset_contract.json``; the two engineered session-history predictors
    (``prev_main_category``, and friends) have no raw-column entry, so their
    domain is documented here from how ``pipeline/features.py`` builds them.
    """
    metadata = read_model_metadata()
    contract = DatasetContract.build()
    specs: list[FieldSpec] = []
    for name in metadata["feature_list"]:
        if name == feat.CURRENT_CATEGORY:
            base = _categorical_field(
                d.MAIN_CATEGORY, "Current click's main category", "current", contract
            )
            specs.append(
                FieldSpec(
                    name=name,
                    label=base.label,
                    kind="categorical",
                    group="current",
                    help=base.help,
                    options=base.options,
                    default=base.default,
                )
            )
        elif name == feat.PREV_CATEGORY:
            base = _categorical_field(
                d.MAIN_CATEGORY,
                "Previous click's category (this session)",
                "history",
                contract,
                extra=(
                    (feat.PREV_CATEGORY_SENTINEL, "None — this is the first click of the session"),
                ),
            )
            specs.append(
                FieldSpec(
                    name=name,
                    label=base.label,
                    kind="categorical",
                    group="history",
                    help=(
                        "Category of the click immediately before this one in the same "
                        "session; 0 means there was no previous click (first click)."
                    ),
                    options=base.options,
                    default=feat.PREV_CATEGORY_SENTINEL,
                )
            )
        elif name in (d.COLOUR, d.LOCATION, d.MODEL_PHOTOGRAPHY, d.PRICE_2, d.PAGE, d.COUNTRY):
            specs.append(
                _categorical_field(name, name.replace("_", " ").title(), "current", contract)
            )
        elif name == d.PRICE:
            col = next(c for c in contract.columns if c.name == d.PRICE)
            lo = col.value_range[0] if col.value_range and col.value_range[0] is not None else 1
            specs.append(
                FieldSpec(
                    name=name,
                    label="Price (US$)",
                    kind="numeric",
                    group="current",
                    help=col.description,
                    min_value=int(lo),
                    max_value=1000,
                    default=43,
                )
            )
        elif name == feat.CLICKS_SO_FAR:
            specs.append(
                FieldSpec(
                    name=name,
                    label="Clicks so far this session (including this one)",
                    kind="numeric",
                    group="history",
                    help="Position of the current click within the session (1 = first click).",
                    min_value=1,
                    max_value=500,
                    default=1,
                )
            )
        elif name == feat.DISTINCT_CATEGORIES_SO_FAR:
            specs.append(
                FieldSpec(
                    name=name,
                    label="Distinct categories seen so far (including this click)",
                    kind="numeric",
                    group="history",
                    help="How many of the 4 main categories have appeared up to and including "
                    "this click.",
                    min_value=1,
                    max_value=4,
                    default=1,
                )
            )
        elif name == feat.COUNT_CURRENT_CATEGORY_SO_FAR:
            specs.append(
                FieldSpec(
                    name=name,
                    label="Times the current category seen so far (including this click)",
                    kind="numeric",
                    group="history",
                    help="How many clicks so far (including this one) were in the current "
                    "category.",
                    min_value=1,
                    max_value=500,
                    default=1,
                )
            )
        else:  # pragma: no cover - guards silent drift if the trained feature set changes
            raise ValueError(f"no prediction-form mapping is defined for predictor '{name}'")
    return specs


def validate_prediction_inputs(values: dict[str, int]) -> ValidationReport:
    """Validate raw form values against the same contract the model was trained on."""
    report = ValidationReport()
    specs = {s.name: s for s in prediction_field_specs()}

    for name, spec in specs.items():
        value = values.get(name)
        if value is None:
            report.add(
                ValidationIssue(
                    rule="missing_field",
                    column=name,
                    message="required field is missing",
                    observed="<missing>",
                    expected=spec.label,
                )
            )
            continue
        if spec.kind == "categorical":
            allowed = [code for code, _ in (spec.options or ())]
            if int(value) not in allowed:
                report.add(
                    ValidationIssue(
                        rule="invalid_category",
                        column=name,
                        message="value is outside the allowed category codes",
                        observed=str(value),
                        expected=str(allowed),
                    )
                )
        else:
            if spec.min_value is not None and int(value) < spec.min_value:
                report.add(
                    ValidationIssue(
                        rule="out_of_range",
                        column=name,
                        message="value is below the minimum",
                        observed=str(value),
                        expected=f">= {spec.min_value}",
                    )
                )
            if spec.max_value is not None and int(value) > spec.max_value:
                report.add(
                    ValidationIssue(
                        rule="out_of_range",
                        column=name,
                        message="value is above the maximum",
                        observed=str(value),
                        expected=f"<= {spec.max_value}",
                    )
                )

    if report.ok:
        clicks = int(values[feat.CLICKS_SO_FAR])
        distinct = int(values[feat.DISTINCT_CATEGORIES_SO_FAR])
        count_cur = int(values[feat.COUNT_CURRENT_CATEGORY_SO_FAR])
        prev = int(values[feat.PREV_CATEGORY])
        if distinct > clicks:
            report.add(
                ValidationIssue(
                    rule="history_inconsistent",
                    column=feat.DISTINCT_CATEGORIES_SO_FAR,
                    message="distinct categories so far cannot exceed clicks so far",
                    observed=str(distinct),
                    expected=f"<= {clicks}",
                )
            )
        if count_cur > clicks:
            report.add(
                ValidationIssue(
                    rule="history_inconsistent",
                    column=feat.COUNT_CURRENT_CATEGORY_SO_FAR,
                    message="count of the current category so far cannot exceed clicks so far",
                    observed=str(count_cur),
                    expected=f"<= {clicks}",
                )
            )
        if clicks == 1 and prev != feat.PREV_CATEGORY_SENTINEL:
            report.add(
                ValidationIssue(
                    rule="history_inconsistent",
                    column=feat.PREV_CATEGORY,
                    message="the first click of a session cannot have a previous category",
                    observed=str(prev),
                    expected=str(feat.PREV_CATEGORY_SENTINEL),
                )
            )
        if clicks > 1 and prev == feat.PREV_CATEGORY_SENTINEL:
            report.add(
                ValidationIssue(
                    rule="history_inconsistent",
                    column=feat.PREV_CATEGORY,
                    message="a session with more than one click so far must have a previous "
                    "category",
                    observed=str(feat.PREV_CATEGORY_SENTINEL),
                    expected="1-4",
                )
            )
    return report


def build_prediction_frame(values: dict[str, int]) -> pd.DataFrame:
    """Build the exact one-row input frame the trained pipeline expects."""
    metadata = read_model_metadata()
    columns: list[str] = list(metadata["feature_list"])
    dtypes: dict[str, str] = metadata["input_schema"]["dtypes"]
    row = {c: [int(values[c])] for c in columns}
    frame = pd.DataFrame(row)
    for c in columns:
        frame[c] = frame[c].astype(dtypes.get(c, "int64"))
    return frame[columns]


@dataclass(frozen=True)
class PredictionResult:
    """A single prediction: the top class plus the full probability ranking."""

    predicted_class: int
    predicted_label: str
    probabilities: tuple[tuple[int, str, float], ...]  # (class, label, prob), sorted desc


def predict_next_category(values: dict[str, int], model: Any | None = None) -> PredictionResult:
    """Validate, build the model's exact input row, and predict.

    Raises :class:`~retail_clickstream_ai.validation.errors.ContractValidationError`
    on invalid input (never predicts on out-of-contract values), and
    :class:`ModelIntegrityError` if the model artifact fails its hash check.
    Pass a pre-loaded ``model`` (e.g. the UI's ``st.cache_resource``-cached
    instance) to avoid re-verifying and reloading the pickle on every call;
    when omitted, a freshly hash-verified model is loaded.
    """
    report = validate_prediction_inputs(values)
    if not report.ok:
        raise ContractValidationError(report)

    frame = build_prediction_frame(values)
    if model is None:
        model = load_verified_model()
    metadata = read_model_metadata()

    proba = model.predict_proba(frame)[0]
    index_to_class = {
        int(k): int(v) for k, v in metadata["class_mapping"]["index_to_class"].items()
    }
    class_to_label = {
        int(k): str(v) for k, v in metadata["class_mapping"]["class_to_label"].items()
    }

    ranked = sorted(
        (
            (index_to_class[i], class_to_label[index_to_class[i]], float(p))
            for i, p in enumerate(proba)
        ),
        key=lambda t: -t[2],
    )
    top = ranked[0]
    return PredictionResult(
        predicted_class=top[0], predicted_label=top[1], probabilities=tuple(ranked)
    )
