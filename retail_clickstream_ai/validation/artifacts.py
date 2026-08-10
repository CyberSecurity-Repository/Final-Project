"""Deterministic validation of the produced Analyst artifacts.

Three layers, all returning an aggregated
:class:`~retail_clickstream_ai.validation.errors.ValidationReport`:

* :func:`validate_clean_frame` — the cleaned frame conforms to the contract
  (exact output columns in order, plus all raw-frame checks);
* :func:`validate_clean_file_against_contract` — the produced ``clean_data.csv``
  matches the contract, optionally pinning the SHA-256/size/shape to the
  verified constants (the "contract matches the cleaned CSV" gate);
* :func:`validate_analyst_artifacts` — all four required artifacts exist, are
  non-empty, agree with the contract, and the reports contain their required
  sections.

An LLM never decides any of these — a failed check ends the Analyst run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from retail_clickstream_ai.pipeline import cleaning
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.validation.contract import DatasetContract
from retail_clickstream_ai.validation.errors import ValidationIssue, ValidationReport
from retail_clickstream_ai.validation.raw import validate_dataframe

# Section anchors every rendered EDA HTML report must contain.
REQUIRED_HTML_SECTIONS: tuple[str, ...] = (
    'id="methodology"',
    'id="dataset-provenance"',
    'id="data-quality"',
    'id="descriptive-tables"',
    'id="figures"',
    'id="findings"',
    'id="limitations"',
)

# Headings (and the mandatory caveat) every insights.md must contain.
REQUIRED_INSIGHTS_MARKERS: tuple[str, ...] = (
    "## What this is",
    "## Data quality",
    "## Key findings",
    "## Limitations",
    "2008",
)


def validate_clean_frame(
    clean_df: pd.DataFrame,
    contract: DatasetContract,
    *,
    require_all_months: bool = True,
) -> ValidationReport:
    """Validate a cleaned frame: exact output columns + all contract checks."""
    report = validate_dataframe(clean_df, contract, require_all_months=require_all_months)

    expected = contract.cleaned_columns
    actual = list(clean_df.columns)
    if actual != expected:
        report.add(
            ValidationIssue(
                rule="output_columns",
                message="cleaned column set/order does not match the contract",
                observed=str(actual),
                expected=str(expected),
            )
        )
    if d.SESSION_KEY not in clean_df.columns:
        report.add(
            ValidationIssue(
                rule="missing_session_key",
                column=d.SESSION_KEY,
                message="cleaned data must expose the derived session key",
                observed="<missing>",
                expected=d.SESSION_KEY,
            )
        )
    return report


def _add_file_presence(path: Path, name: str, report: ValidationReport) -> bool:
    """Record presence/non-empty; return True when the file is usable."""
    if not path.exists():
        report.add(
            ValidationIssue(
                rule="artifact_missing",
                column=name,
                message="required artifact is absent",
                observed="<missing>",
                expected=str(path),
            )
        )
        return False
    if os.path.getsize(path) == 0:
        report.add(
            ValidationIssue(
                rule="artifact_empty",
                column=name,
                message="required artifact is empty",
                observed="0 bytes",
                expected="non-empty file",
            )
        )
        return False
    return True


def validate_clean_file_against_contract(
    clean_path: str | Path,
    contract: DatasetContract,
    *,
    pin_hash: bool = True,
    require_all_months: bool = True,
) -> ValidationReport:
    """Validate the produced ``clean_data.csv`` against the contract.

    When ``pin_hash`` is true, the file's SHA-256, byte size, and shape must
    equal the verified constants pinned in the contract (used for the real,
    committed artifact). Set it false for a fixture that intentionally differs
    from the pinned production data; the schema/column/session checks still run.
    """
    report = ValidationReport()
    path = Path(clean_path)
    if not _add_file_presence(path, "clean_data.csv", report):
        return report

    frame = cleaning.read_clean_csv(path)
    report.issues.extend(
        validate_clean_frame(frame, contract, require_all_months=require_all_months).issues
    )

    if pin_hash:
        actual_sha = d.sha256_file(path)
        if actual_sha != contract.cleaned_sha256:
            report.add(
                ValidationIssue(
                    rule="clean_sha256",
                    column="clean_data.csv",
                    message="cleaned file hash does not match the contract",
                    observed=actual_sha,
                    expected=contract.cleaned_sha256,
                )
            )
        actual_size = os.path.getsize(path)
        if actual_size != contract.cleaned_size_bytes:
            report.add(
                ValidationIssue(
                    rule="clean_size",
                    column="clean_data.csv",
                    message="cleaned file size does not match the contract",
                    observed=str(actual_size),
                    expected=str(contract.cleaned_size_bytes),
                )
            )
        if frame.shape[0] != contract.cleaned_row_count:
            report.add(
                ValidationIssue(
                    rule="clean_row_count",
                    column="clean_data.csv",
                    message="cleaned row count does not match the contract",
                    observed=str(frame.shape[0]),
                    expected=str(contract.cleaned_row_count),
                )
            )
        if frame.shape[1] != contract.cleaned_column_count:
            report.add(
                ValidationIssue(
                    rule="clean_column_count",
                    column="clean_data.csv",
                    message="cleaned column count does not match the contract",
                    observed=str(frame.shape[1]),
                    expected=str(contract.cleaned_column_count),
                )
            )
    return report


def _check_text_markers(
    path: Path, name: str, markers: tuple[str, ...], report: ValidationReport
) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [m for m in markers if m not in text]
    if missing:
        report.add(
            ValidationIssue(
                rule="missing_report_section",
                column=name,
                message="report is missing required section(s)/marker(s)",
                observed=f"missing {missing}",
                expected=f"all of {list(markers)}",
            )
        )


def validate_scientist_artifacts(
    *,
    features_path: str | Path,
    model_path: str | Path,
    eval_report_path: str | Path,
    model_card_path: str | Path,
    metrics_path: str | Path,
    metadata_path: str | Path,
) -> ValidationReport:
    """Validate the four required Scientist artifacts and their agreement.

    Checks: presence + non-empty for each required file; features table exposes
    the predictor/split/target columns; both reports contain their required
    sections; the trusted model round-trips (loads and predicts on the test
    split with a 4-class probability output); ``model_metadata.json`` hashes match
    the on-disk metrics/model; and both reports are exactly the deterministic
    render of ``metrics.json`` (so every reported number agrees with machine
    output). An LLM never decides any of these.
    """
    import json

    from retail_clickstream_ai.pipeline import features as feat
    from retail_clickstream_ai.reporting.model_report import (
        REQUIRED_EVAL_REPORT_MARKERS,
        REQUIRED_MODEL_CARD_MARKERS,
        render_evaluation_report,
        render_model_card,
    )

    report = ValidationReport()

    features = Path(features_path)
    model = Path(model_path)
    eval_report = Path(eval_report_path)
    model_card = Path(model_card_path)
    metrics_file = Path(metrics_path)
    metadata_file = Path(metadata_path)

    features_ok = _add_file_presence(features, "features.csv", report)
    model_ok = _add_file_presence(model, "model.joblib", report)
    eval_ok = _add_file_presence(eval_report, "evaluation_report.md", report)
    card_ok = _add_file_presence(model_card, "model_card.md", report)
    metrics_ok = _add_file_presence(metrics_file, "metrics.json", report)
    metadata_ok = _add_file_presence(metadata_file, "model_metadata.json", report)

    # Feature table exposes the declared predictor/split/target columns.
    if features_ok:
        cols = set(pd.read_csv(features, nrows=0).columns)
        required = {*feat.PREDICTOR_COLUMNS, feat.SPLIT_COLUMN, feat.TARGET}
        missing = sorted(required - cols)
        if missing:
            report.add(
                ValidationIssue(
                    rule="features_schema",
                    column="features.csv",
                    message="features table is missing required columns",
                    observed=f"missing {missing}",
                    expected="all predictor + split + target columns",
                )
            )

    # Required report sections.
    if eval_ok:
        _check_text_markers(
            eval_report, "evaluation_report.md", REQUIRED_EVAL_REPORT_MARKERS, report
        )
    if card_ok:
        _check_text_markers(model_card, "model_card.md", REQUIRED_MODEL_CARD_MARKERS, report)

    # Metadata hash consistency.
    metadata: dict[str, object] = {}
    if metadata_ok:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metrics_ok and metadata.get("metrics_sha256") != d.sha256_file(metrics_file):
            report.add(
                ValidationIssue(
                    rule="metrics_hash_mismatch",
                    column="model_metadata.json",
                    message="metadata metrics_sha256 does not match metrics.json",
                    observed=str(metadata.get("metrics_sha256")),
                    expected=d.sha256_file(metrics_file),
                )
            )
        if model_ok and metadata.get("artifact_sha256") != d.sha256_file(model):
            report.add(
                ValidationIssue(
                    rule="artifact_hash_mismatch",
                    column="model_metadata.json",
                    message="metadata artifact_sha256 does not match model.joblib",
                    observed=str(metadata.get("artifact_sha256")),
                    expected=d.sha256_file(model),
                )
            )
        if metadata.get("round_trip_validated") is not True:
            report.add(
                ValidationIssue(
                    rule="round_trip_not_validated",
                    column="model_metadata.json",
                    message="model was not recorded as round-trip validated",
                    observed=str(metadata.get("round_trip_validated")),
                    expected="True",
                )
            )

    # Trusted model round-trip: load the repo-produced artifact and predict.
    if model_ok and features_ok:
        try:
            import joblib

            # Safe: loading the trusted, repo-produced artifact under validation.
            estimator = joblib.load(model)
            frame = feat.read_features_csv(features)
            test_rows = frame[frame[feat.SPLIT_COLUMN] == "test"]
            sample = test_rows.head(256) if len(test_rows) else frame.head(256)
            X = sample.loc[:, list(feat.PREDICTOR_COLUMNS)]
            preds = estimator.predict(X)
            if len(preds) != len(sample):
                raise ValueError("prediction length mismatch")
            proba = estimator.predict_proba(X)
            if proba.shape[1] != 4:
                raise ValueError(f"expected 4-class probabilities, got {proba.shape[1]}")
        except Exception as exc:  # noqa: BLE001 - any load/predict failure fails the gate
            report.add(
                ValidationIssue(
                    rule="model_round_trip_failed",
                    column="model.joblib",
                    message="trusted model failed to load or predict",
                    observed=str(exc),
                    expected="loads and predicts a 4-class probability output",
                )
            )

    # Reports are exactly the deterministic render of the machine metrics.
    if metrics_ok and eval_ok:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        expected_eval, _ = render_evaluation_report(metrics)
        if eval_report.read_text(encoding="utf-8") != expected_eval:
            report.add(
                ValidationIssue(
                    rule="evaluation_report_mismatch",
                    column="evaluation_report.md",
                    message="report is not the deterministic render of metrics.json",
                    observed="<differs>",
                    expected="render_evaluation_report(metrics.json)",
                )
            )
        if card_ok and metadata_ok:
            expected_card, _ = render_model_card(metrics, metadata)
            if model_card.read_text(encoding="utf-8") != expected_card:
                report.add(
                    ValidationIssue(
                        rule="model_card_mismatch",
                        column="model_card.md",
                        message="model card is not the deterministic render of metrics/metadata",
                        observed="<differs>",
                        expected="render_model_card(metrics.json, model_metadata.json)",
                    )
                )

    return report


def validate_analyst_artifacts(
    *,
    clean_path: str | Path,
    eda_report_path: str | Path,
    insights_path: str | Path,
    contract_path: str | Path,
    contract: DatasetContract,
    pin_hash: bool = True,
    require_all_months: bool = True,
) -> ValidationReport:
    """Validate all four required Analyst artifacts and their agreement.

    Checks: presence + non-empty for each of the four files; the cleaned CSV
    matches the contract; the on-disk contract equals the built contract; the
    HTML report and insights note contain their required sections.
    """
    report = ValidationReport()

    clean = Path(clean_path)
    html = Path(eda_report_path)
    insights = Path(insights_path)
    contract_file = Path(contract_path)

    clean_ok = _add_file_presence(clean, "clean_data.csv", report)
    html_ok = _add_file_presence(html, "eda_report.html", report)
    insights_ok = _add_file_presence(insights, "insights.md", report)
    contract_ok = _add_file_presence(contract_file, "dataset_contract.json", report)

    if clean_ok:
        report.issues.extend(
            validate_clean_file_against_contract(
                clean,
                contract,
                pin_hash=pin_hash,
                require_all_months=require_all_months,
            ).issues
        )

    if contract_ok:
        on_disk = contract_file.read_text(encoding="utf-8")
        if on_disk != DatasetContract.build().to_json():
            report.add(
                ValidationIssue(
                    rule="contract_mismatch",
                    column="dataset_contract.json",
                    message="on-disk contract does not equal the built contract",
                    observed="<differs>",
                    expected="DatasetContract.build().to_json()",
                )
            )

    if html_ok:
        _check_text_markers(html, "eda_report.html", REQUIRED_HTML_SECTIONS, report)
    if insights_ok:
        _check_text_markers(insights, "insights.md", REQUIRED_INSIGHTS_MARKERS, report)

    return report
