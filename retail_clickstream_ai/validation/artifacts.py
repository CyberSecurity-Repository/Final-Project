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
