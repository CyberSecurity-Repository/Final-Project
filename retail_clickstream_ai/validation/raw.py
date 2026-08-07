"""Deterministic validation of a raw/normalized clickstream frame.

:func:`validate_dataframe` checks a normalized DataFrame against a
:class:`~retail_clickstream_ai.validation.contract.DatasetContract` and returns a
:class:`~retail_clickstream_ai.validation.errors.ValidationReport`. It never
stops at the first problem — every fatal issue is collected so a caller sees the
whole picture, and every issue names the column, the rule, the observed value,
and the expected condition.

Click order is validated *after* the contract's deterministic sort, so rows that
are merely stored out of order are fine; only a genuinely broken sequence (a gap
or a duplicate within a session) fails.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.validation.contract import ColumnSpec, DatasetContract
from retail_clickstream_ai.validation.errors import ValidationIssue, ValidationReport

_SAMPLE_LIMIT = 8


def _sample(values: Iterable[Any], limit: int = _SAMPLE_LIMIT) -> str:
    """Render a small, sorted, capped sample of offending values."""
    unique = sorted({str(v) for v in values})
    shown = unique[:limit]
    extra = len(unique) - len(shown)
    body = ", ".join(shown)
    return f"{{{body}}}" + (f" (+{extra} more)" if extra > 0 else "")


def _dtype_matches(series: pd.Series, dtype: str) -> bool:
    if dtype == "int64":
        return bool(pd.api.types.is_integer_dtype(series))
    if dtype == "string":
        return bool(pd.api.types.is_string_dtype(series)) or series.dtype == object
    return True


def _check_column(df: pd.DataFrame, spec: ColumnSpec, report: ValidationReport) -> None:
    """Presence, dtype, nulls, and the column's value constraint."""
    if spec.name not in df.columns:
        report.add(
            ValidationIssue(
                rule="missing_column",
                column=spec.name,
                message="required column is absent",
                observed="<missing>",
                expected=f"column '{spec.name}' (source '{spec.source_name}')",
            )
        )
        return

    series = df[spec.name]

    dtype_ok = _dtype_matches(series, spec.dtype)
    if not dtype_ok:
        report.add(
            ValidationIssue(
                rule="dtype",
                column=spec.name,
                message="unexpected column dtype",
                observed=str(series.dtype),
                expected=spec.dtype,
            )
        )

    null_count = int(series.isna().sum())
    if not spec.nullable and null_count > 0:
        report.add(
            ValidationIssue(
                rule="null_values",
                column=spec.name,
                message="nulls present in a non-nullable column",
                observed=f"{null_count} null(s)",
                expected="0 nulls",
            )
        )

    # Value-domain checks rely on a correctly-typed column; skip them when the
    # dtype is already wrong to avoid noisy, misleading secondary errors.
    non_null = series.dropna()
    if not dtype_ok or non_null.empty:
        return

    if spec.allowed_values is not None:
        allowed = set(spec.allowed_values)
        offending = [v for v in non_null.unique() if v not in allowed]
        if offending:
            report.add(
                ValidationIssue(
                    rule="allowed_values",
                    column=spec.name,
                    message="value(s) outside the allowed set",
                    observed=_sample(offending),
                    expected=f"one of {sorted(allowed)}",
                )
            )

    if spec.value_range is not None:
        low, high = spec.value_range
        numeric = pd.to_numeric(non_null, errors="coerce")
        below = numeric[numeric < low] if low is not None else []
        above = numeric[numeric > high] if high is not None else []
        out_of_range = [*list(below), *list(above)]
        if out_of_range:
            report.add(
                ValidationIssue(
                    rule="range",
                    column=spec.name,
                    message="value(s) outside the allowed range",
                    observed=_sample(out_of_range),
                    expected=f"[{low}, {high}]",
                )
            )

    if spec.pattern is not None:
        compiled = re.compile(spec.pattern)
        offending = [v for v in non_null.unique() if not compiled.match(str(v))]
        if offending:
            report.add(
                ValidationIssue(
                    rule="pattern",
                    column=spec.name,
                    message="value(s) do not match the required pattern",
                    observed=_sample(offending),
                    expected=spec.pattern,
                )
            )


def _check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        report.add(
            ValidationIssue(
                rule="duplicate_rows",
                message="full-row duplicates present",
                observed=f"{dup_count} duplicate row(s)",
                expected="0 duplicate rows",
            )
        )


def _session_columns_ready(df: pd.DataFrame, contract: DatasetContract) -> bool:
    """True when the session/order columns exist and are integer-typed."""
    needed = [*contract.session_key_columns, d.ORDER]
    for col in needed:
        if col not in df.columns or not pd.api.types.is_integer_dtype(df[col]):
            return False
    return True


def _check_session_order(
    df: pd.DataFrame, contract: DatasetContract, report: ValidationReport
) -> None:
    """Validate click-order integrity within each *verified* session.

    Uses the composite session key so a bare ``session_id`` reused on another day
    yields two distinct sessions (no false collision). After the deterministic
    sort, each session's ``order`` must be a contiguous ``1..n`` run.
    """
    if not _session_columns_ready(df, contract):
        return  # missing/mistyped columns are already reported elsewhere.

    keyed = d.add_session_key(df).sort_values([d.SESSION_KEY, d.ORDER], kind="mergesort")

    dup_mask = keyed.duplicated(subset=[d.SESSION_KEY, d.ORDER], keep=False)
    if bool(dup_mask.any()):
        offenders = keyed.loc[dup_mask, d.SESSION_KEY].unique()
        report.add(
            ValidationIssue(
                rule="duplicate_click_order",
                column=d.ORDER,
                message="duplicate (session_key, order) pairs",
                observed=f"sessions {_sample(offenders)}",
                expected="each order unique within a session",
            )
        )

    bad_sessions: list[str] = []
    for key, group in keyed.groupby(d.SESSION_KEY)[d.ORDER]:
        values = sorted(int(v) for v in group.tolist())
        if values != list(range(1, len(values) + 1)):
            bad_sessions.append(str(key))
    if bad_sessions:
        report.add(
            ValidationIssue(
                rule="click_order_sequence",
                column=d.ORDER,
                message="session click order is not a contiguous 1..n sequence",
                observed=f"sessions {_sample(bad_sessions)}",
                expected="order == 1..n per session (no gaps or duplicates)",
            )
        )


def _check_month_coverage(
    df: pd.DataFrame, contract: DatasetContract, report: ValidationReport
) -> None:
    if d.MONTH not in df.columns or not pd.api.types.is_integer_dtype(df[d.MONTH]):
        return
    present = {int(m) for m in df[d.MONTH].dropna().unique()}
    missing = [m for m in contract.required_months if m not in present]
    if missing:
        report.add(
            ValidationIssue(
                rule="month_coverage",
                column=d.MONTH,
                message="required split month(s) absent — chronological split infeasible",
                observed=f"present {sorted(present)}",
                expected=f"all of {contract.required_months}",
            )
        )


def validate_dataframe(
    df: pd.DataFrame,
    contract: DatasetContract,
    *,
    require_all_months: bool = True,
) -> ValidationReport:
    """Validate a normalized frame against ``contract`` and return a report.

    Set ``require_all_months=False`` for small fixtures that intentionally cover
    only a few months (the chronological-split requirement still holds for the
    full dataset).
    """
    report = ValidationReport()
    for spec in contract.columns:
        _check_column(df, spec, report)
    _check_duplicates(df, report)
    _check_session_order(df, contract, report)
    if require_all_months:
        _check_month_coverage(df, contract, report)
    return report
