"""Deterministic cleaning of the raw clickstream frame.

Stage 3 scope. This module turns a bytes-verified *raw* frame into the cleaned
dataset artifact, recording **every** transformation and its affected-row count
along the way. Its cardinal rule mirrors the runtime Data Engineer prompt:

    Never silently impute, drop, coerce, or clip an invalid value.

So cleaning is deliberately *non-destructive*. It first validates the raw frame
against the dataset contract and **fails closed** if anything is wrong (rather
than "fixing" it), then performs only structural, reversible operations:
derive the verified composite session key, sort into canonical click order, and
freeze a stable column order. The row count in equals the row count out — always.

The cleaned CSV is serialized with fixed parameters (comma delimiter, ``\\n``
line ending, UTF-8, no index) so the same input yields byte-identical output and
a reproducible SHA-256. That hash is pinned in the dataset contract; a validator
(:mod:`retail_clickstream_ai.validation.artifacts`) proves the produced file
matches it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.validation.contract import DatasetContract
from retail_clickstream_ai.validation.raw import validate_dataframe

# Serialization facts for the cleaned artifact (deterministic across machines).
CLEAN_DELIMITER = ","
CLEAN_ENCODING = "utf-8"
CLEAN_LINE_ENDING = "\n"


@dataclass(frozen=True)
class TransformationRule:
    """One recorded cleaning step and how many rows it affected.

    ``affected_rows`` is the number of rows the step changed or acted on. It is
    recorded even when zero, so the audit is a complete, honest ledger rather
    than a list of only the interesting steps.
    """

    name: str
    description: str
    affected_rows: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "affected_rows": self.affected_rows,
            "detail": self.detail,
        }


@dataclass
class TransformationAudit:
    """The complete, machine-readable ledger of a cleaning run.

    Records the input/output shape, whether the pre-clean validation passed,
    the count of fatal validation issues, and every transformation rule with its
    affected-row count. A caller can prove no rows were lost by comparing
    ``input_rows`` and ``output_rows``.
    """

    input_rows: int
    input_columns: int
    output_rows: int
    output_columns: int
    validation_passed: bool
    fatal_issue_count: int
    rules: list[TransformationRule]

    @property
    def rows_preserved(self) -> bool:
        """True when cleaning neither added nor dropped a row."""
        return self.input_rows == self.output_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "input_columns": self.input_columns,
            "output_rows": self.output_rows,
            "output_columns": self.output_columns,
            "rows_preserved": self.rows_preserved,
            "validation_passed": self.validation_passed,
            "fatal_issue_count": self.fatal_issue_count,
            "rule_count": len(self.rules),
            "rules": [r.to_dict() for r in self.rules],
        }


def _column_rename_map() -> dict[str, str]:
    """Source header -> normalized name, for the columns whose name changed."""
    return {
        src: norm
        for src, norm in zip(d.SOURCE_COLUMNS, d.NORMALIZED_COLUMNS, strict=True)
        if src != norm
    }


def clean_dataframe(
    raw_df: pd.DataFrame,
    contract: DatasetContract,
    *,
    require_all_months: bool = True,
) -> tuple[pd.DataFrame, TransformationAudit]:
    """Clean a normalized raw frame; return the cleaned frame and its audit.

    Steps (all non-destructive):

    1. Validate against ``contract`` and raise :class:`ContractValidationError`
       on any fatal issue — cleaning never proceeds on invalid data.
    2. Confirm the duplicate policy (reject full-row duplicates; the source has
       none, so this is a zero-count assertion here).
    3. Derive the composite ``session_key``.
    4. Sort deterministically into canonical click order.
    5. Freeze the stable output column order.

    Raises
    ------
    ContractValidationError
        If the raw frame fails a fatal contract rule. The full report is
        attached so the caller can write a failure artifact.
    """
    input_rows, input_columns = int(raw_df.shape[0]), int(raw_df.shape[1])

    # 1. Fail closed on invalid data — do not "repair" anything.
    report = validate_dataframe(raw_df, contract, require_all_months=require_all_months)
    report.raise_if_failed()

    rules: list[TransformationRule] = [
        TransformationRule(
            name="parse_and_validate",
            description=(
                "Parse with the verified delimiter/encoding and validate every "
                "column type, range, and category against the dataset contract. "
                "Fail closed on any fatal issue; never coerce or clip."
            ),
            affected_rows=0,
            detail={"validation_issue_count": len(report.issues)},
        ),
        TransformationRule(
            name="normalize_column_names",
            description="Rename source headers to snake_case by the fixed mapping.",
            affected_rows=0,
            detail={"renamed": _column_rename_map()},
        ),
    ]

    # 2. Duplicate policy: reject full-row duplicates (validation already fails
    #    when any exist, so this is a confirmed zero here).
    duplicate_rows = int(raw_df.duplicated().sum())
    rules.append(
        TransformationRule(
            name="reject_full_row_duplicates",
            description="Full-row duplicates are rejected, never silently dropped.",
            affected_rows=duplicate_rows,
            detail={"policy": "reject"},
        )
    )

    # 3. Derive the verified composite session key.
    keyed = d.add_session_key(raw_df).reset_index(drop=True)
    session_count = int(keyed[d.SESSION_KEY].nunique())
    rules.append(
        TransformationRule(
            name="derive_session_key",
            description="Add session_key = 'YYYY-MM-DD-s<session_id>' (canonical).",
            affected_rows=input_rows,
            detail={"session_count": session_count},
        )
    )

    # 4. Sort into canonical click order, counting how many rows actually moved.
    keyed = keyed.assign(_orig_pos=np.arange(input_rows))
    ordered = d.canonical_sort(keyed)
    rows_moved = int((ordered["_orig_pos"].to_numpy() != np.arange(input_rows)).sum())
    ordered = ordered.drop(columns="_orig_pos")
    rules.append(
        TransformationRule(
            name="canonical_sort",
            description=(
                "Sort by (year, month, day, session_id, order) ascending, stably. "
                "Defines click order regardless of on-disk row order."
            ),
            affected_rows=rows_moved,
            detail={"sort_columns": list(d.SORT_COLUMNS)},
        )
    )

    # 5. Freeze the stable output column order.
    clean_df = ordered[list(d.REQUIRED_OUTPUT_COLUMNS)].reset_index(drop=True)
    rules.append(
        TransformationRule(
            name="order_output_columns",
            description="Emit the fixed output column order (schema + session_key).",
            affected_rows=0,
            detail={"columns": list(d.REQUIRED_OUTPUT_COLUMNS)},
        )
    )

    audit = TransformationAudit(
        input_rows=input_rows,
        input_columns=input_columns,
        output_rows=int(clean_df.shape[0]),
        output_columns=int(clean_df.shape[1]),
        validation_passed=report.ok,
        fatal_issue_count=len(report.fatal_issues),
        rules=rules,
    )
    # Invariant: cleaning is non-destructive.
    if not audit.rows_preserved:
        raise AssertionError(
            f"Cleaning changed the row count ({input_rows} -> {audit.output_rows}); "
            "this must never happen."
        )
    return clean_df, audit


def read_clean_csv(path: str | Path) -> pd.DataFrame:
    """Read a cleaned CSV back with the correct string dtypes.

    Both the product code and the derived ``session_key`` are text; everything
    else is integer. Used by the artifact validator to re-check a produced file.
    """
    string_cols = dict.fromkeys((d.CLOTHING_MODEL, d.SESSION_KEY), "string")
    return pd.read_csv(path, sep=CLEAN_DELIMITER, dtype=string_cols)


def write_clean_csv(clean_df: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Serialize the cleaned frame deterministically and return its file stats.

    Fixed serialization (comma delimiter, ``\\n`` line ending, UTF-8, no index)
    makes the bytes — and therefore the SHA-256 — reproducible across machines.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(
        out,
        index=False,
        sep=CLEAN_DELIMITER,
        lineterminator=CLEAN_LINE_ENDING,
        encoding=CLEAN_ENCODING,
    )
    return {
        "path": str(out),
        "sha256": d.sha256_file(out),
        "size_bytes": int(os.path.getsize(out)),
        "row_count": int(clean_df.shape[0]),
        "column_count": int(clean_df.shape[1]),
    }
