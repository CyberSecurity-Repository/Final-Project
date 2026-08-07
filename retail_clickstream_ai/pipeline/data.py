"""Data acquisition, inspection, and deterministic normalization.

Stage 2 scope: read the semicolon-delimited clickstream CSV exactly, normalize
its column names by a fixed algorithm, derive the *verified* composite session
key, sort rows into a canonical order, profile the frame, and expose a
``validate-raw`` / ``build-contract`` command line. All numeric work is
deterministic and testable; no LLM is involved.

Observed facts about the real file (``e-shop clothing 2008.csv``) are recorded
in the dataset contract (see :mod:`retail_clickstream_ai.validation.contract`),
not hard-coded into behavior here beyond the delimiter and the page-2 string
dtype, which are structural parsing facts.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Structural parsing facts (verified from the raw bytes).
# --------------------------------------------------------------------------- #
RAW_FILENAME = "e-shop clothing 2008.csv"
RAW_DELIMITER = ";"
RAW_ENCODING = "utf-8"  # the file is pure ASCII, a strict subset of UTF-8.

# Original header, in file order, exactly as shipped.
SOURCE_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "order",
    "country",
    "session ID",
    "page 1 (main category)",
    "page 2 (clothing model)",
    "colour",
    "location",
    "model photography",
    "price",
    "price 2",
    "page",
)

# The single non-integer source column keeps product codes like "A13" as text.
SOURCE_STRING_COLUMNS: tuple[str, ...] = ("page 2 (clothing model)",)

# --------------------------------------------------------------------------- #
# Normalized column names (the deterministic output of normalize_column_name).
# --------------------------------------------------------------------------- #
YEAR = "year"
MONTH = "month"
DAY = "day"
ORDER = "order"
COUNTRY = "country"
SESSION_ID = "session_id"
MAIN_CATEGORY = "page_1_main_category"
CLOTHING_MODEL = "page_2_clothing_model"
COLOUR = "colour"
LOCATION = "location"
MODEL_PHOTOGRAPHY = "model_photography"
PRICE = "price"
PRICE_2 = "price_2"
PAGE = "page"

NORMALIZED_COLUMNS: tuple[str, ...] = (
    YEAR,
    MONTH,
    DAY,
    ORDER,
    COUNTRY,
    SESSION_ID,
    MAIN_CATEGORY,
    CLOTHING_MODEL,
    COLOUR,
    LOCATION,
    MODEL_PHOTOGRAPHY,
    PRICE,
    PRICE_2,
    PAGE,
)

# The derived session identifier and the columns that build it.
SESSION_KEY = "session_key"
SESSION_KEY_COLUMNS: tuple[str, ...] = (YEAR, MONTH, DAY, SESSION_ID)
# Canonical row order: by session, then click order within the session.
SORT_COLUMNS: tuple[str, ...] = (YEAR, MONTH, DAY, SESSION_ID, ORDER)

# The columns a cleaned dataset must expose downstream (schema + derived key).
REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (*NORMALIZED_COLUMNS, SESSION_KEY)

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_column_name(name: str) -> str:
    """Normalize one raw header to snake_case, deterministically.

    Algorithm: strip surrounding whitespace, lowercase, replace every run of
    non-alphanumeric characters with a single underscore, then trim leading and
    trailing underscores. Examples::

        "session ID"             -> "session_id"
        "page 1 (main category)" -> "page_1_main_category"
        "price 2"                -> "price_2"
    """
    lowered = name.strip().lower()
    collapsed = _NON_ALNUM.sub("_", lowered)
    return collapsed.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with every column name run through :func:`normalize_column_name`.

    Raises ``ValueError`` if normalization collapses two headers to the same
    name (a collision that would silently drop a column).
    """
    mapping = {col: normalize_column_name(str(col)) for col in df.columns}
    normalized = list(mapping.values())
    duplicates = {n for n in normalized if normalized.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"Column normalization produced duplicate names: {sorted(duplicates)}. "
            "Two source headers normalize to the same snake_case name."
        )
    return df.rename(columns=mapping)


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    """Read the raw clickstream CSV and normalize its column names.

    Uses the verified semicolon delimiter and keeps the product-code column as
    text. No rows are dropped, coerced, or reordered here.
    """
    dtype_map = dict.fromkeys(SOURCE_STRING_COLUMNS, "string")
    df = pd.read_csv(path, sep=RAW_DELIMITER, dtype=dtype_map)
    return normalize_columns(df)


def add_session_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the derived composite ``session_key`` column.

    The key is ``"YYYY-MM-DD-s<session_id>"``. It is canonical because a bare
    ``session_id`` is only guaranteed unique *within a day*; composing it with
    the date keeps sessions distinct even if ids are ever reused across days.
    """
    missing = [c for c in SESSION_KEY_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot build session_key; missing columns: {missing}")
    key = (
        df[YEAR].astype("int64").astype(str)
        + "-"
        + df[MONTH].astype("int64").astype(str).str.zfill(2)
        + "-"
        + df[DAY].astype("int64").astype(str).str.zfill(2)
        + "-s"
        + df[SESSION_ID].astype("int64").astype(str)
    )
    out = df.copy()
    out[SESSION_KEY] = key
    return out


def canonical_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows into canonical order (session, then click order), stably.

    This is the documented deterministic sort policy: click order within a
    session is defined by the ``order`` column, regardless of how the rows were
    stored on disk. A stable mergesort makes the result reproducible.
    """
    return df.sort_values(list(SORT_COLUMNS), kind="mergesort").reset_index(drop=True)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a compact, JSON-serializable profile of a normalized frame.

    Reports shape, dtypes, null counts, full-row duplicates, per-column
    cardinality, numeric ranges, the verified session count, and the month
    distribution. Returns observed facts only — it never returns raw rows.
    """
    keyed = add_session_key(df) if SESSION_KEY not in df.columns else df
    profile: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        "full_row_duplicates": int(df.duplicated().sum()),
        "cardinality": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    profile["numeric_ranges"] = {
        c: {"min": int(df[c].min()), "max": int(df[c].max())} for c in numeric_cols
    }
    if {YEAR, MONTH, DAY, SESSION_ID}.issubset(df.columns):
        profile["session_count"] = int(keyed[SESSION_KEY].nunique())
    if MONTH in df.columns:
        counts = df[MONTH].value_counts().sort_index()
        profile["month_distribution"] = {int(k): int(v) for k, v in counts.items()}
    return profile


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _cmd_validate_raw(args: argparse.Namespace) -> int:
    # Imported lazily to keep this module's import graph free of validation.
    from retail_clickstream_ai.validation.contract import DatasetContract
    from retail_clickstream_ai.validation.raw import validate_dataframe

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        print(
            "See data/README.md for acquisition instructions "
            "(Kaggle slug: tunguz/clickstream-data-for-online-shopping).",
            file=sys.stderr,
        )
        return 2

    file_sha = sha256_file(path)
    df = read_raw_csv(path)
    contract = DatasetContract.build()
    report = validate_dataframe(df, contract)

    print(f"file:       {path.name}")
    print(f"sha256:     {file_sha}")
    print(f"rows x cols: {df.shape[0]} x {df.shape[1]}")
    expected_sha = contract.raw_sha256
    if expected_sha:
        match = "MATCH" if file_sha == expected_sha else "MISMATCH"
        print(f"contract sha256: {expected_sha} [{match}]")
    print()
    print(report.render())
    return 0 if report.ok else 1


def _cmd_build_contract(args: argparse.Namespace) -> int:
    from retail_clickstream_ai.validation.contract import DatasetContract

    contract = DatasetContract.build()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    contract.write(output)
    print(f"wrote deterministic contract -> {output}")
    return 0


def _cmd_build_codebook(args: argparse.Namespace) -> int:
    from retail_clickstream_ai.reference import codebook

    output = codebook.write(args.output)
    print(f"wrote display-only codebook -> {output}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``python -m retail_clickstream_ai.pipeline.data`` parser."""
    parser = argparse.ArgumentParser(
        prog="python -m retail_clickstream_ai.pipeline.data",
        description="Inspect, validate, and contract the raw clickstream dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate-raw", help="Validate a raw CSV against the dataset contract."
    )
    p_validate.add_argument("--input", required=True, help="Path to the raw CSV.")
    p_validate.set_defaults(func=_cmd_validate_raw)

    p_build = sub.add_parser(
        "build-contract", help="Write dataset_contract.json deterministically."
    )
    p_build.add_argument(
        "--output",
        default="artifacts/analyst/dataset_contract.json",
        help="Where to write the contract (default: artifacts/analyst/dataset_contract.json).",
    )
    p_build.set_defaults(func=_cmd_build_contract)

    p_codebook = sub.add_parser(
        "build-codebook", help="Write the display-only value codebook (JSON)."
    )
    p_codebook.add_argument(
        "--output",
        default="artifacts/analyst/codebook.json",
        help="Where to write the codebook (default: artifacts/analyst/codebook.json).",
    )
    p_codebook.set_defaults(func=_cmd_build_codebook)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the module CLI. Returns a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
