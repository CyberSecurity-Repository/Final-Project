"""Leakage-safe feature engineering.

Builds next-main-category targets within verified sessions, drops each session's
final click, restricts predictors to the current and prior clicks, and produces
the chronological (month-based) train/validation/test split. No random row
splits: rows in a session are related and random splitting would leak.

The cardinal rule: a predictor for click ``t`` may read the current click and
prior clicks only — never click ``t+1`` or later. The *target* is the one value
allowed to look ahead (it is the label, `shift(-1)` within the session). Strictly
past aggregates (e.g. the previous click's category) apply ``shift(1)`` before any
running calculation; current-and-prior running counts are computed cumulatively
up to and including ``t`` (explicitly permitted). :func:`run_leakage_audit`
proves these invariants deterministically, and a matching unit-test probe shows
that mutating a future click never changes a current row's predictors.

All numeric work here is deterministic and tested; no LLM is involved. The
Scientist crew's ``run_feature_pipeline`` / ``run_leakage_audit`` tools call these
same functions, so an agent run and the offline run produce identical features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from retail_clickstream_ai.pipeline import cleaning
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.validation.contract import DatasetContract

# --------------------------------------------------------------------------- #
# Feature contract (predeclared).
# --------------------------------------------------------------------------- #
TARGET = "next_main_category"
SPLIT_COLUMN = "split"
CURRENT_CATEGORY = "current_main_category"
PREV_CATEGORY = "prev_main_category"
CLICKS_SO_FAR = "clicks_so_far"
DISTINCT_CATEGORIES_SO_FAR = "distinct_categories_so_far"
COUNT_CURRENT_CATEGORY_SO_FAR = "count_current_category_so_far"

# Sentinel for the first click of a session, which has no prior category.
PREV_CATEGORY_SENTINEL = 0

# Predictors the model may train on, grouped by how the preprocessor treats them.
# Every one is a function of the current click or strictly earlier clicks.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    CURRENT_CATEGORY,  # current click's main category (the baseline signal)
    PREV_CATEGORY,  # prior click's category — shift(1), strictly past
    d.COLOUR,
    d.LOCATION,
    d.MODEL_PHOTOGRAPHY,
    d.PRICE_2,
    d.PAGE,
    d.COUNTRY,
)
NUMERIC_FEATURES: tuple[str, ...] = (
    d.PRICE,
    CLICKS_SO_FAR,  # position within the session (current + prior clicks)
    DISTINCT_CATEGORIES_SO_FAR,  # distinct categories seen in clicks 1..t
    COUNT_CURRENT_CATEGORY_SO_FAR,  # times current category seen in clicks 1..t
)
PREDICTOR_COLUMNS: tuple[str, ...] = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)

# Kept for auditing/grouping/splitting only — NEVER fed to an estimator. In
# particular the raw session id and key are never training inputs.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    d.SESSION_KEY,
    d.SESSION_ID,
    d.YEAR,
    d.MONTH,
    d.DAY,
    d.ORDER,
)

# Final on-disk column order for features.csv.
FEATURE_CSV_COLUMNS: tuple[str, ...] = (
    *IDENTIFIER_COLUMNS,
    *PREDICTOR_COLUMNS,
    SPLIT_COLUMN,
    TARGET,
)

SPLIT_POLICY = "April-June train; July validation; August test"
_SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")

# Serialization mirrors the cleaned CSV so features.csv is byte-reproducible.
_FEATURE_STRING_COLUMNS: tuple[str, ...] = (d.SESSION_KEY,)


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def _split_months(contract: DatasetContract) -> dict[str, set[int]]:
    split = contract.data["split"]
    return {
        "train": set(split["train_months"]),
        "validation": set(split["validation_months"]),
        "test": set(split["test_months"]),
    }


def assign_split(month: pd.Series, contract: DatasetContract) -> pd.Series:
    """Label each row train/validation/test by its month (chronological split).

    A month outside the contract's split maps to ``"unknown"`` so the leakage
    audit can flag it rather than silently mis-bucketing a row.
    """
    months = _split_months(contract)

    def label(m: int) -> str:
        for name in _SPLIT_NAMES:
            if int(m) in months[name]:
                return name
        return "unknown"

    return month.map(label).astype("string")


def build_features(
    clean_df: pd.DataFrame, contract: DatasetContract
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the leakage-safe next-main-category feature table.

    Returns ``(features_df, feature_schema)``. The frame carries identifier
    columns (audit/grouping only), the predictors, a chronological ``split``
    label, and the ``next_main_category`` target. Each session's final click is
    dropped because it has no successor to label.
    """
    df = clean_df if d.SESSION_KEY in clean_df.columns else d.add_session_key(clean_df)
    df = d.canonical_sort(df)  # (session, click order); resets the index

    grouped = df.groupby(d.SESSION_KEY, sort=False)[d.MAIN_CATEGORY]

    # Target: the next click's category within the same session (the only
    # forward-looking value; last click of each session becomes NaN).
    df[TARGET] = grouped.shift(-1)

    # Current-click predictor.
    df[CURRENT_CATEGORY] = df[d.MAIN_CATEGORY].astype("int64")

    # Strictly-past predictor: previous click's category (shift before use).
    df[PREV_CATEGORY] = grouped.shift(1).fillna(PREV_CATEGORY_SENTINEL).astype("int64")

    # Current-and-prior running features (cumulative up to and including t).
    df[CLICKS_SO_FAR] = df.groupby(d.SESSION_KEY, sort=False).cumcount() + 1
    is_new_category = (~df.duplicated([d.SESSION_KEY, d.MAIN_CATEGORY], keep="first")).astype(
        "int64"
    )
    df[DISTINCT_CATEGORIES_SO_FAR] = (
        is_new_category.groupby(df[d.SESSION_KEY], sort=False).cumsum().astype("int64")
    )
    df[COUNT_CURRENT_CATEGORY_SO_FAR] = (
        df.groupby([d.SESSION_KEY, d.MAIN_CATEGORY], sort=False).cumcount() + 1
    )

    # Drop each session's final click (no next-category label).
    labeled = df.dropna(subset=[TARGET]).copy()
    labeled[TARGET] = labeled[TARGET].astype("int64")

    # Chronological split label.
    labeled[SPLIT_COLUMN] = assign_split(labeled[d.MONTH], contract)

    features = labeled.loc[:, list(FEATURE_CSV_COLUMNS)].reset_index(drop=True)
    return features, feature_schema(contract)


def feature_schema(contract: DatasetContract) -> dict[str, Any]:
    """Machine-readable description of the feature table (predictors + target)."""
    return {
        "target": TARGET,
        "target_source_column": d.MAIN_CATEGORY,
        "target_definition": "next click's page_1_main_category within the session (shift(-1))",
        "target_classes": list(contract.data["target"]["classes"]),
        "split_column": SPLIT_COLUMN,
        "split_policy": SPLIT_POLICY,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "predictor_count": len(PREDICTOR_COLUMNS),
        "feature_windows": {
            CURRENT_CATEGORY: "current click (t)",
            PREV_CATEGORY: "prior click (t-1), shift(1)",
            CLICKS_SO_FAR: "clicks 1..t (current + prior)",
            DISTINCT_CATEGORIES_SO_FAR: "clicks 1..t (current + prior)",
            COUNT_CURRENT_CATEGORY_SO_FAR: "clicks 1..t (current + prior)",
        },
        "leakage_rules": [
            "predictors use the current click and prior clicks only",
            "no future row, future session length, future/final category, or t+1 proxy",
            "the raw session id/key is an identifier, never a training input",
        ],
    }


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def write_features_csv(features_df: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Serialize the feature table deterministically and return its file stats."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(
        out,
        index=False,
        sep=cleaning.CLEAN_DELIMITER,
        lineterminator=cleaning.CLEAN_LINE_ENDING,
        encoding=cleaning.CLEAN_ENCODING,
    )
    return {
        "path": str(out),
        "sha256": d.sha256_file(out),
        "size_bytes": int(os.path.getsize(out)),
        "row_count": int(features_df.shape[0]),
        "column_count": int(features_df.shape[1]),
    }


def read_features_csv(path: str | Path) -> pd.DataFrame:
    """Read features.csv back with the correct dtypes (session_key stays text)."""
    string_cols = dict.fromkeys(_FEATURE_STRING_COLUMNS, "string")
    return pd.read_csv(path, sep=cleaning.CLEAN_DELIMITER, dtype=string_cols)


# --------------------------------------------------------------------------- #
# Split manifest
# --------------------------------------------------------------------------- #
def build_split_manifest(features_df: pd.DataFrame, contract: DatasetContract) -> dict[str, Any]:
    """Row/session/class counts per split — the immutable split evidence."""
    classes = [int(c) for c in contract.data["target"]["classes"]]
    split = contract.data["split"]
    per_split: dict[str, Any] = {}
    all_present = True
    for name in _SPLIT_NAMES:
        part = features_df[features_df[SPLIT_COLUMN] == name]
        counts = part[TARGET].value_counts()
        class_counts = {str(c): int(counts.get(c, 0)) for c in classes}
        present = all(class_counts[str(c)] > 0 for c in classes)
        all_present = all_present and present
        per_split[name] = {
            "rows": int(len(part)),
            "sessions": int(part[d.SESSION_KEY].nunique()),
            "class_counts": class_counts,
            "all_classes_present": present,
        }
    return {
        "policy": SPLIT_POLICY,
        "strategy": split["strategy"],
        "train_months": list(split["train_months"]),
        "validation_months": list(split["validation_months"]),
        "test_months": list(split["test_months"]),
        "target_classes": classes,
        "splits": per_split,
        "all_classes_present_each_split": all_present,
        "train_rows": per_split["train"]["rows"],
        "validation_rows": per_split["validation"]["rows"],
        "test_rows": per_split["test"]["rows"],
    }


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
# The predictor set is a fixed allowlist; any column that could encode the
# future is forbidden by name as an extra guard.
_FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = ("future", "final", "next_", "last_", "session_length")

# Minimum rows a split must contain to be usable (defensive; the real data has
# thousands per split). Below this the audit fails closed rather than training.
MIN_ROWS_PER_SPLIT = 4


def _check(name: str, passed: bool, detail: str, *, fatal: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": "fatal" if fatal else "warning",
        "detail": detail,
    }


def _probe_no_future_leak() -> tuple[bool, str]:
    """Prove predictors are past-only by mutating a future click.

    On a tiny deterministic session, flip the *last* click's category and rebuild
    the features. Every earlier row's predictors must be byte-identical; only the
    target of the second-to-last row (whose label is that last click) may change.
    """
    contract = DatasetContract.build()
    base = pd.DataFrame(
        {
            d.YEAR: [2008] * 4,
            d.MONTH: [4, 4, 7, 7],
            d.DAY: [1, 1, 2, 2],
            d.ORDER: [1, 2, 1, 2],
            d.COUNTRY: [29, 29, 29, 29],
            d.SESSION_ID: [1, 1, 2, 2],
            d.MAIN_CATEGORY: [1, 2, 3, 4],
            d.CLOTHING_MODEL: ["A1", "B1", "C1", "P1"],
            d.COLOUR: [1, 2, 3, 4],
            d.LOCATION: [1, 2, 3, 4],
            d.MODEL_PHOTOGRAPHY: [1, 2, 1, 2],
            d.PRICE: [20, 30, 40, 50],
            d.PRICE_2: [1, 2, 1, 2],
            d.PAGE: [1, 2, 3, 4],
        }
    )
    original, _ = build_features(base, contract)
    mutated_frame = base.copy()
    # Flip the final click of session 1 (order 2, index 1).
    mutated_frame.loc[1, d.MAIN_CATEGORY] = 3
    mutated, _ = build_features(mutated_frame, contract)

    predictors = list(PREDICTOR_COLUMNS)
    # Align on the surviving (non-last) row of session 1: session 1 keeps only
    # its first click after dropping the last; predictors must be unchanged.
    key = original[d.SESSION_KEY] == "2008-04-01-s1"
    orig_row = original.loc[key, predictors].reset_index(drop=True)
    mut_row = mutated.loc[mutated[d.SESSION_KEY] == "2008-04-01-s1", predictors].reset_index(
        drop=True
    )
    predictors_stable = orig_row.equals(mut_row)
    # The target of that row DID move (its next click was flipped) — confirming
    # the label looks ahead while predictors do not.
    orig_target = int(original.loc[key, TARGET].iloc[0])
    mut_target = int(mutated.loc[mutated[d.SESSION_KEY] == "2008-04-01-s1", TARGET].iloc[0])
    target_moved = orig_target != mut_target
    ok = predictors_stable and target_moved
    return ok, (
        f"predictors_stable={predictors_stable}, target_moved={target_moved} "
        f"({orig_target}->{mut_target})"
    )


def run_leakage_audit(
    clean_df: pd.DataFrame, features_df: pd.DataFrame, contract: DatasetContract
) -> dict[str, Any]:
    """Deterministic leakage checks. Returns a report with ``passed`` = no fatal fail."""
    checks: list[dict[str, Any]] = []

    keyed = clean_df if d.SESSION_KEY in clean_df.columns else d.add_session_key(clean_df)
    keyed = d.canonical_sort(keyed)
    n_sessions = int(keyed[d.SESSION_KEY].nunique())
    n_clean = int(len(keyed))

    # 1. Target reconstructs exactly and never crosses a session boundary.
    expected = keyed[[d.SESSION_KEY, d.ORDER, d.MAIN_CATEGORY]].copy()
    expected["expected_next"] = expected.groupby(d.SESSION_KEY, sort=False)[d.MAIN_CATEGORY].shift(
        -1
    )
    merged = features_df.merge(
        expected[[d.SESSION_KEY, d.ORDER, "expected_next"]],
        on=[d.SESSION_KEY, d.ORDER],
        how="left",
    )
    target_ok = bool((merged["expected_next"].astype("Int64") == merged[TARGET]).all()) and bool(
        merged["expected_next"].notna().all()
    )
    checks.append(
        _check(
            "target_shift_within_session",
            target_ok,
            "every features target equals the same session's next-click category",
        )
    )

    # 2. Final click of every session removed (labeled == clean - sessions).
    expected_labeled = n_clean - n_sessions
    max_order = keyed.groupby(d.SESSION_KEY, sort=False)[d.ORDER].transform("max")
    last_clicks = keyed.loc[keyed[d.ORDER] == max_order, [d.SESSION_KEY, d.ORDER]]
    last_in_features = features_df.merge(last_clicks, on=[d.SESSION_KEY, d.ORDER], how="inner")
    final_removed = int(len(features_df)) == expected_labeled and len(last_in_features) == 0
    checks.append(
        _check(
            "final_clicks_removed",
            final_removed,
            f"rows={len(features_df)} expected={expected_labeled}; "
            f"last-clicks present={len(last_in_features)}",
        )
    )

    # 3. No forbidden future-derived columns; target not among predictors.
    predictor_set = set(PREDICTOR_COLUMNS)
    bad_names = [
        c
        for c in features_df.columns
        if c not in FEATURE_CSV_COLUMNS
        or (c in predictor_set and any(tok in c for tok in _FORBIDDEN_FEATURE_TOKENS))
    ]
    forbidden_ok = not bad_names and TARGET not in predictor_set
    checks.append(
        _check(
            "no_forbidden_future_columns",
            forbidden_ok,
            f"unexpected/forbidden columns={bad_names}; "
            f"target_in_predictors={TARGET in predictor_set}",
        )
    )

    # 4. Past aggregates cannot see the current/future target (mutation probe).
    probe_ok, probe_detail = _probe_no_future_leak()
    checks.append(_check("past_aggregates_no_future_leak", probe_ok, probe_detail))

    # 5. Month split boundaries exact and disjoint; no unknown bucket.
    months = _split_months(contract)
    disjoint = not (
        months["train"] & months["validation"]
        or months["train"] & months["test"]
        or months["validation"] & months["test"]
    )
    labels = set(features_df[SPLIT_COLUMN].dropna().unique())
    no_unknown = "unknown" not in labels
    month_map_ok = True
    for name in _SPLIT_NAMES:
        part_months = {
            int(m) for m in features_df.loc[features_df[SPLIT_COLUMN] == name, d.MONTH].unique()
        }
        if not part_months <= months[name]:
            month_map_ok = False
    split_ok = disjoint and no_unknown and month_map_ok
    checks.append(
        _check(
            "month_split_boundaries_exact",
            split_ok,
            f"disjoint={disjoint} no_unknown={no_unknown} month_map_ok={month_map_ok}",
        )
    )

    # 6. Every split has all classes and enough rows (fail closed, never resplit).
    manifest = build_split_manifest(features_df, contract)
    sufficiency_ok = manifest["all_classes_present_each_split"] and all(
        manifest["splits"][name]["rows"] >= MIN_ROWS_PER_SPLIT for name in _SPLIT_NAMES
    )
    checks.append(
        _check(
            "splits_have_all_classes_and_rows",
            sufficiency_ok,
            f"each split contains all target classes and at least {MIN_ROWS_PER_SPLIT} rows",
        )
    )

    passed = all(c["passed"] for c in checks if c["severity"] == "fatal")
    return {
        "passed": passed,
        "n_clean_rows": n_clean,
        "n_sessions": n_sessions,
        "n_feature_rows": int(len(features_df)),
        "checks": checks,
        "split_manifest": manifest,
    }
