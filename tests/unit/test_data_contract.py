"""Column normalization, the dataset contract, and validators.

These run fully offline against a tiny synthetic fixture (no network, no key,
never the full raw dataset). They prove the anti-hallucination guarantees:
valid data passes, deliberate violations fail with clear issues, the composite
session key resolves id collisions, the documented sort policy handles
out-of-order rows, and the contract serializes reproducibly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline.data import (
    add_session_key,
    canonical_sort,
    normalize_column_name,
    normalize_columns,
    read_raw_csv,
    sha256_bytes,
)
from retail_clickstream_ai.validation.contract import DatasetContract
from retail_clickstream_ai.validation.raw import validate_dataframe

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "clickstream_sample.csv"
CONTRACT_ARTIFACT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "analyst" / "dataset_contract.json"
)


@pytest.fixture
def contract() -> DatasetContract:
    return DatasetContract.build()


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return read_raw_csv(FIXTURE)


def _row(month: int, day: int, order: int, sid: int, cat: int, model: str) -> dict[str, Any]:
    """Build one fully-populated, in-domain synthetic click row (normalized names)."""
    return {
        d.YEAR: 2008,
        d.MONTH: month,
        d.DAY: day,
        d.ORDER: order,
        d.COUNTRY: 10,
        d.SESSION_ID: sid,
        d.MAIN_CATEGORY: cat,
        d.CLOTHING_MODEL: model,
        d.COLOUR: 1,
        d.LOCATION: 1,
        d.MODEL_PHOTOGRAPHY: 1,
        d.PRICE: 20,
        d.PRICE_2: 2,
        d.PAGE: 1,
    }


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[d.CLOTHING_MODEL] = df[d.CLOTHING_MODEL].astype("string")
    return df


# --------------------------------------------------------------------------- #
# Column normalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("year", "year"),
        ("session ID", "session_id"),
        ("page 1 (main category)", "page_1_main_category"),
        ("page 2 (clothing model)", "page_2_clothing_model"),
        ("price 2", "price_2"),
        ("model photography", "model_photography"),
        ("  Weird  Name!! ", "weird_name"),
    ],
)
def test_normalize_column_name(raw: str, expected: str) -> None:
    assert normalize_column_name(raw) == expected


def test_read_normalizes_all_source_columns(valid_df: pd.DataFrame) -> None:
    assert tuple(valid_df.columns) == d.NORMALIZED_COLUMNS


def test_normalize_columns_rejects_collision() -> None:
    # Both headers normalize to "price_2".
    df = pd.DataFrame(columns=["price 2", "price  2"])
    with pytest.raises(ValueError, match="duplicate names"):
        normalize_columns(df)


# --------------------------------------------------------------------------- #
# Valid input passes
# --------------------------------------------------------------------------- #
def test_valid_fixture_passes(valid_df: pd.DataFrame, contract: DatasetContract) -> None:
    report = validate_dataframe(valid_df, contract)
    assert report.ok, report.render()
    assert report.issues == []


def test_session_key_format(valid_df: pd.DataFrame) -> None:
    keys = set(add_session_key(valid_df)[d.SESSION_KEY])
    # Same bare session_id=1 on two different days -> two distinct composite keys.
    assert "2008-04-01-s1" in keys
    assert "2008-05-02-s1" in keys


# --------------------------------------------------------------------------- #
# Deliberate violations fail
# --------------------------------------------------------------------------- #
def test_missing_required_column_fails(valid_df: pd.DataFrame, contract: DatasetContract) -> None:
    broken = valid_df.drop(columns=[d.PAGE])
    report = validate_dataframe(broken, contract, require_all_months=False)
    assert not report.ok
    assert ("missing_column", d.PAGE) in {(i.rule, i.column) for i in report.issues}


def test_wrong_dtype_fails(valid_df: pd.DataFrame, contract: DatasetContract) -> None:
    broken = valid_df.copy()
    broken[d.COLOUR] = broken[d.COLOUR].astype("object")
    broken.loc[0, d.COLOUR] = "purple"  # forces a non-integer column
    report = validate_dataframe(broken, contract, require_all_months=False)
    assert not report.ok
    assert any(i.rule == "dtype" and i.column == d.COLOUR for i in report.issues)


def test_invalid_category_fails(valid_df: pd.DataFrame, contract: DatasetContract) -> None:
    broken = valid_df.copy()
    broken.loc[0, d.MAIN_CATEGORY] = 9  # outside {1,2,3,4}
    report = validate_dataframe(broken, contract, require_all_months=False)
    assert not report.ok
    assert any(i.rule == "allowed_values" and i.column == d.MAIN_CATEGORY for i in report.issues)


def test_out_of_range_value_fails(valid_df: pd.DataFrame, contract: DatasetContract) -> None:
    broken = valid_df.copy()
    broken.loc[0, d.COUNTRY] = 99  # country range is [1, 47]; not a session/order column
    report = validate_dataframe(broken, contract, require_all_months=False)
    assert not report.ok
    assert any(i.rule == "range" and i.column == d.COUNTRY for i in report.issues)
    # Changing country must not disturb session/click-order integrity.
    assert all(i.rule != "click_order_sequence" for i in report.issues)


def test_invalid_page2_pattern_fails(valid_df: pd.DataFrame, contract: DatasetContract) -> None:
    broken = valid_df.copy()
    broken.loc[0, d.CLOTHING_MODEL] = "13A"  # violates ^[A-P][0-9]+$
    report = validate_dataframe(broken, contract, require_all_months=False)
    assert any(i.rule == "pattern" and i.column == d.CLOTHING_MODEL for i in report.issues)


# --------------------------------------------------------------------------- #
# Click order: documented sort policy vs a genuinely broken sequence
# --------------------------------------------------------------------------- #
def test_stored_out_of_order_passes_via_sort_policy(
    valid_df: pd.DataFrame, contract: DatasetContract
) -> None:
    shuffled = valid_df.iloc[::-1].reset_index(drop=True)  # reverse on-disk order
    report = validate_dataframe(shuffled, contract)
    assert report.ok, report.render()
    # The canonical sort restores a 1..n run inside every session.
    keyed = add_session_key(canonical_sort(shuffled))
    for _, group in keyed.groupby(d.SESSION_KEY):
        assert group[d.ORDER].tolist() == list(range(1, len(group) + 1))


def test_broken_order_sequence_fails(valid_df: pd.DataFrame, contract: DatasetContract) -> None:
    broken = valid_df.copy()
    # Session 2008-04-01-s1 has orders {1,2,3}; duplicate the 3 to create a gap+dup.
    broken.loc[1, d.ORDER] = 3
    report = validate_dataframe(broken, contract, require_all_months=False)
    assert not report.ok
    assert any(i.rule in {"click_order_sequence", "duplicate_click_order"} for i in report.issues)


# --------------------------------------------------------------------------- #
# Session-ID collision handled by the verified composite session key
# --------------------------------------------------------------------------- #
def test_session_id_collision_resolved_by_composite_key(
    contract: DatasetContract,
) -> None:
    # session_id=7 reused across two different days, each a clean 1..n sequence.
    df = _frame(
        [
            _row(4, 1, 1, 7, 1, "A1"),
            _row(4, 1, 2, 7, 2, "B1"),
            _row(4, 2, 1, 7, 3, "C1"),
            _row(4, 2, 2, 7, 4, "P1"),
        ]
    )
    report = validate_dataframe(df, contract, require_all_months=False)
    assert report.ok, report.render()  # composite key => two valid sessions
    assert add_session_key(df)[d.SESSION_KEY].nunique() == 2
    # Grouping by the bare session_id would look broken (orders 1,1,2,2).
    assert sorted(df[d.ORDER].tolist()) == [1, 1, 2, 2]


# --------------------------------------------------------------------------- #
# Contract is deterministic, reproducible, and records verified facts
# --------------------------------------------------------------------------- #
def test_contract_serialization_is_stable() -> None:
    first = DatasetContract.build().to_json()
    second = DatasetContract.build().to_json()
    assert first == second
    assert json.loads(first) == DatasetContract.build().data  # round-trips


def test_contract_records_verified_facts() -> None:
    c = DatasetContract.build()
    assert c.raw_sha256 == ("fcc167bbd0badd4c9685bd8543097e318f8228e48075335db7cd781cee88115d")
    assert c.data["dataset"]["raw_file"]["row_count"] == 165474
    assert c.data["dataset"]["raw_file"]["committed"] is False
    assert c.session_key_columns == ["year", "month", "day", "session_id"]
    assert c.sort_columns == ["year", "month", "day", "session_id", "order"]
    assert c.required_months == [4, 5, 6, 7, 8]
    assert c.data["target"]["source_column"] == d.MAIN_CATEGORY
    assert c.data["split"]["train_months"] == [4, 5, 6]


def test_committed_contract_matches_build() -> None:
    assert CONTRACT_ARTIFACT.exists(), (
        "Missing artifact — regenerate with: "
        "python -m retail_clickstream_ai.pipeline.data build-contract"
    )
    on_disk = CONTRACT_ARTIFACT.read_text(encoding="utf-8")
    assert on_disk == DatasetContract.build().to_json()


def test_sha256_bytes_is_correct() -> None:
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
