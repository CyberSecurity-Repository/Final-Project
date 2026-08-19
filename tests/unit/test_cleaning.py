"""deterministic cleaning: idempotence, order, audit, fail-closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline.cleaning import (
    clean_dataframe,
    read_clean_csv,
    write_clean_csv,
)
from retail_clickstream_ai.validation.contract import DatasetContract
from retail_clickstream_ai.validation.errors import ContractValidationError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "clickstream_sample.csv"


@pytest.fixture
def contract() -> DatasetContract:
    return DatasetContract.build()


@pytest.fixture
def raw_df():
    return d.read_raw_csv(FIXTURE)


def test_cleaning_preserves_rows_and_column_order(raw_df, contract) -> None:
    clean, audit = clean_dataframe(raw_df, contract)
    assert list(clean.columns) == list(d.REQUIRED_OUTPUT_COLUMNS)
    assert audit.rows_preserved
    assert audit.input_rows == audit.output_rows == raw_df.shape[0]
    assert clean.shape[1] == 15  # 14 source columns + derived session_key


def test_cleaning_is_idempotent(raw_df, contract) -> None:
    clean, _ = clean_dataframe(raw_df, contract)
    # Feeding the cleaned frame back through cleaning yields an identical frame.
    again, _ = clean_dataframe(clean, contract)
    assert again.reset_index(drop=True).equals(clean.reset_index(drop=True))


def test_cleaning_output_is_stably_ordered(raw_df, contract) -> None:
    clean1, _ = clean_dataframe(raw_df, contract)
    shuffled = raw_df.iloc[::-1].reset_index(drop=True)
    clean2, _ = clean_dataframe(shuffled, contract)
    # Canonical sort makes on-disk order irrelevant.
    assert clean1.equals(clean2)


def test_transformation_audit_records_every_rule(raw_df, contract) -> None:
    _, audit = clean_dataframe(raw_df, contract)
    names = {r.name for r in audit.rules}
    assert {
        "parse_and_validate",
        "normalize_column_names",
        "reject_full_row_duplicates",
        "derive_session_key",
        "canonical_sort",
        "order_output_columns",
    } <= names
    by_name = {r.name: r for r in audit.rules}
    # Session key touches every row; duplicates policy removed none.
    assert by_name["derive_session_key"].affected_rows == raw_df.shape[0]
    assert by_name["reject_full_row_duplicates"].affected_rows == 0
    assert audit.fatal_issue_count == 0
    assert audit.validation_passed


def test_write_clean_csv_is_byte_deterministic(raw_df, contract, tmp_path) -> None:
    clean, _ = clean_dataframe(raw_df, contract)
    a = write_clean_csv(clean, tmp_path / "a.csv")
    b = write_clean_csv(clean, tmp_path / "b.csv")
    assert a["sha256"] == b["sha256"]
    assert a["row_count"] == clean.shape[0]
    # Round-trips with the correct string dtypes.
    reloaded = read_clean_csv(tmp_path / "a.csv")
    assert list(reloaded.columns) == list(d.REQUIRED_OUTPUT_COLUMNS)
    assert reloaded[d.SESSION_KEY].dtype == "string"


def test_cleaning_fails_closed_on_invalid_data(raw_df, contract) -> None:
    broken = raw_df.copy()
    broken.loc[0, d.MAIN_CATEGORY] = 9  # outside {1,2,3,4}
    with pytest.raises(ContractValidationError):
        clean_dataframe(broken, contract, require_all_months=False)


def test_cleaning_never_drops_rows_on_out_of_order_input(raw_df, contract) -> None:
    shuffled = raw_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    clean, audit = clean_dataframe(shuffled, contract)
    assert clean.shape[0] == raw_df.shape[0]
    assert audit.rows_preserved
