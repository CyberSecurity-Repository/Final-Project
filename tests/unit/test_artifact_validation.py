"""Stage 3 — artifact validators: contract↔clean agreement, presence, mismatch.

Runs offline against the committed real artifacts (no raw dataset needed) plus a
tmp fixture run, and proves the validator catches tampering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline.cleaning import clean_dataframe, write_clean_csv
from retail_clickstream_ai.validation.artifacts import (
    validate_analyst_artifacts,
    validate_clean_file_against_contract,
    validate_clean_frame,
)
from retail_clickstream_ai.validation.contract import DatasetContract

REPO = Path(__file__).resolve().parents[2]
ANALYST = REPO / "artifacts" / "analyst"
FIXTURE = REPO / "tests" / "fixtures" / "clickstream_sample.csv"


@pytest.fixture
def contract() -> DatasetContract:
    return DatasetContract.build()


# --- committed real artifacts (offline, no raw data) ----------------------- #
def test_committed_clean_matches_contract(contract) -> None:
    report = validate_clean_file_against_contract(
        ANALYST / "clean_data.csv", contract, pin_hash=True
    )
    assert report.ok, report.render()


def test_all_committed_analyst_artifacts_validate(contract) -> None:
    report = validate_analyst_artifacts(
        clean_path=ANALYST / "clean_data.csv",
        eda_report_path=ANALYST / "eda_report.html",
        insights_path=ANALYST / "insights.md",
        contract_path=ANALYST / "dataset_contract.json",
        contract=contract,
        pin_hash=True,
    )
    assert report.ok, report.render()


# --- fixture (schema-only, hash not pinned) -------------------------------- #
def test_fixture_clean_frame_validates(contract) -> None:
    clean, _ = clean_dataframe(d.read_raw_csv(FIXTURE), contract)
    report = validate_clean_frame(clean, contract)
    assert report.ok, report.render()


def test_fixture_clean_file_validates_without_pin(contract, tmp_path) -> None:
    clean, _ = clean_dataframe(d.read_raw_csv(FIXTURE), contract)
    path = tmp_path / "clean_data.csv"
    write_clean_csv(clean, path)
    ok = validate_clean_file_against_contract(path, contract, pin_hash=False)
    assert ok.ok, ok.render()
    # With the pin on, the fixture's hash cannot match the production constant.
    pinned = validate_clean_file_against_contract(path, contract, pin_hash=True)
    assert not pinned.ok
    assert any(i.rule == "clean_sha256" for i in pinned.issues)


# --- tampering / presence -------------------------------------------------- #
def test_missing_artifact_is_flagged(contract, tmp_path) -> None:
    report = validate_analyst_artifacts(
        clean_path=tmp_path / "nope.csv",
        eda_report_path=tmp_path / "nope.html",
        insights_path=tmp_path / "nope.md",
        contract_path=tmp_path / "nope.json",
        contract=contract,
        pin_hash=False,
    )
    assert not report.ok
    assert any(i.rule == "artifact_missing" for i in report.issues)


def test_empty_artifact_is_flagged(contract, tmp_path) -> None:
    empty = tmp_path / "clean_data.csv"
    empty.write_text("", encoding="utf-8")
    report = validate_clean_file_against_contract(empty, contract, pin_hash=False)
    assert not report.ok
    assert any(i.rule == "artifact_empty" for i in report.issues)


def test_tampered_contract_is_flagged(contract, tmp_path) -> None:
    clean, _ = clean_dataframe(d.read_raw_csv(FIXTURE), contract)
    clean_path = tmp_path / "clean_data.csv"
    write_clean_csv(clean, clean_path)
    # A contract file that is not the built contract must be rejected.
    bad_contract = tmp_path / "dataset_contract.json"
    bad_contract.write_text('{"contract_version": "tampered"}\n', encoding="utf-8")
    (tmp_path / "r.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "i.md").write_text("2008", encoding="utf-8")
    report = validate_analyst_artifacts(
        clean_path=clean_path,
        eda_report_path=tmp_path / "r.html",
        insights_path=tmp_path / "i.md",
        contract_path=bad_contract,
        contract=contract,
        pin_hash=False,
    )
    assert any(i.rule == "contract_mismatch" for i in report.issues)
