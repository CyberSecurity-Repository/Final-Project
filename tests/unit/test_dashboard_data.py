"""Unit tests for the Streamlit-free dashboard data services."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from retail_clickstream_ai.dashboard import data as svc
from retail_clickstream_ai.validation.errors import ContractValidationError


# --------------------------------------------------------------------------- #
# Artifact status / latest run
# --------------------------------------------------------------------------- #
def test_artifact_status_missing_file(tmp_path: Path) -> None:
    status = svc.artifact_status(tmp_path / "does_not_exist.json")
    assert status.exists is False
    assert status.sha256 is None
    assert status.modified_at is None


def test_artifact_status_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "thing.txt"
    path.write_text("hello", encoding="utf-8")
    status = svc.artifact_status(path)
    assert status.exists is True
    assert status.size_bytes == 5
    assert status.sha256 is not None
    assert status.modified_at is not None


def test_find_latest_run_none_when_runs_dir_absent(tmp_path: Path) -> None:
    latest = svc.find_latest_run(tmp_path / "nonexistent")
    assert latest.kind == "none"


def test_find_latest_run_picks_newest_manifest(tmp_path: Path) -> None:
    older = tmp_path / "run-a"
    newer = tmp_path / "run-b"
    older.mkdir()
    newer.mkdir()
    (older / "run_manifest.json").write_text('{"run_id": "run-a"}', encoding="utf-8")
    time.sleep(0.01)
    (newer / "run_manifest.json").write_text('{"run_id": "run-b"}', encoding="utf-8")

    latest = svc.find_latest_run(tmp_path)
    assert latest.kind == "success"
    assert latest.run_id == "run-b"


def test_find_latest_run_reports_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-fail"
    run_dir.mkdir()
    (run_dir / "failure_report.json").write_text(
        '{"run_id": "run-fail", "failed_step": "analyst_gate"}', encoding="utf-8"
    )
    latest = svc.find_latest_run(tmp_path)
    assert latest.kind == "failed"
    assert latest.payload is not None
    assert latest.payload["failed_step"] == "analyst_gate"


def test_find_latest_run_handles_corrupt_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-bad"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{not json", encoding="utf-8")
    latest = svc.find_latest_run(tmp_path)
    assert latest.kind == "corrupt"
    assert latest.payload is None


# --------------------------------------------------------------------------- #
# Missing-artifact behavior
# --------------------------------------------------------------------------- #
def test_read_insights_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    with pytest.raises(svc.ArtifactMissingError):
        svc.read_insights()


def test_read_model_metadata_raises_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    scientist_dir = tmp_path / "scientist"
    scientist_dir.mkdir(parents=True)
    (scientist_dir / "model_metadata.json").write_text("", encoding="utf-8")
    with pytest.raises(svc.ArtifactMissingError):
        svc.read_model_metadata()


# --------------------------------------------------------------------------- #
# Trusted model loading (hash pre-check, using the real committed artifacts)
# --------------------------------------------------------------------------- #
def test_verify_model_artifact_passes_for_committed_model() -> None:
    ok, observed, expected = svc.verify_model_artifact()
    assert ok is True
    assert observed == expected


def test_load_verified_model_refuses_tampered_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    real_root = Path("artifacts")
    fake_root = tmp_path / "artifacts"
    shutil.copytree(real_root / "scientist", fake_root / "scientist")
    with (fake_root / "scientist" / "model.joblib").open("ab") as fh:
        fh.write(b"tampered-bytes")

    monkeypatch.setenv("ARTIFACT_ROOT", str(fake_root))
    ok, observed, expected = svc.verify_model_artifact()
    assert ok is False
    assert observed != expected
    with pytest.raises(svc.ModelIntegrityError):
        svc.load_verified_model()


# --------------------------------------------------------------------------- #
# Prediction contract, validation, and inference (real committed model)
# --------------------------------------------------------------------------- #
def test_prediction_field_specs_cover_every_predictor() -> None:
    from retail_clickstream_ai.pipeline import features as feat

    specs = svc.prediction_field_specs()
    assert {s.name for s in specs} == set(feat.PREDICTOR_COLUMNS)


def test_country_field_uses_range_domain_not_allowed_values() -> None:
    specs = {s.name: s for s in svc.prediction_field_specs()}
    country = specs["country"]
    assert country.options is not None
    assert len(country.options) == 47


def test_valid_prediction_returns_category_and_probabilities() -> None:
    specs = svc.prediction_field_specs()
    values = {s.name: s.default for s in specs}
    result = svc.predict_next_category(values)
    assert result.predicted_class in (1, 2, 3, 4)
    assert len(result.probabilities) == 4
    total = sum(p for _, _, p in result.probabilities)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_invalid_category_code_is_rejected_before_inference() -> None:
    specs = svc.prediction_field_specs()
    values: dict[str, Any] = {s.name: s.default for s in specs}
    values["current_main_category"] = 99  # not in {1,2,3,4}
    with pytest.raises(ContractValidationError) as excinfo:
        svc.predict_next_category(values)
    assert any(i.rule == "invalid_category" for i in excinfo.value.report.issues)


def test_out_of_range_numeric_is_rejected() -> None:
    specs = svc.prediction_field_specs()
    values: dict[str, Any] = {s.name: s.default for s in specs}
    values["price"] = 0  # below the contract's minimum of 1
    report = svc.validate_prediction_inputs(values)
    assert not report.ok
    assert any(i.rule == "out_of_range" for i in report.issues)


def test_first_click_cannot_have_a_previous_category() -> None:
    specs = svc.prediction_field_specs()
    values: dict[str, Any] = {s.name: s.default for s in specs}
    values["clicks_so_far"] = 1
    values["prev_main_category"] = 2  # inconsistent with "first click"
    report = svc.validate_prediction_inputs(values)
    assert not report.ok
    assert any(i.rule == "history_inconsistent" for i in report.issues)


def test_distinct_categories_cannot_exceed_clicks_so_far() -> None:
    specs = svc.prediction_field_specs()
    values: dict[str, Any] = {s.name: s.default for s in specs}
    values["clicks_so_far"] = 2
    values["prev_main_category"] = 1
    values["distinct_categories_so_far"] = 3
    report = svc.validate_prediction_inputs(values)
    assert not report.ok
    assert any(i.rule == "history_inconsistent" for i in report.issues)


# --------------------------------------------------------------------------- #
# Markdown section extraction
# --------------------------------------------------------------------------- #
def test_extract_markdown_section_returns_body_between_headings() -> None:
    text = "# Title\n\n## First\n\nbody one\n\n## Second\n\nbody two\n"
    assert svc.extract_markdown_section(text, "First") == "body one"
    assert svc.extract_markdown_section(text, "Second") == "body two"


def test_extract_markdown_section_returns_empty_when_absent() -> None:
    assert svc.extract_markdown_section("# Title\n", "Missing") == ""


# --------------------------------------------------------------------------- #
# Against fixture artifacts (offline, no key, small synthetic dataset)
# --------------------------------------------------------------------------- #
def test_full_pipeline_fixture_renders_every_reader(full_artifact_env: Path) -> None:
    assert svc.read_dataset_contract()["contract_version"]
    assert svc.read_insights()
    assert svc.read_eda_metrics()
    assert svc.read_eda_figure_manifest()["figure_count"] > 0
    assert not svc.read_eda_table("main_category_frequency").empty
    assert svc.read_scientist_metrics()["selected_candidate_id"]
    assert svc.read_model_metadata()["feature_list"]
    assert svc.read_model_card()
    assert svc.read_evaluation_report()
    assert svc.read_feature_schema()["target"]

    ok, _, _ = svc.verify_model_artifact()
    assert ok is True

    specs = svc.prediction_field_specs()
    values = {s.name: s.default for s in specs}
    result = svc.predict_next_category(values)
    assert result.predicted_class in (1, 2, 3, 4)
