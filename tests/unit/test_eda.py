"""deterministic EDA: metric correctness, determinism, figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline.cleaning import clean_dataframe
from retail_clickstream_ai.pipeline.eda import compute_metrics, figure_manifest, make_figures
from retail_clickstream_ai.validation.contract import DatasetContract

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "clickstream_sample.csv"


@pytest.fixture
def metrics():
    contract = DatasetContract.build()
    raw = d.read_raw_csv(FIXTURE)
    clean, _ = clean_dataframe(raw, contract)
    return clean, compute_metrics(
        clean, input_sha256="abc", clean_sha256="def", contract_version="1.0.0"
    )


def test_metrics_are_deterministic(metrics) -> None:
    clean, m1 = metrics
    m2 = compute_metrics(clean, input_sha256="abc", clean_sha256="def", contract_version="1.0.0")
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)


def test_target_rows_equal_clicks_minus_sessions(metrics) -> None:
    _, m = metrics
    assert m["counts"]["target_rows"] == m["counts"]["rows"] - m["counts"]["sessions"]
    assert m["transitions"]["target_rows"] == m["counts"]["target_rows"]


def test_transition_probabilities_bounded(metrics) -> None:
    _, m = metrics
    assert 0.0 <= m["transitions"]["next_same_category_pct"] <= 100.0
    for value in m["transitions"]["row_normalized_pct"].values():
        assert 0.0 <= value <= 100.0


def test_category_shares_sum_to_100(metrics) -> None:
    _, m = metrics
    shares = [v["share_pct"] for v in m["main_category"]["by_category"].values()]
    assert abs(sum(shares) - 100.0) < 0.5  # rounding tolerance


def test_headline_and_quality_present(metrics) -> None:
    _, m = metrics
    for key in ("rows", "sessions", "next_same_category_pct", "top_country_label"):
        assert key in m["headline"]
    q = m["data_quality"]
    assert q["null_total"] == 0
    assert q["full_row_duplicates"] == 0
    assert q["rows_dropped_in_cleaning"] == 0


def test_at_least_five_figures_with_content(metrics) -> None:
    clean, m = metrics
    figs = make_figures(clean, m)
    assert len(figs) >= 5
    for fig in figs:
        assert fig.png_base64  # non-empty base64
        assert fig.evidence_ref.startswith("artifacts/analyst/eda/eda_metrics.json#")
    manifest = figure_manifest(figs)
    assert manifest["figure_count"] == len(figs)
