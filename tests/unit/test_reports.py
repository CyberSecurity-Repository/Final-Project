"""deterministic reports: required sections and metric-backed numbers.

Proves the anti-hallucination guarantee for prose: every number in ``insights.md``
is emitted through the evidence recorder and traces to a metric, and a fabricated
claim is rejected. Also proves the HTML report is self-contained and complete.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline.cleaning import clean_dataframe
from retail_clickstream_ai.pipeline.eda import compute_metrics, make_figures
from retail_clickstream_ai.reporting.eda_report import render_eda_report
from retail_clickstream_ai.reporting.evidence import (
    Claim,
    UnsupportedClaimError,
    resolve_metric,
    validate_claims,
)
from retail_clickstream_ai.reporting.insights import render_insights
from retail_clickstream_ai.validation.artifacts import REQUIRED_HTML_SECTIONS
from retail_clickstream_ai.validation.contract import DatasetContract

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "clickstream_sample.csv"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


@pytest.fixture
def contract() -> DatasetContract:
    return DatasetContract.build()


@pytest.fixture
def bundle(contract):
    raw = d.read_raw_csv(FIXTURE)
    clean, _ = clean_dataframe(raw, contract)
    metrics = compute_metrics(
        clean, input_sha256="abc", clean_sha256="def", contract_version="1.0.0"
    )
    return clean, metrics


def test_insights_has_required_sections_and_2008_caveat(bundle) -> None:
    _, metrics = bundle
    md, _ = render_insights(metrics)
    for marker in ("## What this is", "## Data quality", "## Key findings", "## Limitations"):
        assert marker in md
    assert "2008" in md


def test_every_insights_claim_traces_to_a_metric(bundle) -> None:
    _, metrics = bundle
    _, ev = render_insights(metrics)
    assert ev.claims  # some numbers were emitted
    for claim in ev.claims:
        assert claim.stated == resolve_metric(metrics, claim.ref)
    # The renderer's own validation must not raise on a faithful render.
    ev.validate()


def test_no_number_in_insights_is_unbacked(bundle) -> None:
    _, metrics = bundle
    md, ev = render_insights(metrics)
    allowed = {c.text for c in ev.claims} | {"2008"}
    for line in md.splitlines():
        if line.lstrip().startswith("#"):
            continue  # headings carry structural numbers like "Finding 1"
        body = re.sub(r"^\s*\d+\.\s", "", line)  # strip markdown list ordinals
        for token in _NUMBER.findall(body):
            assert token in allowed, f"unbacked number {token!r} in: {line}"


def test_fabricated_claim_is_rejected(bundle) -> None:
    _, metrics = bundle
    real = resolve_metric(metrics, "headline.sessions")
    bad = [Claim("artifacts/analyst/eda/eda_metrics.json#headline.sessions", real + 1, "999")]
    with pytest.raises(UnsupportedClaimError):
        validate_claims(bad, metrics)


def test_eda_report_has_required_sections(bundle, contract) -> None:
    clean, metrics = bundle
    figures = make_figures(clean, metrics)
    html, _ = render_eda_report(metrics, figures, contract=contract)
    for section in REQUIRED_HTML_SECTIONS:
        assert section in html


def test_eda_report_is_self_contained(bundle, contract) -> None:
    clean, metrics = bundle
    figures = make_figures(clean, metrics)
    html, _ = render_eda_report(metrics, figures, contract=contract)
    assert "data:image/png;base64," in html  # figures embedded, not linked
    assert "<script" not in html.lower()
    assert 'src="http' not in html  # no external asset fetches
    assert "<link" not in html.lower()
