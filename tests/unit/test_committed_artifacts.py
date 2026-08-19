"""Integrity checks over the *committed* required artifacts and run manifest.

The repository tracks the eight required artifacts plus a successful deterministic run
manifest in the repository, so an evaluator can inspect them and run the app
without regenerating anything. These tests guard that the committed set stays
internally consistent:

* the Analyst set validates against the pinned dataset contract;
* the Scientist set's recorded hashes match its files, its reports match
  ``metrics.json``, and ``model.joblib`` loads and predicts in a trusted round trip;
* the committed ``run_manifest.json`` is a truthful fingerprint — every artifact
  hash it records equals the bytes of the committed file.

If any committed artifact or the manifest drifts, CI fails here. All checks read
the real ``artifacts/`` tree directly (independent of ``ARTIFACT_ROOT``) and are
fully offline — no key, no network, no paid call.
"""

from __future__ import annotations

import json

from retail_clickstream_ai import paths
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.validation import artifacts as A
from retail_clickstream_ai.validation import contract as C

_REPO_ARTIFACTS = paths.PROJECT_ROOT / "artifacts"
_ANALYST = _REPO_ARTIFACTS / "analyst"
_SCIENTIST = _REPO_ARTIFACTS / "scientist"
_MANIFEST = _REPO_ARTIFACTS / "runs" / "flow-final-deterministic" / "run_manifest.json"


def test_committed_analyst_artifacts_validate_against_pinned_contract() -> None:
    report = A.validate_analyst_artifacts(
        clean_path=_ANALYST / "clean_data.csv",
        eda_report_path=_ANALYST / "eda_report.html",
        insights_path=_ANALYST / "insights.md",
        contract_path=_ANALYST / "dataset_contract.json",
        contract=C.DatasetContract.build(),
        pin_hash=True,
        require_all_months=True,
    )
    assert report.ok, [i.to_dict() for i in report.fatal_issues]


def test_committed_scientist_artifacts_are_hash_consistent_and_load() -> None:
    report = A.validate_scientist_artifacts(
        features_path=_SCIENTIST / "features.csv",
        model_path=_SCIENTIST / "model.joblib",
        eval_report_path=_SCIENTIST / "evaluation_report.md",
        model_card_path=_SCIENTIST / "model_card.md",
        metrics_path=_SCIENTIST / "metrics.json",
        metadata_path=_SCIENTIST / "model_metadata.json",
    )
    assert report.ok, [i.to_dict() for i in report.fatal_issues]


def test_committed_run_manifest_fingerprints_committed_artifacts() -> None:
    """Every artifact hash in the committed manifest matches the committed file."""
    m = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert m["status"] == "success"
    assert m["engine"] == "deterministic"
    assert m["gates"] == {"analyst_gate": "passed", "model_gate": "passed"}
    assert m["input"]["sha256"] == C.RAW_SHA256

    for section, base in (("analyst", _ANALYST), ("scientist", _SCIENTIST)):
        artifacts = m[section]["artifacts"]
        assert artifacts, section
        for name, art in artifacts.items():
            f = base / name
            assert f.exists(), f"missing committed artifact: {f}"
            assert d.sha256_file(f) == art["sha256"], f"hash drift for {section}/{name}"
            assert art["size_bytes"] == f.stat().st_size
