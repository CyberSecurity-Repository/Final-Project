"""CrewAI Flow integration tests (mocked crew/LLM boundary, fully offline).

Every test records the crew call order **explicitly** so the automated Analyst -> Scientist
handoff is proven rather than inferred, and asserts that a failing deterministic gate blocks
all downstream work. No OpenAI key and no network are used: the crew entry points
(``run_analyst_crew`` / ``run_scientist_crew``) are patched at the flow module. The happy
fakes lay down *real* artifacts via the tested deterministic pipelines; the failure fakes
break exactly one thing so a single deterministic gate is exercised.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from retail_clickstream_ai import flow as F
from retail_clickstream_ai import paths
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline import features as feat
from retail_clickstream_ai.pipeline.analyst_pipeline import run_analyst_pipeline
from retail_clickstream_ai.pipeline.scientist_pipeline import run_scientist_pipeline

pytestmark = pytest.mark.integration

EXPECTED_ORDER = [
    "prepare",
    "analyst_crew",
    "analyst_gate",
    "scientist_crew",
    "model_gate",
    "manifest",
]


# --------------------------------------------------------------------------- #
# Mock crew boundaries: they record their call and (for the happy path) produce
# the real artifacts the deterministic gates then validate.
# --------------------------------------------------------------------------- #
def _analyst_fake(
    calls: list[str],
    raw_path: Path,
    *,
    mutate: Callable[[Path], None] | None = None,
    exc: Exception | None = None,
) -> Callable[..., Any]:
    def fake(
        input_path: str,
        *,
        run_id: str | None = None,
        pin_hash: bool = True,
        require_all_months: bool = True,
        verbose: bool = False,
    ) -> Any:
        calls.append("analyst_crew")
        if exc is not None:
            raise exc
        res = run_analyst_pipeline(
            raw_path, run_id="analyst-synth", pin_contract_hash=False, require_all_months=True
        )
        if mutate is not None:
            mutate(paths.analyst_artifacts())
        return ("analyst-crew-output", res)

    return fake


def _scientist_fake(
    calls: list[str],
    *,
    mutate: Callable[[Path], None] | None = None,
    exc: Exception | None = None,
) -> Callable[..., Any]:
    def fake(*, run_id: str | None = None, verbose: bool = False) -> Any:
        calls.append("scientist_crew")
        if exc is not None:
            raise exc
        res = run_scientist_pipeline(run_id="scientist-synth", pin_analyst_hash=False)
        if mutate is not None:
            mutate(paths.scientist_artifacts())
        return ("scientist-crew-output", res)

    return fake


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "runs" / run_id


# --------------------------------------------------------------------------- #
# 1. Happy path reaches the success manifest in the correct order.
# --------------------------------------------------------------------------- #
def test_happy_path_reaches_manifest_in_order(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(F, "run_analyst_crew", _analyst_fake(calls, synthetic_raw_path))
    monkeypatch.setattr(F, "run_scientist_crew", _scientist_fake(calls))

    state = F.run_flow(
        synthetic_raw_path,
        run_id="flow-happy",
        engine="crew",
        pin_hash=False,
        artifact_root=tmp_path,
    )

    assert state.status is F.RunStatus.SUCCESS
    assert [e.stage for e in state.stage_events] == EXPECTED_ORDER
    assert calls == ["analyst_crew", "scientist_crew"]  # automated handoff, in order

    manifest = _run_dir(tmp_path, "flow-happy") / "run_manifest.json"
    assert manifest.exists()
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["status"] == "success"
    assert m["app_name"] == "retail-clickstream-ai"
    assert m["package_version"]
    assert m["gates"] == {"analyst_gate": "passed", "model_gate": "passed"}
    assert set(m["analyst"]["artifacts"]) == {
        "clean_data.csv",
        "eda_report.html",
        "insights.md",
        "dataset_contract.json",
    }
    assert {"features.csv", "model.joblib", "evaluation_report.md", "model_card.md"} <= set(
        m["scientist"]["artifacts"]
    )
    assert m["input"]["sha256"] == state.input_sha256
    assert m["selection"]["selected_candidate_id"] == "baseline_transition"
    assert m["selection"]["primary_metric"] == "macro_f1"
    assert m["selection"]["primary_metric_value"] is not None
    # Every recorded artifact carries a hash + size (the manifest ties artifacts to hashes).
    for art in {**m["analyst"]["artifacts"], **m["scientist"]["artifacts"]}.values():
        assert art["sha256"] and art["size_bytes"] > 0


# --------------------------------------------------------------------------- #
# 2. Raw validation failure prevents BOTH the Analyst and Scientist crew calls.
# --------------------------------------------------------------------------- #
def test_raw_validation_failure_blocks_both_crews(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(F, "run_analyst_crew", _analyst_fake(calls, synthetic_raw_path))
    monkeypatch.setattr(F, "run_scientist_crew", _scientist_fake(calls))

    df = pd.read_csv(synthetic_raw_path, sep=";")
    df.loc[0, "page 1 (main category)"] = 99  # outside the allowed {1,2,3,4}
    broken = tmp_path / "broken_raw.csv"
    df.to_csv(broken, sep=";", index=False)

    state = F.run_flow(
        broken, run_id="flow-raw", engine="crew", pin_hash=False, artifact_root=tmp_path
    )

    assert state.status is F.RunStatus.FAILED
    assert state.failed_stage == "prepare"
    assert calls == []  # neither crew ever ran
    assert not (_run_dir(tmp_path, "flow-raw") / "run_manifest.json").exists()
    fr = json.loads((_run_dir(tmp_path, "flow-raw") / "failure_report.json").read_text())
    assert fr["failed_step"] == "prepare"
    assert fr["downstream_blocked"] is True
    assert fr["validation_rules"]  # at least one contract rule reported


# --------------------------------------------------------------------------- #
# 3. Missing Analyst artifact routes to failure and prevents the Scientist call.
# --------------------------------------------------------------------------- #
def test_missing_analyst_artifact_blocks_scientist(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def drop_insights(analyst_dir: Path) -> None:
        (analyst_dir / "insights.md").unlink()

    monkeypatch.setattr(
        F, "run_analyst_crew", _analyst_fake(calls, synthetic_raw_path, mutate=drop_insights)
    )
    monkeypatch.setattr(F, "run_scientist_crew", _scientist_fake(calls))

    state = F.run_flow(
        synthetic_raw_path,
        run_id="flow-miss",
        engine="crew",
        pin_hash=False,
        artifact_root=tmp_path,
    )

    assert state.status is F.RunStatus.FAILED
    assert state.failed_stage == "analyst_gate"
    assert calls == ["analyst_crew"]  # Scientist crew never called
    assert not (_run_dir(tmp_path, "flow-miss") / "run_manifest.json").exists()
    assert "artifact_missing" in {e.get("rule") for e in state.validation_errors}


# --------------------------------------------------------------------------- #
# 4. Corrupted dataset_contract.json routes to failure and prevents the Scientist call.
# --------------------------------------------------------------------------- #
def test_corrupted_contract_blocks_scientist(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def corrupt_contract(analyst_dir: Path) -> None:
        (analyst_dir / "dataset_contract.json").write_text("{ not a valid contract }", "utf-8")

    monkeypatch.setattr(
        F, "run_analyst_crew", _analyst_fake(calls, synthetic_raw_path, mutate=corrupt_contract)
    )
    monkeypatch.setattr(F, "run_scientist_crew", _scientist_fake(calls))

    state = F.run_flow(
        synthetic_raw_path,
        run_id="flow-contract",
        engine="crew",
        pin_hash=False,
        artifact_root=tmp_path,
    )

    assert state.status is F.RunStatus.FAILED
    assert state.failed_stage == "analyst_gate"
    assert calls == ["analyst_crew"]  # Scientist crew never called
    assert not (_run_dir(tmp_path, "flow-contract") / "run_manifest.json").exists()
    assert "contract_mismatch" in {e.get("rule") for e in state.validation_errors}


# --------------------------------------------------------------------------- #
# 5. A missing model feature fails the model gate BEFORE any manifest is published.
# --------------------------------------------------------------------------- #
def test_missing_feature_fails_model_gate_before_publish(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def drop_predictor(scientist_dir: Path) -> None:
        f = scientist_dir / "features.csv"
        df = pd.read_csv(f)
        df.drop(columns=[feat.PREDICTOR_COLUMNS[0]]).to_csv(f, index=False)

    monkeypatch.setattr(F, "run_analyst_crew", _analyst_fake(calls, synthetic_raw_path))
    monkeypatch.setattr(F, "run_scientist_crew", _scientist_fake(calls, mutate=drop_predictor))

    state = F.run_flow(
        synthetic_raw_path,
        run_id="flow-feat",
        engine="crew",
        pin_hash=False,
        artifact_root=tmp_path,
    )

    assert state.status is F.RunStatus.FAILED
    assert state.failed_stage == "model_gate"
    assert calls == ["analyst_crew", "scientist_crew"]
    assert not (_run_dir(tmp_path, "flow-feat") / "run_manifest.json").exists()
    assert "features_schema" in {e.get("rule") for e in state.validation_errors}


# --------------------------------------------------------------------------- #
# 6. A corrupted/unknown model artifact fails safely WITHOUT untrusted loading.
# --------------------------------------------------------------------------- #
def test_corrupted_model_fails_without_untrusted_load(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def corrupt_model(scientist_dir: Path) -> None:
        (scientist_dir / "model.joblib").write_bytes(b"not a real joblib pickle payload")

    monkeypatch.setattr(F, "run_analyst_crew", _analyst_fake(calls, synthetic_raw_path))
    monkeypatch.setattr(F, "run_scientist_crew", _scientist_fake(calls, mutate=corrupt_model))

    state = F.run_flow(
        synthetic_raw_path,
        run_id="flow-model",
        engine="crew",
        pin_hash=False,
        artifact_root=tmp_path,
    )

    assert state.status is F.RunStatus.FAILED
    assert state.failed_stage == "model_gate"
    assert calls == ["analyst_crew", "scientist_crew"]
    assert not (_run_dir(tmp_path, "flow-model") / "run_manifest.json").exists()
    # The trusted-hash pre-check caught the tamper before any joblib.load happened.
    assert "model_artifact_untrusted" in {e.get("rule") for e in state.validation_errors}


# --------------------------------------------------------------------------- #
# 7. An unexpected exception produces a sanitized failure state + a local log.
# --------------------------------------------------------------------------- #
def test_unexpected_exception_is_sanitized_and_logged(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        F,
        "run_analyst_crew",
        _analyst_fake(calls, synthetic_raw_path, exc=RuntimeError("boom-secretless")),
    )
    monkeypatch.setattr(F, "run_scientist_crew", _scientist_fake(calls))

    state = F.run_flow(
        synthetic_raw_path,
        run_id="flow-exc",
        engine="crew",
        pin_hash=False,
        artifact_root=tmp_path,
    )

    assert state.status is F.RunStatus.FAILED
    assert state.failed_stage == "analyst_crew"
    assert calls == ["analyst_crew"]  # Scientist crew never called
    assert not (_run_dir(tmp_path, "flow-exc") / "run_manifest.json").exists()
    assert "RuntimeError" in state.failure_detail

    # Full traceback is kept locally in flow.log ...
    log_text = (_run_dir(tmp_path, "flow-exc") / "flow.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in log_text
    assert "boom-secretless" in log_text

    # ... but the failure report is sanitized (no raw traceback, generic cause).
    fr = json.loads((_run_dir(tmp_path, "flow-exc") / "failure_report.json").read_text())
    assert "Traceback (most recent call last)" not in json.dumps(fr)
    assert fr["cause"].startswith("an unexpected error")


# --------------------------------------------------------------------------- #
# 8. A repeated run id never silently overwrites a prior successful run.
# --------------------------------------------------------------------------- #
def test_repeated_run_id_does_not_overwrite(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(F, "run_analyst_crew", _analyst_fake(calls, synthetic_raw_path))
    monkeypatch.setattr(F, "run_scientist_crew", _scientist_fake(calls))

    first = F.run_flow(
        synthetic_raw_path, run_id="dup", engine="crew", pin_hash=False, artifact_root=tmp_path
    )
    assert first.status is F.RunStatus.SUCCESS
    manifest = _run_dir(tmp_path, "dup") / "run_manifest.json"
    assert manifest.exists()
    original_bytes = manifest.read_bytes()
    original_mtime = manifest.stat().st_mtime_ns
    calls_after_first = len(calls)

    second = F.run_flow(
        synthetic_raw_path, run_id="dup", engine="crew", pin_hash=False, artifact_root=tmp_path
    )

    assert second.status is F.RunStatus.FAILED
    assert second.failed_stage == "prepare"
    assert "reuse" in second.failure_detail
    assert len(calls) == calls_after_first  # no crew ran on the second attempt
    # The prior run is untouched (never overwritten, never deleted).
    assert manifest.read_bytes() == original_bytes
    assert manifest.stat().st_mtime_ns == original_mtime
    # The rejection is reported to a separate directory, not the protected run.
    rejected = list((tmp_path / "runs").glob("dup__rejected-*/failure_report.json"))
    assert rejected


# --------------------------------------------------------------------------- #
# 9. The deterministic engine runs the entire Flow with NO LLM and writes a
#    manifest whose recorded artifact hashes match the bytes on disk. This is the
#    offline, reproducible path used to generate the committed run manifest.
# --------------------------------------------------------------------------- #
def test_deterministic_engine_end_to_end_writes_manifest(
    tmp_path: Path, synthetic_raw_path: Path, no_network: None
) -> None:
    state = F.run_flow(
        synthetic_raw_path,
        run_id="flow-det",
        engine="deterministic",
        pin_hash=False,
        artifact_root=tmp_path,
    )

    assert state.status is F.RunStatus.SUCCESS
    assert [e.stage for e in state.stage_events] == EXPECTED_ORDER
    assert all(e.status == "passed" for e in state.stage_events)

    manifest = _run_dir(tmp_path, "flow-det") / "run_manifest.json"
    assert manifest.exists()
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["status"] == "success"
    assert m["engine"] == "deterministic"
    assert m["gates"] == {"analyst_gate": "passed", "model_gate": "passed"}
    assert m["input"]["sha256"] == state.input_sha256
    assert m["selection"]["primary_metric"] == "macro_f1"
    assert m["selection"]["primary_metric_value"] is not None

    # Integrity: every artifact hash recorded in the manifest matches the bytes on
    # disk, so the manifest is a trustworthy fingerprint of the produced run.
    for name, art in m["analyst"]["artifacts"].items():
        assert d.sha256_file(tmp_path / "analyst" / name) == art["sha256"]
        assert art["size_bytes"] > 0
    for name, art in m["scientist"]["artifacts"].items():
        assert d.sha256_file(tmp_path / "scientist" / name) == art["sha256"]
        assert art["size_bytes"] > 0
