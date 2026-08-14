"""Unit tests for the background Flow run control.

Every test monkeypatches ``retail_clickstream_ai.flow.run_flow`` so no real
CrewAI Flow, OpenAI call, or write to the real ``artifacts/`` tree ever
happens here.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from retail_clickstream_ai.dashboard import pipeline_control as ctl
from retail_clickstream_ai.flow import FlowState, RunStatus


@pytest.fixture(autouse=True)
def _reset_module_state() -> Any:
    ctl._reset_for_tests()
    yield
    ctl._reset_for_tests()


def _fake_success(*_args: Any, **_kwargs: Any) -> FlowState:
    return FlowState(
        run_id="fake-run",
        status=RunStatus.SUCCESS,
        manifest_path="artifacts/runs/fake-run/run_manifest.json",
    )


def _fake_failure(*_args: Any, **_kwargs: Any) -> FlowState:
    return FlowState(
        run_id="fake-run",
        status=RunStatus.FAILED,
        failed_stage="analyst_gate",
        failure_cause="synthetic failure for a test",
        failure_remediation="none needed — this is a test",
    )


def test_start_run_reaches_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("retail_clickstream_ai.flow.run_flow", _fake_success)

    assert ctl.is_active() is False
    run_id = ctl.start_run(input_path="unused.csv", engine="deterministic", run_id="fake-run")
    assert run_id == "fake-run"

    for _ in range(100):
        if not ctl.is_active():
            break
        time.sleep(0.01)

    snap = ctl.snapshot()
    assert snap["active"] is False
    assert snap["result"] == "success"
    assert snap["manifest_path"] == "artifacts/runs/fake-run/run_manifest.json"


def test_start_run_reaches_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("retail_clickstream_ai.flow.run_flow", _fake_failure)

    ctl.start_run(input_path="unused.csv", engine="crew", run_id="fake-run")
    for _ in range(100):
        if not ctl.is_active():
            break
        time.sleep(0.01)

    snap = ctl.snapshot()
    assert snap["result"] == "failed"
    assert snap["failed_stage"] == "analyst_gate"
    assert snap["failure_cause"] == "synthetic failure for a test"


def test_start_run_never_starts_a_second_concurrent_run(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def _blocking(*_args: Any, **_kwargs: Any) -> FlowState:
        started.set()
        release.wait(timeout=5)
        return _fake_success()

    monkeypatch.setattr("retail_clickstream_ai.flow.run_flow", _blocking)

    first = ctl.start_run(input_path="unused.csv", engine="deterministic", run_id="run-1")
    started.wait(timeout=5)
    assert first == "run-1"
    assert ctl.is_active() is True

    second = ctl.start_run(input_path="unused.csv", engine="deterministic", run_id="run-2")
    assert second is None  # refused: a run is already active

    release.set()
    for _ in range(200):
        if not ctl.is_active():
            break
        time.sleep(0.01)
    assert ctl.snapshot()["run_id"] == "run-1"


def test_start_run_records_sanitized_error_on_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> FlowState:
        raise RuntimeError("boom")

    monkeypatch.setattr("retail_clickstream_ai.flow.run_flow", _boom)

    ctl.start_run(input_path="unused.csv", engine="deterministic", run_id="fake-run")
    for _ in range(100):
        if not ctl.is_active():
            break
        time.sleep(0.01)

    snap = ctl.snapshot()
    assert snap["result"] == "failed"
    assert snap["error"] == "RuntimeError: boom"


def test_tail_log_returns_empty_list_when_absent(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    assert ctl.tail_log("no-such-run") == []
