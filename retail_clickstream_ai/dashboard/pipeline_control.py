"""Background Flow run control — process-wide single-flight, Streamlit-free.

A "Run full pipeline" button in a Streamlit app reruns the whole script on
every interaction, so a `st.session_state` flag alone cannot reliably stop a
double click (or two browser tabs against the same server process) from
starting two Flow runs. This module holds one process-wide lock instead: at
most one Flow run is ever in flight, regardless of how many Streamlit sessions
or reruns ask for one.

The Flow itself runs in a daemon background thread so the UI thread stays
responsive; callers poll :func:`snapshot` / :func:`tail_log` to render
progress. Importing this module starts nothing — a run begins only inside
:func:`start_run`.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from retail_clickstream_ai import paths

_lock = threading.Lock()
_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "active": False,
    "run_id": None,
    "engine": None,
    "input_path": None,
    "started_at": None,
    "ended_at": None,
    "result": None,  # "success" | "failed" | None (still running)
    "failed_stage": None,
    "failure_cause": None,
    "failure_remediation": None,
    "manifest_path": None,
    "failure_report_path": None,
    "error": None,  # sanitized "ExceptionType: message" only — never a traceback
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def is_active() -> bool:
    """True while a Flow run started through this module is still executing."""
    with _state_lock:
        return bool(_state["active"])


def snapshot() -> dict[str, Any]:
    """A copy of the current/last run's state (safe to read from any thread)."""
    with _state_lock:
        return dict(_state)


def tail_log(run_id: str, *, lines: int = 40) -> list[str]:
    """Last ``lines`` of the run's `flow.log`, or `[]` if it doesn't exist yet."""
    log_path = paths.runs_artifacts() / run_id / "flow.log"
    if not log_path.exists():
        return []
    try:
        content = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return content[-lines:]


def start_run(*, input_path: str | Path, engine: str, run_id: str | None = None) -> str | None:
    """Start one Flow run in a background thread.

    Returns the resolved run id, or ``None`` if a run is already active (never
    starts a second concurrent run). ``engine="crew"`` requires OpenAI
    credentials only once the crew actually starts (the Flow's own contract);
    ``engine="deterministic"`` never calls OpenAI.
    """
    if not _lock.acquire(blocking=False):
        return None

    try:
        # Deferred import: importing `flow` is safe (it runs nothing), but
        # keeping it inside the guarded section documents that a real run is
        # about to begin only past this point.
        from retail_clickstream_ai.flow import run_flow

        resolved_run_id = run_id or f"ui-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        with _state_lock:
            _state.update(
                active=True,
                run_id=resolved_run_id,
                engine=engine,
                input_path=str(input_path),
                started_at=_now(),
                ended_at=None,
                result=None,
                failed_stage=None,
                failure_cause=None,
                failure_remediation=None,
                manifest_path=None,
                failure_report_path=None,
                error=None,
            )
    except BaseException:
        _lock.release()
        raise

    def _worker() -> None:
        try:
            final_state = run_flow(input_path, run_id=resolved_run_id, engine=engine)
            with _state_lock:
                _state["result"] = final_state.status.value
                _state["failed_stage"] = final_state.failed_stage or None
                _state["failure_cause"] = final_state.failure_cause or None
                _state["failure_remediation"] = final_state.failure_remediation or None
                _state["manifest_path"] = final_state.manifest_path or None
                _state["failure_report_path"] = final_state.failure_report_path or None
        except Exception as exc:  # noqa: BLE001 - surfaced as a sanitized summary only
            with _state_lock:
                _state["result"] = "failed"
                _state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with _state_lock:
                _state["active"] = False
                _state["ended_at"] = _now()
            _lock.release()

    threading.Thread(target=_worker, daemon=True, name=f"flow-run-{resolved_run_id}").start()
    return resolved_run_id


def _reset_for_tests() -> None:
    """Reset module-level state between unit tests (test-only helper)."""
    if _lock.locked():
        _lock.release()
    with _state_lock:
        _state.update(
            active=False,
            run_id=None,
            engine=None,
            input_path=None,
            started_at=None,
            ended_at=None,
            result=None,
            failed_stage=None,
            failure_cause=None,
            failure_remediation=None,
            manifest_path=None,
            failure_report_path=None,
            error=None,
        )
