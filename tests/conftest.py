"""Shared pytest fixtures.

Includes a network guard so the offline test suite fails loudly if any code
attempts a real socket connection. Combined with fresh re-imports in the smoke
tests, this proves the import and configuration paths stay fully offline.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest

# Disable CrewAI/OpenTelemetry background telemetry so importing crewai in the
# offline suite never attempts a network call.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# Suppress CrewAI's version-availability check (a network round-trip) so the Flow
# integration tests stay fully offline and quiet.
os.environ.setdefault("CREWAI_DISABLE_VERSION_CHECK", "true")


class NoNetworkError(RuntimeError):
    """Raised when code attempts network access during an offline test."""


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block outbound socket connections for the duration of a test."""

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise NoNetworkError("Network access is not allowed in offline tests.")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    yield


# --------------------------------------------------------------------------- #
# Synthetic data for the Scientist tests.
#
# The committed fixture (clickstream_sample.csv) has too few rows per month for a
# real train/validation/test model comparison, so these fixtures build a small,
# deterministic, contract-valid dataset spanning all five months with all four
# categories and enough rows/sessions per split.
# --------------------------------------------------------------------------- #
_CATEGORY_LETTER = {1: "A", 2: "B", 3: "C", 4: "P"}


def _synthetic_raw_frame(sessions_per_month: int = 14, clicks: int = 5) -> Any:
    """Build a raw-schema DataFrame (raw headers) that passes contract validation."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    sid = 0
    for month in (4, 5, 6, 7, 8):
        for s in range(sessions_per_month):
            sid += 1
            day = (s % 28) + 1
            for o in range(1, clicks + 1):
                cat = ((o - 1) % 4) + 1
                rows.append(
                    {
                        "year": 2008,
                        "month": month,
                        "day": day,
                        "order": o,
                        "country": 9,
                        "session ID": sid,
                        "page 1 (main category)": cat,
                        "page 2 (clothing model)": f"{_CATEGORY_LETTER[cat]}{(o % 9) + 1}",
                        "colour": ((o + s) % 14) + 1,
                        "location": (o % 6) + 1,
                        "model photography": (o % 2) + 1,
                        "price": 20 + ((o * 7 + s) % 60),
                        "price 2": (o % 2) + 1,
                        "page": (o % 5) + 1,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_clean_frame() -> Any:
    """A contract-valid cleaned frame (all months, all classes) for feature tests."""
    from retail_clickstream_ai.pipeline import cleaning
    from retail_clickstream_ai.pipeline import data as d
    from retail_clickstream_ai.validation.contract import DatasetContract

    raw = d.normalize_columns(_synthetic_raw_frame())
    clean_df, _audit = cleaning.clean_dataframe(
        raw, DatasetContract.build(), require_all_months=True
    )
    return clean_df


@pytest.fixture
def synthetic_raw_path(tmp_path: Any) -> Any:
    """Write a contract-valid synthetic raw CSV (raw headers) and return its path.

    Reused by the Flow integration tests, which drive the whole Flow over a
    tmp ``ARTIFACT_ROOT`` with the crew boundary mocked.
    """
    raw_path = tmp_path / "synthetic_raw.csv"
    _synthetic_raw_frame().to_csv(raw_path, sep=";", index=False)
    return raw_path


@pytest.fixture
def scientist_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """Set ARTIFACT_ROOT to a tmp dir and lay down valid Analyst artifacts.

    Runs the real deterministic Analyst pipeline on a synthetic raw file (with
    ``pin_contract_hash=False``) so the four Analyst artifacts exist and validate,
    then returns the tmp artifact root. Scientist code then runs with
    ``pin_hash=False`` because the synthetic data differs from the pinned file.
    """
    from retail_clickstream_ai.pipeline.analyst_pipeline import run_analyst_pipeline

    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    raw_path = tmp_path / "synthetic_raw.csv"
    _synthetic_raw_frame().to_csv(raw_path, sep=";", index=False)
    run_analyst_pipeline(
        raw_path,
        run_id="analyst-synth",
        pin_contract_hash=False,
        require_all_months=True,
    )
    return tmp_path


@pytest.fixture
def full_artifact_env(scientist_env: Any) -> Any:
    """Extend `scientist_env` with a full deterministic Scientist run.

    Produces a complete, valid set of Analyst + Scientist artifacts under a
    temp ``ARTIFACT_ROOT``, so dashboard tests can render every section from
    committed-shaped fixtures without touching the real repo artifacts.
    """
    from retail_clickstream_ai.pipeline.scientist_pipeline import run_scientist_pipeline

    run_scientist_pipeline(pin_analyst_hash=False)
    return scientist_env
