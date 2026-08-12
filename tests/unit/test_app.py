"""Headless Streamlit tests for `app.py`, using Streamlit's `AppTest`.

Every test either points ``ARTIFACT_ROOT`` at a temp fixture directory or
patches ``load_settings``/``pipeline_control.start_run`` — none of them touch
the real committed ``artifacts/`` tree, start the Flow, or call OpenAI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from retail_clickstream_ai.config import Settings
from retail_clickstream_ai.paths import PROJECT_ROOT

_RUN_TIMEOUT = 60
_APP_PATH = str(PROJECT_ROOT / "app.py")


@pytest.fixture(autouse=True)
def _clear_streamlit_caches() -> Any:
    """`st.cache_data` is a process-global store keyed by function+args, not by
    `ARTIFACT_ROOT` — without this, one test's cached artifact reads would leak
    into the next test's differently-configured fixture directory."""
    from retail_clickstream_ai.dashboard import cache as dash_cache

    dash_cache.clear_all()
    yield
    dash_cache.clear_all()


def _launch() -> AppTest:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT)
    return at


def _goto(at: AppTest, section: str) -> AppTest:
    at.sidebar.radio[0].set_value(section).run(timeout=_RUN_TIMEOUT)
    return at


def _no_openai_settings(artifact_root: str) -> Settings:
    return Settings(
        openai_api_key=None, openai_model_name=None, log_level="INFO", artifact_root=artifact_root
    )


# --------------------------------------------------------------------------- #
# 1. App imports and loads offline without OpenAI credentials.
# --------------------------------------------------------------------------- #
def test_app_loads_offline_without_openai_credentials(
    monkeypatch: pytest.MonkeyPatch, full_artifact_env: Path
) -> None:
    monkeypatch.setattr(
        "retail_clickstream_ai.dashboard.sections.load_settings",
        lambda: _no_openai_settings(str(full_artifact_env)),
    )
    at = _launch()
    assert not at.exception


# --------------------------------------------------------------------------- #
# 2. Overview/EDA/evaluation sections render from fixtures.
# --------------------------------------------------------------------------- #
def test_all_four_sections_render_from_fixtures(full_artifact_env: Path) -> None:
    at = _launch()
    assert not at.exception
    for section in ("Overview", "EDA", "Model evaluation", "Predict next category"):
        _goto(at, section)
        assert not at.exception, f"{section} raised: {at.exception}"


# --------------------------------------------------------------------------- #
# 3. Missing artifacts show a helpful message.
# --------------------------------------------------------------------------- #
def test_missing_artifacts_show_a_helpful_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))  # nothing produced here
    at = _launch()
    assert not at.exception  # Overview degrades gracefully (no crash)
    assert len(at.info) >= 1  # "No Flow run has been recorded..." message

    _goto(at, "EDA")
    assert not at.exception
    assert len(at.warning) >= 1
    assert "isn't available yet" in at.warning[0].value


# --------------------------------------------------------------------------- #
# 4. Hash mismatch blocks model loading.
# --------------------------------------------------------------------------- #
def test_hash_mismatch_blocks_model_loading(full_artifact_env: Path) -> None:
    model_path = full_artifact_env / "scientist" / "model.joblib"
    with model_path.open("ab") as fh:
        fh.write(b"tampered-bytes")

    at = _launch()
    _goto(at, "Predict next category")
    assert not at.exception
    assert any("hash mismatch" in e.value.lower() for e in at.error)
    # The form must not be offered when the model can't be trusted.
    assert len(at.selectbox) == 0


# --------------------------------------------------------------------------- #
# 5. Valid prediction form returns a category and probabilities.
# --------------------------------------------------------------------------- #
def test_valid_prediction_returns_category_and_probabilities(full_artifact_env: Path) -> None:
    at = _launch()
    _goto(at, "Predict next category")
    at.button[0].click().run(timeout=_RUN_TIMEOUT)
    assert not at.exception
    assert len(at.success) == 1
    assert "Predicted next category" in at.success[0].value
    assert len(at.dataframe) >= 1  # the probability table


# --------------------------------------------------------------------------- #
# 6. Invalid category/range is rejected before inference.
# --------------------------------------------------------------------------- #
def test_invalid_cross_field_input_is_rejected_before_inference(full_artifact_env: Path) -> None:
    at = _launch()
    _goto(at, "Predict next category")

    # Widgets alone can't express an out-of-domain code (selectbox/number_input
    # already constrain to valid values), but the model's own cross-field
    # invariant (first click => no previous category) is a real contract rule
    # a naive per-field-only check would miss.
    prev_select = next(
        sb for sb in at.selectbox if sb.label.startswith("Previous click's category")
    )
    prev_select.set_value(1).run(timeout=_RUN_TIMEOUT)  # clicks_so_far stays at its default of 1

    at.button[0].click().run(timeout=_RUN_TIMEOUT)
    assert not at.exception
    assert len(at.success) == 0
    assert len(at.error) >= 1
    assert "previous category" in at.error[0].value.lower()


# --------------------------------------------------------------------------- #
# 7. Pipeline run is not invoked on initial render or an ordinary widget rerun.
# --------------------------------------------------------------------------- #
def test_pipeline_not_invoked_on_initial_render_or_rerun(
    full_artifact_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "retail_clickstream_ai.dashboard.pipeline_control.start_run",
        lambda **kwargs: calls.append(kwargs),
    )

    at = _launch()
    for section in ("EDA", "Model evaluation", "Predict next category", "Overview"):
        _goto(at, section)
    # An ordinary widget interaction (picking the engine) is also not a run.
    at.radio(key="pipeline_engine").set_value("deterministic").run(timeout=_RUN_TIMEOUT)

    assert calls == []


# --------------------------------------------------------------------------- #
# 8. Disabled run control explains missing OpenAI configuration.
# --------------------------------------------------------------------------- #
def test_disabled_run_control_explains_missing_openai_configuration(
    full_artifact_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "retail_clickstream_ai.dashboard.sections.load_settings",
        lambda: _no_openai_settings(str(full_artifact_env)),
    )
    at = _launch()
    at.radio(key="pipeline_engine").set_value("crew").run(timeout=_RUN_TIMEOUT)
    at.checkbox(key="pipeline_confirm").set_value(True).run(timeout=_RUN_TIMEOUT)

    assert at.button(key="pipeline_start").disabled is True
    assert any("OPENAI_API_KEY" in w.value for w in at.warning)


# --------------------------------------------------------------------------- #
# Extra: double-click / duplicate-run guard at the control-panel level.
# --------------------------------------------------------------------------- #
def test_start_button_disabled_while_a_run_is_already_active(
    full_artifact_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("retail_clickstream_ai.dashboard.pipeline_control.is_active", lambda: True)
    at = _launch()
    at.checkbox(key="pipeline_confirm").set_value(True).run(timeout=_RUN_TIMEOUT)
    assert at.button(key="pipeline_start").disabled is True
