"""Streamlit caching wrappers around :mod:`retail_clickstream_ai.dashboard.data`.

Kept separate from `dashboard.data` so the underlying service functions stay
importable and unit-testable without a Streamlit runtime. Stable file reads use
``st.cache_data``; the verified model (an in-memory fitted estimator, not
serializable data) uses ``st.cache_resource`` per Streamlit's own guidance.

:func:`clear_all` must be called after a pipeline run completes (or on a manual
refresh) — cached reads do not otherwise notice that a background Flow run
just rewrote the underlying artifact files.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from retail_clickstream_ai.dashboard import data as svc


@st.cache_data(show_spinner=False)
def dataset_contract() -> dict[str, Any]:
    return svc.read_dataset_contract()


@st.cache_data(show_spinner=False)
def insights_text() -> str:
    return svc.read_insights()


@st.cache_data(show_spinner=False)
def eda_metrics() -> dict[str, Any]:
    return svc.read_eda_metrics()


@st.cache_data(show_spinner=False)
def eda_report_html() -> str:
    return svc.read_eda_report_html()


@st.cache_data(show_spinner=False)
def eda_figure_manifest() -> dict[str, Any]:
    return svc.read_eda_figure_manifest()


@st.cache_data(show_spinner=False)
def eda_table(name: str) -> Any:
    return svc.read_eda_table(name)


@st.cache_data(show_spinner=False)
def scientist_metrics() -> dict[str, Any]:
    return svc.read_scientist_metrics()


@st.cache_data(show_spinner=False)
def model_metadata() -> dict[str, Any]:
    return svc.read_model_metadata()


@st.cache_data(show_spinner=False)
def model_card_text() -> str:
    return svc.read_model_card()


@st.cache_data(show_spinner=False)
def evaluation_report_text() -> str:
    return svc.read_evaluation_report()


@st.cache_data(show_spinner=False)
def feature_schema() -> dict[str, Any]:
    return svc.read_feature_schema()


@st.cache_data(show_spinner=False)
def prediction_field_specs() -> list[svc.FieldSpec]:
    return svc.prediction_field_specs()


@st.cache_data(show_spinner=False)
def analyst_artifact_statuses() -> list[svc.ArtifactStatus]:
    return svc.analyst_artifact_statuses()


@st.cache_data(show_spinner=False)
def scientist_artifact_statuses() -> list[svc.ArtifactStatus]:
    return svc.scientist_artifact_statuses()


@st.cache_resource(show_spinner="Loading the verified model...")
def verified_model() -> Any:
    return svc.load_verified_model()


def clear_all() -> None:
    """Drop every cached artifact read and the cached model (call after a run)."""
    st.cache_data.clear()
    st.cache_resource.clear()
