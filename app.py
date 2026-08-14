"""Streamlit entry point.

Run with::

    streamlit run app.py

Page load never runs the CrewAI Flow and never calls OpenAI — it only reads
committed artifacts (via :mod:`retail_clickstream_ai.dashboard.cache`) and
loads the hash-verified model. A real pipeline run starts only from the
explicit, confirmation-gated control inside the Overview section.

All business/pipeline logic lives in :mod:`retail_clickstream_ai.dashboard` —
this file only wires page config, navigation, and section dispatch.
"""

from __future__ import annotations


def main() -> None:
    import streamlit as st

    from retail_clickstream_ai.dashboard import sections

    st.set_page_config(
        page_title="Retail Clickstream AI",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🛍️ Retail Clickstream AI")
    st.sidebar.caption("Next-product-category prediction · workflow demonstration on 2008 data")
    page = st.sidebar.radio(
        "Section",
        options=("Overview", "EDA", "Model evaluation", "Predict next category"),
        key="nav_section",
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Source: [UCI dataset 553](https://archive.ics.uci.edu/dataset/553/"
        "clickstream+data+for+online+shopping) · CC BY 4.0"
    )

    renderers = {
        "Overview": (sections.render_overview, "Overview"),
        "EDA": (sections.render_eda, "EDA"),
        "Model evaluation": (sections.render_evaluation, "Model evaluation"),
        "Predict next category": (sections.render_predict, "Predict next category"),
    }
    render, name = renderers[page]
    sections.safe_section(name, render)


if __name__ == "__main__":
    main()
