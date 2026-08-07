"""Streamlit entry point (full four-section dashboard implemented in Stage 6).

Run with:

    streamlit run app.py

Until Stage 6 this shows a placeholder only. It performs no network access and
makes no LLM calls on load. Business/pipeline logic will live in tested service
modules imported here — never inline in this file.
"""

from __future__ import annotations


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Retail Clickstream AI", page_icon="🛍️")
    st.title("Retail Clickstream AI")
    st.caption("Next-product-category prediction · workflow demonstration on 2008 data")
    st.info(
        "The interactive dashboard (Overview · EDA · Model evaluation · Predict "
        "next category) is implemented in Stage 6. This placeholder confirms the "
        "app entry point is wired up."
    )


if __name__ == "__main__":
    main()
