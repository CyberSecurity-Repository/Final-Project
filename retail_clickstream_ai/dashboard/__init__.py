"""Streamlit dashboard support code.

Business/pipeline logic for the app lives here, never inline in the root
``app.py``:

* :mod:`retail_clickstream_ai.dashboard.data` — Streamlit-free artifact reads,
  manifest resolution, and verified-model inference. Fully unit-testable.
* :mod:`retail_clickstream_ai.dashboard.pipeline_control` — process-wide
  single-flight background Flow runs. Also Streamlit-free.
* :mod:`retail_clickstream_ai.dashboard.cache` — thin ``st.cache_data`` /
  ``st.cache_resource`` wrappers around ``data``.
* :mod:`retail_clickstream_ai.dashboard.sections` — the four page renderers.

Importing any of these modules performs no I/O and never starts the Flow.
"""

from __future__ import annotations
