# ADR 0004 — Streamlit for the app (over Flask)

**Status:** Accepted · **Applies to:** the product surface ·
**Enforced by:** `app.py` + `retail_clickstream_ai/dashboard/`

## Context

The brief allows **Streamlit or Flask** for the user-facing surface. The surface's
only job is to let a reviewer browse the committed artifacts and run a single
prediction — a read-mostly data app, not a general web service.

## Decision

Use **Streamlit**.

## Why

- **Purpose-built for data apps:** a four-section dashboard (Overview, EDA, Model
  evaluation, Predict) reading artifacts is a few hundred lines with no HTML/JS/CSS,
  no routing, and no template layer.
- **Testable headless:** Streamlit's `AppTest` drives the app in `tests/unit/test_app.py`
  with no browser and no OpenAI key, so the UI is part of the offline suite.
- **Clean separation:** all data access, model loading, and inference validation live
  in a framework-free `dashboard/` service layer; only `dashboard/sections.py` imports
  Streamlit, so the logic is unit-tested independently of the UI.

## Consequences

- It is a **single-user, local** app, not a hardened multi-user web server — which is
  exactly the intended demo scope (out of scope: deployment, Flask, Docker).
- The app never runs the Flow or calls OpenAI on page load; it reads committed
  artifacts and verifies the model hash before loading (see [[0005-model-persistence-joblib]]).

## References

- `app.py`, `retail_clickstream_ai/dashboard/`, `tests/unit/test_app.py`.
