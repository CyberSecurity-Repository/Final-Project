# ADR 0003 — Deterministic Python owns computation and gates; agents only interpret

**Status:** Accepted (Stages 3–5) · **Applies to:** crews, pipelines, Flow ·
**Enforced by:** Flow routers (`flow.py`), `reporting/evidence.py`, `validation/`

## Context

LLM agents are useful for interpretation and narrative but are non-deterministic and
can hallucinate numbers. A project that lets an agent compute a metric or decide a
pass/fail gate cannot be reproducible, testable, or trustworthy.

## Decision

Draw a hard **determinism boundary**:

- **All numeric/data work** (cleaning, EDA metrics, feature building, training,
  evaluation, hashing, validation) lives in tested Python under
  `retail_clickstream_ai/pipeline` and `retail_clickstream_ai/validation`.
- **Every pass/fail gate** is owned by a deterministic validator called from a Flow
  router — **never by an LLM**.
- **Agents interpret** the computed results and write prose. Every number in a report
  is emitted through the `Evidence` recorder, which validates the figure against a
  machine-readable metric before it can reach the file.

## Consequences

- The pipeline is **fully reproducible offline**: `--engine deterministic` runs the
  entire Flow with no LLM and produces the same model and manifest as `--engine crew`.
- The crews add value where LLMs are strong — narrative insights, governance framing —
  without ever being on the critical path for correctness.
- A weak or misbehaving LLM cannot produce a passing run with bad artifacts; the
  deterministic gates fail closed (see [[0001-no-random-row-split]] and the Flow tests).

## References

- `flow.py` (routers `analyst_gate` / `model_gate`), `reporting/evidence.py`,
  `validation/artifacts.py`, `tests/integration/test_flow.py`.
