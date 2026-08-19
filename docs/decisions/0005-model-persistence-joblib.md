# ADR 0005 — Model persistence with joblib + trusted-hash loading

**Status:** Accepted · **Applies to:** model artifact ·
**Enforced by:** `validation/artifacts.py`, `flow.py` model gate,
`dashboard/data.py::load_verified_model`

## Context

The trained scikit-learn winner must be persisted as a required artifact and later
reloaded by the Flow gate and the app. `joblib`/`pickle` is the standard sklearn
persistence format, but **loading a pickle from an untrusted source can execute
arbitrary code**.

## Decision

- Persist the winner with **`joblib.dump`** to `artifacts/scientist/model.joblib`.
- Record its **`artifact_sha256`** in `model_metadata.json`, and **round-trip verify**
  (reload + predict + probability parity) at write time.
- **Never load the pickle without first verifying its hash.** Both the Flow model gate
  and the Streamlit app compute `sha256(model.joblib)` and compare it to
  `model_metadata.json#artifact_sha256` **before** any `joblib.load`; a mismatch is
  reported (`model_artifact_untrusted`) and the file is refused.

## Consequences

- **Trust warning is mandatory** wherever the model is documented (README, model card):
  load only the trusted, repo-produced, hash-verified artifact.
- The model hash is **recorded and round-trip-verified**, not pinned as a hard contract
  constant: pickle bytes are reproducible within a fixed environment + seed 42 (they are,
  here) but not guaranteed across environments, so byte-identity is not a correctness
  requirement — prediction/probability parity and metric reproducibility are.
- `features.csv` and `metrics.json` **are** byte-reproducible and used for the strict
  integrity checks.

## References

- `pipeline/modeling.py::lock_winner_and_evaluate_test`, `validation/artifacts.py`,
  `flow.py` (model gate), `dashboard/data.py`, `artifacts/scientist/model_card.md`.
