# PR self-review

The brief frames the work as **three logical pull requests**. This project's actual
git workflow used **one branch + one PR per stage** (`feature/stage-N-<slug>`, PRs
**#1–#10**, all merged into `main`; Stage 7 is `feature/stage-7-quality-ci`). This
document maps the real, merged work onto the three logical scopes and provides a
reviewer checklist for each. **No PR numbers or URLs are invented**; the numbers
below are the real merged stage PRs.

> Legend for "Reviewer checklist": tick items are the evidence a reviewer should
> confirm — all are already in the repository.

---

## PR A — `feature/data-analyst` (Analyst crew + data contract)

**Realised by:** Stage 1 (#1), Stage 2 (#2), Stage 3 (#3/#4).

**Scope:** project scaffold; dataset acquisition + pinned contract; deterministic
Analyst pipeline (clean → EDA → insights) and the 3-agent Analyst crew with inline
prompt specs.

**Artifacts:** `artifacts/analyst/{clean_data.csv, eda_report.html, insights.md,
dataset_contract.json}` (+ `eda/eda_metrics.json`).

**Tests:** `test_data_contract.py`, `test_cleaning.py`, `test_eda.py`,
`test_analyst_pipeline.py`, `test_analyst_crew.py`, `test_analyst_models.py`,
`test_reports.py`.

**Screenshots:** none (data/report stage; the EDA output is the self-contained
`eda_report.html`).

**Risks:** dataset licensing/attribution (mitigated: CC BY 4.0, not committed,
SHA-256 pinned); cleaning determinism (mitigated: byte-reproducibility test).

**Reviewer checklist:**
- ☑ Raw CSV untracked; SHA-256 pinned in `data/README.md`.
- ☑ `validate-raw` prints `[MATCH]` + `PASS`.
- ☑ Contract carries the cleaned-file hash; `validate_analyst_artifacts` passes on the committed set.
- ☑ Every number in `insights.md` / `eda_report.html` cites a metric key (Evidence recorder).

---

## PR B — `feature/model-flow` (Scientist crew + CrewAI Flow)

**Realised by:** Stage 4 (#5), Stage 5 (#7), crew-handoff robustness (#8/#9).

**Scope:** leakage-safe features + temporal split; baseline + 2 models with fixed
seed; winner locked on validation, tested once; the 3-agent Scientist crew; and the
CrewAI Flow wiring both crews behind deterministic gates with fail-closed routing.

**Artifacts:** `artifacts/scientist/{features.csv, model.joblib, evaluation_report.md,
model_card.md, metrics.json, model_metadata.json, split_manifest.json, leakage_audit.json}`;
run manifest `artifacts/runs/flow-final-deterministic/run_manifest.json`.

**Tests:** `test_scientist_features.py`, `test_scientist_modeling.py`,
`test_scientist_pipeline.py`, `test_scientist_crew.py`, `test_guardrails.py`,
`tests/integration/test_flow.py` (9 gate scenarios incl. deterministic E2E),
`test_committed_artifacts.py`.

**Screenshots:** none (headless); confusion matrices are committed under
`artifacts/scientist/figures/`.

**Risks:** target/feature leakage (mitigated: chronological split ADR 0001, leakage
audit, past-only features); untrusted pickle load (mitigated: trusted-hash pre-check,
ADR 0005); weak-LLM crew handoffs (mitigated: deterministic gates own pass/fail — the
crew cannot force a passing run).

**Reviewer checklist:**
- ☑ No random split; `test_month_split_boundaries_are_exact` passes.
- ☑ Winner chosen on validation macro-F1; test scored once (test macro-F1 0.8195).
- ☑ Model gate refuses a tampered pickle (`model_artifact_untrusted`).
- ☑ Failed gate → `failure_report.json`, downstream blocked, no manifest.
- ☑ Committed manifest hashes match the committed artifacts.

---

## PR C — `feature/app-release` (Streamlit product + quality/CI/docs)

**Realised by:** Stage 6 (#10) and Stage 7 (`feature/stage-7-quality-ci`, this PR).

**Scope:** four-section Streamlit dashboard reading committed artifacts + latest
manifest (hash-verified model load; prediction form from `model_metadata.json`); then
the reproducibility/quality hardening: CI, coverage, hygiene cleanup, README rewrite,
ADRs, troubleshooting, license attribution, rubric, and the committed run manifest.

**Artifacts:** `app.py`, `retail_clickstream_ai/dashboard/`; `.github/workflows/ci.yml`;
`README.md`; `docs/decisions/000{2..6}-*.md`; `docs/troubleshooting.md`;
`docs/rubric_checklist.md`; `LICENSES/DATASET.md`.

**Tests:** `test_app.py` (headless `AppTest`, runs without OpenAI),
`test_dashboard_data.py` (inference validation + trusted load),
`test_dashboard_pipeline_control.py`; full suite green under CI-equivalent checks.

**Screenshots:** the app is verified to start and serve all four sections in the
clean-clone rehearsal (Stage 7 notes); demo capture is Stage 8 scope.

**Risks:** app accidentally running the Flow / calling OpenAI on load (mitigated: it
only reads artifacts, verifies the model hash, never kicks off a run); CI touching
secrets or making paid calls (mitigated: no secrets, network-guarded offline tests,
mocked LLM boundary); pinned-lock resolution on 3.11 (see ADR 0006 — verified by the
actual CI run, documented fallback).

**Reviewer checklist:**
- ☑ `streamlit run app.py` starts; Overview/EDA/Model/Predict all render.
- ☑ App refuses to load a model whose hash does not match its metadata.
- ☑ CI runs on push + PR, Python 3.11, no secrets, no paid calls.
- ☑ README commands are copied from the CLI and actually run.
- ☑ Rubric checklist maps every mandatory requirement to evidence.
