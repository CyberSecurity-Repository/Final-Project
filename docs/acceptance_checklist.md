# Acceptance Checklist — Rubric → Evidence

This checklist freezes every mandatory requirement from the brief
(`docs/reference/final_project_brief.html`) and the implementation plan
(`docs/reference/00_project_implementation_plan.md`) as a testable item mapped
to its planned evidence and owning stage. Status is updated at each stage's exit
gate.

Legend: ✅ done · 🟡 in progress · ⬜ pending

## Mandatory stack

| # | Requirement | Planned evidence | Stage | Status |
|---|---|---|---|---|
| 1 | CrewAI | Two crews orchestrated by one CrewAI Flow | 3–5 | ✅ two crews (6 agents) orchestrated by one CrewAI Flow (`flow.py`) |
| 2 | Python | `retail_clickstream_ai/` package, Python 3.11+ | 1 | ✅ |
| 3 | Git + GitHub with Pull Requests | 3 feature branches + 3 documented PRs | 1–8 | 🟡 |
| 4 | Streamlit (or Flask) | `app.py` Streamlit dashboard | 6 | ⬜ |
| 5 | Pandas, scikit-learn, Matplotlib/Seaborn | Declared deps; used in pipelines | 1–4 | ✅ pandas + Matplotlib/Seaborn (Stage 3); scikit-learn (Stage 4) |

## Eight required artifacts

| # | Artifact | Path | Stage | Status |
|---|---|---|---|---|
| 1 | Clean data | `artifacts/analyst/clean_data.csv` | 3 | ✅ |
| 2 | EDA report | `artifacts/analyst/eda_report.html` | 3 | ✅ |
| 3 | Insights | `artifacts/analyst/insights.md` | 3 | ✅ |
| 4 | Dataset contract | `artifacts/analyst/dataset_contract.json` | 2–3 | ✅ (now carries the cleaned-file hash) |
| 5 | Features | `artifacts/scientist/features.csv` | 4 | ✅ |
| 6 | Model | `artifacts/scientist/model.joblib` | 4 | ✅ (winner: current-category transition baseline) |
| 7 | Evaluation report | `artifacts/scientist/evaluation_report.md` | 4 | ✅ |
| 8 | Model card | `artifacts/scientist/model_card.md` | 4 | ✅ |

## Six agents (≥3 per crew, sequential)

| Crew | Agent | Runtime prompt | Stage | Status |
|---|---|---|---|---|
| Analyst | Source & Quality Analyst | `crews/analyst/specs.py::SOURCE_QUALITY` | 3 | ✅ |
| Analyst | Data Engineer | `crews/analyst/specs.py::DATA_ENGINEER` | 3 | ✅ |
| Analyst | EDA & Business Analyst | `crews/analyst/specs.py::EDA_BUSINESS` | 3 | ✅ |
| Scientist | Contract & Feature Engineer | `crews/scientist/specs.py::CONTRACT_FEATURE_ENGINEER` | 4 | ✅ |
| Scientist | Model Trainer | `crews/scientist/specs.py::MODEL_TRAINER` | 4 | ✅ |
| Scientist | Evaluation & Governance Reviewer | `crews/scientist/specs.py::EVALUATION_GOVERNANCE` | 4 | ✅ |

## Flow requirements

| # | Requirement | Planned evidence | Stage | Status |
|---|---|---|---|---|
| 1 | Automated Analyst → Scientist handoff | Flow passes typed state only after Analyst gate | 5 | ✅ `flow.py` routes to the Scientist crew only after the deterministic Analyst gate; call order proven in `tests/integration/test_flow.py` |
| 2 | Contract matches cleaned dataset | Deterministic validator in Analyst gate | 3–5 | ✅ Analyst gate calls `validate_analyst_artifacts` (`validation/artifacts.py`) — the Flow owns the pass/fail |
| 3 | Required features exist before modeling | Deterministic model gate | 4–5 | ✅ model gate calls `validate_scientist_artifacts` (+ trusted-hash pre-check) before any manifest |
| 4 | Reproducibility | Seeds, chronological split, lock, run manifest, hashes, logs | 1–5 | ✅ `artifacts/runs/<run_id>/run_manifest.json` records statuses, durations, hashes, package version; sequential only; no auto-retry on gate failure |
| 5 | Graceful failure | Failed gate → `failure_report.json` + remediation; downstream blocked | 5 | ✅ terminal handler writes `failure_report.json` (failed step, rules, observed, remediation, downstream blocked); 8 Flow integration tests |

## Anti-leakage & correctness controls

| # | Requirement | Planned evidence | Stage | Status |
|---|---|---|---|---|
| 1 | No random row split | Month-based split; decision record | 2/4 | ✅ chronological split enforced + `test_month_split_boundaries_are_exact` |
| 2 | Session-aware next-click target | Target shift within verified session; last click dropped | 4 | ✅ `test_target_is_next_click_within_session`, `test_final_click_of_each_session_is_dropped` |
| 3 | Past-only features | Leakage audit + unit tests | 4 | ✅ `run_leakage_audit` + mutation probe; `test_past_aggregates_cannot_see_future` |
| 4 | Winner chosen on validation; test used once | Selection on macro F1; single test evaluation | 4 | ✅ `lock_winner_and_evaluate_test`; `test_winner_is_top_validation_macro_f1` |
| 5 | Trusted model loading | Hash-verified repo artifact only (joblib is pickle-based) | 4/6 | ✅ round-trip + hash in `validate_scientist_artifacts`; UI-load Stage 6 |

## Final deliverables

| # | Requirement | Planned evidence | Stage | Status |
|---|---|---|---|---|
| 1 | Repo: source, tests, artifacts, docs, reproducible commands | Full repository + README | 1–7 | 🟡 |
| 2 | Business presentation, 10–12 slides (target 11) | `docs/presentation_outline.md` | 8 | ⬜ |
| 3 | Demo recording ≤5 min | `docs/demo_script.md` + `docs/demo_checklist.md` | 8 | ⬜ |
| 4 | Three GitHub PRs | `feature/data-analyst`, `feature/model-flow`, `feature/app-release` | 3/5/8 | 🟡 |

## Quality & CI

| # | Requirement | Planned evidence | Stage | Status |
|---|---|---|---|---|
| 1 | Offline tests (no key, no paid LLM) | `pytest` smoke + unit + integration (mocked) | 1–7 | 🟡 134 tests pass (126 unit + 8 Flow integration, mocked crew boundary); CI wiring Stage 7 |
| 2 | Lint / format / type check | `ruff`, `mypy` configured in `pyproject.toml` | 1 | ✅ |
| 3 | GitHub Actions CI (no secrets, no paid calls) | `.github/workflows/ci.yml` | 7 | ⬜ |
| 4 | No secrets / raw dataset tracked | `.gitignore`; diff scan | 1–7 | ✅ |

## Honesty & ethics guardrails

- The 2008 time period is stated as a limitation in the UI, EDA report, model
  card, and slides. — _Stages 3, 4, 6, 8_
- No causal claims from EDA; no present-day-market or production-readiness
  claims. — _Stages 3, 4, 8_
- Every reported metric is traceable to a machine-readable source file. —
  _Stages 3, 4_
