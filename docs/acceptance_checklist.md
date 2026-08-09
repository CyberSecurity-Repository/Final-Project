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
| 1 | CrewAI | Two crews orchestrated by one CrewAI Flow | 3–5 | 🟡 Analyst crew (3 agents) built; Scientist + Flow next |
| 2 | Python | `retail_clickstream_ai/` package, Python 3.11+ | 1 | ✅ |
| 3 | Git + GitHub with Pull Requests | 3 feature branches + 3 documented PRs | 1–8 | 🟡 |
| 4 | Streamlit (or Flask) | `app.py` Streamlit dashboard | 6 | ⬜ |
| 5 | Pandas, scikit-learn, Matplotlib/Seaborn | Declared deps; used in pipelines | 1–4 | 🟡 pandas + Matplotlib/Seaborn used (Stage 3); sklearn Stage 4 |

## Eight required artifacts

| # | Artifact | Path | Stage | Status |
|---|---|---|---|---|
| 1 | Clean data | `artifacts/analyst/clean_data.csv` | 3 | ✅ |
| 2 | EDA report | `artifacts/analyst/eda_report.html` | 3 | ✅ |
| 3 | Insights | `artifacts/analyst/insights.md` | 3 | ✅ |
| 4 | Dataset contract | `artifacts/analyst/dataset_contract.json` | 2–3 | ✅ (now carries the cleaned-file hash) |
| 5 | Features | `artifacts/scientist/features.csv` | 4 | ⬜ |
| 6 | Model | `artifacts/scientist/model.joblib` | 4 | ⬜ |
| 7 | Evaluation report | `artifacts/scientist/evaluation_report.md` | 4 | ⬜ |
| 8 | Model card | `artifacts/scientist/model_card.md` | 4 | ⬜ |

## Six agents (≥3 per crew, sequential)

| Crew | Agent | Runtime prompt | Stage | Status |
|---|---|---|---|---|
| Analyst | Source & Quality Analyst | `crews/analyst/specs.py::SOURCE_QUALITY` | 3 | ✅ |
| Analyst | Data Engineer | `crews/analyst/specs.py::DATA_ENGINEER` | 3 | ✅ |
| Analyst | EDA & Business Analyst | `crews/analyst/specs.py::EDA_BUSINESS` | 3 | ✅ |
| Scientist | Contract & Feature Engineer | `crews/scientist/specs.py::CONTRACT_FEATURE_ENGINEER` | 4 | ⬜ |
| Scientist | Model Trainer | `crews/scientist/specs.py::MODEL_TRAINER` | 4 | ⬜ |
| Scientist | Evaluation & Governance Reviewer | `crews/scientist/specs.py::EVALUATION_GOVERNANCE` | 4 | ⬜ |

## Flow requirements

| # | Requirement | Planned evidence | Stage | Status |
|---|---|---|---|---|
| 1 | Automated Analyst → Scientist handoff | Flow passes typed state only after Analyst gate | 5 | ⬜ |
| 2 | Contract matches cleaned dataset | Deterministic validator in Analyst gate | 3–5 | ✅ validator built (`validation/artifacts.py`); Flow gate wiring Stage 5 |
| 3 | Required features exist before modeling | Deterministic model gate | 4–5 | ⬜ |
| 4 | Reproducibility | Seeds, chronological split, lock, run manifest, hashes, logs | 1–5 | 🟡 |
| 5 | Graceful failure | Failed gate → `failure_report.json` + remediation; downstream blocked | 5 | ⬜ |

## Anti-leakage & correctness controls

| # | Requirement | Planned evidence | Stage | Status |
|---|---|---|---|---|
| 1 | No random row split | Month-based split; decision record | 2/4 | 🟡 |
| 2 | Session-aware next-click target | Target shift within verified session; last click dropped | 4 | ⬜ |
| 3 | Past-only features | Leakage audit + unit tests | 4 | ⬜ |
| 4 | Winner chosen on validation; test used once | Selection on macro F1; single test evaluation | 4 | ⬜ |
| 5 | Trusted model loading | Hash-verified repo artifact only (joblib is pickle-based) | 4/6 | ⬜ |

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
| 1 | Offline tests (no key, no paid LLM) | `pytest` smoke + unit + integration (mocked) | 1–7 | 🟡 |
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
