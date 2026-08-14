# Final rubric checklist — requirement → evidence

Every mandatory requirement mapped to concrete, checkable evidence: a **file**, a
**test**, a **command**, or the **run manifest**. Run all offline checks with
`pytest && ruff check . && ruff format --check . && mypy`.

## Mandatory stack

| Requirement | Evidence |
| --- | --- |
| CrewAI (two crews, one Flow) | `retail_clickstream_ai/flow.py`; `crews/analyst/`, `crews/scientist/`; test `tests/integration/test_flow.py` |
| Python 3.11+ | `pyproject.toml` `requires-python = ">=3.11,<3.14"`; CI job on 3.13 (lock's native; see ADR 0006) |
| Git + GitHub with PRs | one branch/PR per stage (PRs #1–#10 merged); `docs/pr_self_review.md`; §18 of README |
| Streamlit | `app.py`, `retail_clickstream_ai/dashboard/`; test `tests/unit/test_app.py` (headless `AppTest`) |
| Pandas / scikit-learn / Matplotlib / Seaborn | `pipeline/eda.py` (pandas + figures), `pipeline/modeling.py` (sklearn) |
| joblib persistence | `pipeline/modeling.py`; ADR `docs/decisions/0005-model-persistence-joblib.md` |
| pytest + lint + types | 186 tests; `.github/workflows/ci.yml`; `pyproject.toml` `[tool.ruff]`/`[tool.mypy]` |

## Eight required artifacts (all tracked)

| Artifact | Path | Verified by |
| --- | --- | --- |
| Clean data | `artifacts/analyst/clean_data.csv` | `test_committed_artifacts.py`; contract `clean_sha256` |
| EDA report | `artifacts/analyst/eda_report.html` | `validate_analyst_artifacts` |
| Insights | `artifacts/analyst/insights.md` | `validate_analyst_artifacts` |
| Dataset contract | `artifacts/analyst/dataset_contract.json` | `test_data_contract.py`; `validate_clean_file_against_contract` |
| Features | `artifacts/scientist/features.csv` | `test_scientist_features.py`; `leakage_audit.json` |
| Model | `artifacts/scientist/model.joblib` | `test_scientist_modeling.py`; `artifact_sha256` round-trip |
| Evaluation report | `artifacts/scientist/evaluation_report.md` | `validate_scientist_artifacts` (Evidence recorder) |
| Model card | `artifacts/scientist/model_card.md` | `validate_scientist_artifacts` |

## Six agents (≥3 per crew, sequential)

| Agent | Runtime prompt |
| --- | --- |
| Source & Quality Analyst | `crews/analyst/specs.py::SOURCE_QUALITY` |
| Data Engineer | `crews/analyst/specs.py::DATA_ENGINEER` |
| EDA & Business Analyst | `crews/analyst/specs.py::EDA_BUSINESS` |
| Contract & Feature Engineer | `crews/scientist/specs.py::CONTRACT_FEATURE_ENGINEER` |
| Model Trainer | `crews/scientist/specs.py::MODEL_TRAINER` |
| Evaluation & Governance Reviewer | `crews/scientist/specs.py::EVALUATION_GOVERNANCE` |

Crew wiring tested in `tests/unit/test_analyst_crew.py`, `tests/unit/test_scientist_crew.py`
(mocked LLM). Full 6-agent Flow verified end-to-end — see "Reproducibility" below.

## Business-critical correctness

| Requirement | Evidence (code → test) |
| --- | --- |
| Contract validation | `validation/contract.py`, `validation/raw.py` → `test_data_contract.py` (17), `test_artifact_validation.py` (7) |
| Session next-click target | `pipeline/features.py::build_features` → `test_scientist_features.py::test_target_is_next_click_within_session`, `::test_final_click_of_each_session_is_dropped` |
| Temporal split (no random split) | `pipeline/features.py::assign_split`, ADR `0001` → `test_scientist_features.py::test_month_split_boundaries_are_exact` |
| Model save/load | `pipeline/modeling.py::lock_winner_and_evaluate_test` → `test_scientist_modeling.py::test_saved_model_round_trips`, `::test_metadata_hashes_match_files` |
| Flow gates (fail-closed) | `flow.py` routers → `tests/integration/test_flow.py` (9 scenarios incl. deterministic E2E) |
| Inference validation | `dashboard/data.py::validate_prediction_inputs` → `test_dashboard_data.py` (invalid category / out-of-range / history-consistency) |
| Trusted model loading | `flow.py` model gate + `dashboard/data.py::load_verified_model` → `test_flow.py::test_corrupted_model_fails_without_untrusted_load`, `test_dashboard_data.py::test_load_verified_model_refuses_tampered_artifact` |

## Reproducibility, CI, and integrity

| Requirement | Evidence |
| --- | --- |
| Seeded, deterministic pipeline | seed 42 (`pipeline/modeling.py`); `--engine deterministic` reproduces the model + manifest |
| Run manifest (final successful run) | `artifacts/runs/flow-final-deterministic/run_manifest.json` (status success, both gates passed, macro-F1 0.8195) |
| Artifact integrity verification | `test_committed_artifacts.py::test_committed_run_manifest_fingerprints_committed_artifacts` (manifest hashes == committed files) |
| Full paid CrewAI Flow works | manual external check on 2026-08-14: `--engine crew` completed successfully (engine=crew, gates passed, macro-F1 0.8195) — see `docs/pr_self_review.md` / Stage-7 notes |
| GitHub Actions CI (no secrets, no paid calls) | `.github/workflows/ci.yml` (push + PR, Python 3.13, offline) |
| No secrets / raw data tracked | `.gitignore` (`.env*`, `kaggle.json`, `*.pem`, `data/raw/*`); scan reports zero tracked keys |
| Offline tests need no key/network | `tests/conftest.py` network guard + telemetry opt-outs; mocked crew/LLM boundaries |

## Honesty & ethics

| Requirement | Evidence |
| --- | --- |
| 2008 caveat stated | README §4/§17, `model_card.md`, `eda_report.html`, app UI |
| No causal / production claims | `model_card.md` (Limitations, Ethical considerations) |
| Every metric traceable to a source file | `reporting/evidence.py` (Evidence recorder); `eda_metrics.json`, `metrics.json` |
