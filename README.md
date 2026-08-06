# Retail Clickstream AI

Industry-simulated AI product workflow that predicts the **next main product
category** within an online-shopping session, using a **CrewAI Flow** that
orchestrates a Data Analyst crew and a Data Scientist crew over deterministic
Python pipelines and validators, surfaced through a Streamlit app.

> **Status: Stage 1 complete** (project scaffold + frozen acceptance criteria).
> Later-stage commands below are marked _pending_ until their stage is built.

## Scope

- **In scope:** CrewAI (OpenAI runtime), Pandas, scikit-learn, Matplotlib /
  Seaborn, Streamlit, Git/GitHub with pull requests, reproducible local runs.
- **Out of scope:** Supabase, Flask, deployment, Docker.

## Prerequisites

- **Python 3.11+** (developed and tested on 3.13).
- **Git**.
- An **OpenAI API key** — required **only** to run the CrewAI Flow/crews
  (Stage 3+). It is never needed to install, import, or run the offline tests.

## Install (venv + pip)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For a pinned, reproducible environment from the committed lock instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Configuration

Copy `.env.example` to `.env` and fill in values locally (never commit `.env`):

| Variable            | Purpose                                             |
| ------------------- | --------------------------------------------------- |
| `OPENAI_API_KEY`    | Required only for LLM-backed runs (Stage 3+).       |
| `OPENAI_MODEL_NAME` | Any OpenAI model available to your account.          |
| `LOG_LEVEL`         | Optional; defaults to `INFO`.                        |
| `ARTIFACT_ROOT`     | Optional; defaults to `artifacts`.                   |

## Tests & quality

```bash
pytest              # offline smoke tests — no key, no network
ruff check .        # lint
ruff format --check .
mypy                # type check (targets retail_clickstream_ai/ via config)
```

## Later-stage commands (pending)

- **Raw-data validation** — _pending Stage 2_.
- **Analyst-only / Scientist-only runs** — _pending Stage 3–4_.
- **Full CrewAI Flow** — _pending Stage 5_.
- **Streamlit app** — `streamlit run app.py` (placeholder until _Stage 6_).

## Dataset

Clickstream Data for Online Shopping — UCI dataset #553 / Kaggle
`tunguz/clickstream-data-for-online-shopping`, licensed **CC BY 4.0**. Five
months of 2008 clicks from an online clothing shop (semicolon-delimited,
165,474 rows, 14 columns).

> **2008 data is used for workflow demonstration only — it is not evidence of
> current shopping behavior.** Acquisition instructions arrive in Stage 2
> (`data/README.md`); the raw CSV is never committed.

## Repository status & workflow

See [`docs/acceptance_checklist.md`](docs/acceptance_checklist.md) for the
rubric → evidence map. Development uses three feature branches
(`feature/data-analyst`, `feature/model-flow`, `feature/app-release`) with
documented pull requests. The staged implementation plan is preserved at
[`docs/reference/00_project_implementation_plan.md`](docs/reference/00_project_implementation_plan.md).
