# ADR 0006 — venv + pip (not uv) for development and CI

**Status:** Accepted (Stage 1; reaffirmed Stage 7) · **Applies to:** environment & CI ·
**Enforced by:** `README.md`, `requirements.txt`, `.github/workflows/ci.yml`

## Context

The implementation plan and the Stage 7 prompt both suggest **`uv`** for dependency
management / CI. However, `CLAUDE.md` fixes the project convention as **`venv` + `pip`
(not uv)**, recorded at Stage 1, and the pinned lock (`requirements.txt`, a `pip
freeze`) was produced on Python 3.13. Switching CI to `uv` would introduce a second
toolchain and a second lock format for no functional gain on a solo, offline project.

## Decision

Use **`venv` + `pip`** everywhere, with `requirements.txt` (a `pip freeze`) as the
locked dependency set — for local development **and** CI. CI installs the **locked**
dependencies, runs lint/format/type/test, and uses **no secrets and no paid LLM
calls**. It runs on **Python 3.13** — the version the lock was frozen on — because a
pinned dependency (`numpy==2.5.1`) requires `>=3.12`, so the lock cannot resolve on the
Stage 7 prompt's suggested 3.11 (see Consequences).

## Consequences

- CI installs with `pip install -r requirements.txt && pip install -e . --no-deps` — the
  same pinned path documented in the README, so "works in CI" implies "works locally."
- **Python 3.11 vs. the lock:** the first CI run confirmed the pinned lock cannot
  install on 3.11 (`numpy==2.5.1` requires `>=3.12`), so CI targets **3.13** — the lock's
  native version and the project's actual runtime. This keeps reproducibility (the pinned
  lock, and the exact numpy that produced `model.joblib` and the committed metrics)
  intact, which is the Stage 7 priority. The **source** stays 3.11-compatible
  (`requires-python >=3.11,<3.14`, `ruff` targets `py311`); a user who needs 3.11 can
  install unpinned via `pip install -e ".[dev]"`, which resolves a 3.11-compatible numpy.
- `uv` remains usable ad hoc (it is even present in the frozen lock) but is not required
  by, or wired into, any project workflow.

## References

- `CLAUDE.md` (conventions), `requirements.txt`, `.github/workflows/ci.yml`,
  `pyproject.toml` (`requires-python`, `[tool.ruff] target-version`).
