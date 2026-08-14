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
locked dependency set — for local development **and** CI. CI still honours the rest of
the Stage 7 requirement: it runs on **Python 3.11**, installs the **locked**
dependencies, runs lint/format/type/test, and uses **no secrets and no paid LLM calls**.

## Consequences

- CI installs with `pip install -r requirements.txt && pip install -e . --no-deps` — the
  same pinned path documented in the README, so "works in CI" implies "works locally."
- **Residual risk:** the lock was frozen on 3.13, so a pinned wheel could in principle
  fail to resolve on CI's 3.11. This is verified by the actual GitHub Actions run; the
  documented fallback is to pin CI to the lock's native version (3.13) or regenerate the
  lock on 3.11. `requires-python` is `>=3.11,<3.14`, and `ruff` targets `py311`, so the
  source is kept 3.11-compatible regardless.
- `uv` remains usable ad hoc (it is even present in the frozen lock) but is not required
  by, or wired into, any project workflow.

## References

- `CLAUDE.md` (conventions), `requirements.txt`, `.github/workflows/ci.yml`,
  `pyproject.toml` (`requires-python`, `[tool.ruff] target-version`).
