# Troubleshooting

Recovery steps for the most common problems. All commands assume the virtual
environment is active (`source .venv/bin/activate`).

## Missing dataset

**Symptom:** a full-data command exits `2` with a message pointing to
`data/README.md`; or `prepare` fails with "the raw input file is missing or empty."

**Cause / fix:** the raw CSV is never committed. Download it and place it at exactly
`data/raw/e-shop clothing 2008.csv` (see [`data/README.md`](../data/README.md) — Kaggle
CLI or manual). Then re-run. The offline test suite does **not** need this file; if
only `pytest` is failing, this is not your problem.

## Invalid contract (raw data fails validation)

**Symptom:** `validate-raw` prints labeled issues and exits non-zero, or the Flow
fails at `prepare` / the Analyst gate with `contract_mismatch` / schema errors.

**Cause / fix:** the file does not match the pinned dataset contract (wrong file,
truncated download, altered columns/values, or an out-of-range category). Verify the
SHA-256 matches the value in `data/README.md`:

```bash
shasum -a 256 "data/raw/e-shop clothing 2008.csv"
python -m retail_clickstream_ai.pipeline.data validate-raw --input "data/raw/e-shop clothing 2008.csv"
```

Re-download if the hash differs. Do not hand-edit the CSV.

## OpenAI configuration missing

**Symptom:** `ConfigurationError: Missing required OpenAI configuration:
OPENAI_API_KEY, OPENAI_MODEL_NAME` when starting an `--engine crew` run.

**Cause / fix:** LLM runs need credentials. Copy `.env.example` to `.env` and set
`OPENAI_API_KEY` and `OPENAI_MODEL_NAME` (names only — never commit `.env`). To run
**without** a key, use the offline path instead:

```bash
python -m retail_clickstream_ai.flow --engine deterministic \
  --input "data/raw/e-shop clothing 2008.csv" --run-id my-run
```

## OpenAI rate limit / transient API error

**Symptom:** an `--engine crew` run errors mid-flow with a 429 / rate-limit or timeout.

**Cause / fix:** the crews make real API calls. Wait and retry, lower concurrency on
your account, or choose a model with more headroom via `OPENAI_MODEL_NAME`. The Flow
fails **closed** — it writes a `failure_report.json` and does not publish a partial
manifest — so a rate-limited run leaves no corrupt artifacts. The deterministic engine
never calls the API and is the reliable path for reproducing results.

## Missing artifact

**Symptom:** the Analyst or model gate fails with `artifact_missing` / "required model
artifacts are missing or empty"; or the app cannot find an artifact.

**Cause / fix:** a required artifact was deleted or written to a different
`ARTIFACT_ROOT`. Regenerate the committed set with the documented deterministic Flow
(README §11), or restore from git:

```bash
git checkout -- artifacts/
```

Check that `ARTIFACT_ROOT` (if set) points where you expect.

## Hash mismatch (tampered or drifted artifact)

**Symptom:** `model_artifact_untrusted` (Flow gate or app), a `ModelIntegrityError` in
the app, or `test_committed_artifacts.py` failing on "hash drift."

**Cause / fix:** `model.joblib`'s bytes do not match `model_metadata.json#artifact_sha256`,
or a committed artifact drifted from the run manifest. **The system refuses to load an
unverified pickle by design.** Regenerate the artifacts from the trusted deterministic
run (README §11) so the model, metadata, and manifest agree again; never bypass the
hash check to load an unknown file.

## Version mismatch (dependencies / Python)

**Symptom:** import errors, resolver conflicts, or `mypy`/`ruff` behaving differently
from CI.

**Cause / fix:** recreate the environment from the pinned lock rather than loose
installs:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

CI runs Python **3.13** (to match the pinned lock — `numpy==2.5.1` needs ≥3.12);
local development uses 3.11–3.13 (`requires-python
>=3.11,<3.14`). If a pinned wheel refuses to install on your Python, use a 3.11–3.13
interpreter, or see [`docs/decisions/0006-venv-pip-ci.md`](decisions/0006-venv-pip-ci.md).
After any intentional dependency change, refresh the lock with
`pip freeze --exclude-editable > requirements.txt`.
