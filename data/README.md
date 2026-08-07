# Data — acquisition & verification

> **The raw dataset is NOT committed to this repository.** `data/raw/` is
> gitignored (only `.gitkeep` is tracked). You must download the file yourself
> and place it at the path below before running any full-data command.

## Dataset

**Clickstream Data for Online Shopping** — five months (April–August 2008) of
clickstream data from an online clothing shop.

| | |
| --- | --- |
| Primary record | UCI ML Repository, dataset **#553** — <https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping> |
| Kaggle mirror | `tunguz/clickstream-data-for-online-shopping` — <https://www.kaggle.com/datasets/tunguz/clickstream-data-for-online-shopping> |
| Publisher | Mariusz Łapczyński and Sylwester Białowąs |
| License | **CC BY 4.0** (attribution required) |
| Expected file | `data/raw/e-shop clothing 2008.csv` |
| Format | semicolon-delimited (`;`), UTF-8/ASCII, CRLF line endings |
| Size / shape | 6,675,312 bytes · 165,474 rows × 14 columns |
| SHA-256 | `fcc167bbd0badd4c9685bd8543097e318f8228e48075335db7cd781cee88115d` |

> ⚠️ **2008 data is a workflow demonstration only** — it is not evidence of
> current shopping behavior. This limitation is repeated in the EDA report,
> model card, and slides.

## Acquire the file (choose one)

### Option A — Kaggle CLI

Set up Kaggle credentials **outside** this repo (never commit them). The
`kaggle.json` credential file is gitignored here as a safety net.

```bash
# One-time: place your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600).
# Do NOT put credentials in this repository or paste them into a prompt.
kaggle datasets download -d tunguz/clickstream-data-for-online-shopping -p data/raw
unzip -o "data/raw/clickstream-data-for-online-shopping.zip" -d data/raw
rm -f "data/raw/clickstream-data-for-online-shopping.zip"
```

### Option B — Manual download

1. Open the Kaggle or UCI link above and download the archive.
2. Extract `e-shop clothing 2008.csv`.
3. Place it at exactly `data/raw/e-shop clothing 2008.csv`.

## Verify what you downloaded

```bash
# 1. Confirm the file hash matches the expected raw SHA-256.
shasum -a 256 "data/raw/e-shop clothing 2008.csv"
# expected: fcc167bbd0badd4c9685bd8543097e318f8228e48075335db7cd781cee88115d

# 2. Run the deterministic contract validator (activate the venv first).
python -m retail_clickstream_ai.pipeline.data validate-raw \
  --input "data/raw/e-shop clothing 2008.csv"
```

A healthy run prints `[MATCH]` for the SHA-256 and `PASS: no issues.` (exit
code `0`). Any schema, type, range, session-key, or click-order problem is
reported as a labeled issue and exits non-zero. If the file is missing, the
command exits `2` and points back to this README.

## Regenerate the dataset contract

The machine-readable contract at `artifacts/analyst/dataset_contract.json` is
generated **deterministically** (no timestamps) from verified constants:

```bash
python -m retail_clickstream_ai.pipeline.data build-contract
```

## Credentials & safety

Never commit `.env`, an `OPENAI_API_KEY`, or Kaggle credentials. `.gitignore`
already excludes `.env*`, `kaggle.json`, `*.pem`, and `data/raw/*`. Importing the
package or running `pytest` never needs a key or network access — the offline
tests run against a tiny synthetic fixture (`tests/fixtures/clickstream_sample.csv`),
never the raw dataset.
