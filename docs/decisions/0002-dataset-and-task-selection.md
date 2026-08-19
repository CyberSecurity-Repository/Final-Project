# ADR 0002 — Dataset and prediction task

**Status:** Accepted · **Applies to:** whole project · **Enforced by:**
`validation/contract.py` (pinned schema/hashes) + `data/README.md`

## Context

The project needs a real, tabular, **sessionized** dataset that supports an honest
"what happens next" prediction task, is openly licensed, and is small enough to run
offline on a laptop without paid infrastructure.

## Decision

Use **Clickstream Data for Online Shopping** (UCI ML Repository dataset **#553**,
Kaggle mirror `tunguz/clickstream-data-for-online-shopping`), and frame the task as:
**predict the next main product category (`1` trousers, `2` skirts, `3` blouses,
`4` sale) a shopper will view within the same session**, given the current click and
past-only session history.

## Why this dataset and task

- **Sessionized + time-ordered** (`year/month/day` + per-session `order`), which is
  exactly what makes a leakage-safe, forecast-style split meaningful (see
  [[0001-no-random-row-split]]).
- **A well-posed 4-class problem** with all classes present in every month, so any
  month can serve as train / validation / test.
- **CC BY 4.0**, so it can be attributed and used freely (see
  [`LICENSES/DATASET.md`](../../LICENSES/DATASET.md)).
- **Small** (165,474 rows, ~6.4 MB): the whole pipeline runs in minutes with no GPU.

## Consequences

- The data is from **April–August 2008** — a hard limitation that must be stated in
  the EDA report, model card, app, and slides. It is **not** evidence of present-day
  behaviour.
- It captures **browsing only** — no purchases, revenue, or identity — so the model
  makes no conversion or causal claim.
- The raw CSV is **never committed**; it is acquired per `data/README.md` and pinned
  by SHA-256 so any substitution is detected.

## References

- `data/README.md`, `retail_clickstream_ai/validation/contract.py`.
