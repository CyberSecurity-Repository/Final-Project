# ADR 0001 — No random row split; use a chronological month-based split

**Status:** Accepted (Stage 2) · **Applies to:** Stages 2 & 4 · **Enforced by:**
`dataset_contract.json` (`split`) + leakage unit tests (Stage 4)

## Context

One training row is a single click `t` in a shopping session. The prediction
target is the **next** click's main category (`t+1`) **within the same
session**. Clicks in a session are therefore highly correlated — they share the
session, its products, prices, time, and country.

The dataset is naturally ordered in time: `year`/`month`/`day` plus a per-session
`order`. Verified coverage is **April–August 2008**, and all four main
categories appear in **every** month (so any month can serve as train,
validation, or test).

## Decision

**A random row split is forbidden.** The project splits **chronologically by
month**:

| Split | Months |
| --- | --- |
| Train | April, May, June (`4, 5, 6`) |
| Validation | July (`7`) |
| Test | August (`8`) |

The winner is chosen on **validation** macro-F1; the **test** month is scored
**once**.

## Why random splitting leaks

- The target is built by shifting within a session (`t → t+1`). If two clicks of
  the same session land in different splits, the training set contains a row
  whose label *is literally another split's feature row*. The model effectively
  sees the answer.
- Even without the shift, session-level regularities (a shopper's price band,
  browsing depth, country) would appear on both sides of the split, inflating
  every score and producing a model that looks great offline and fails on any
  genuinely unseen session.
- A random split also destroys the real use case: predicting the **future** from
  the **past**. Time-based evaluation is the honest analogue of deployment.

## Consequences

- Feature engineering (Stage 4) may use **only** the current and earlier clicks
  of the same session. No value derived from click `t+1` or later may enter the
  features (the target is the sole exception).
- Each session's **last click is dropped** (it has no next-click label):
  `165,474` rows → `141,448` labeled rows across `24,026` verified sessions.
- If a required class were ever missing from a split, we would **stop** and
  document a revised month grouping before training (checked: not an issue here).
- The alternative — a session-grouped random split — would prevent target
  leakage but still not measure forecasting over time. Chronological split is
  strictly more honest for a "what happens next" product, so it wins.

## References

- Contract: `artifacts/analyst/dataset_contract.json` → `split`, `target`,
  `constraints`.
- Leakage controls: `retail_clickstream_ai/pipeline/features.py` and
  `artifacts/scientist/leakage_audit.json`.
