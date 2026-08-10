# Model Card

**Model:** `baseline_transition` (`current_category_transition_baseline`) — next-main-category classifier.

## Purpose and intended users

- **Purpose:** demonstrate a leakage-safe workflow that predicts the next main product category in an online-shopping session.
- **Intended users:** the project's own analysts and reviewers, as a **teaching / workflow** artifact. It is a demonstration, **not for production** use or real business decisions.

## Training data and provenance

- **Source:** *Clickstream Data for Online Shopping* (UCI dataset 553), one Polish online clothing shop, **April–August 2008**.
- **Training rows:** 99,064 labelled click-to-next-click examples across 13,399 sessions (April–June).
- **Training-data hash:** `2d256c2ecb348002f84f53cb04ba97871464d87a6cc78b9615b265c9c16511bd` (clean_data.csv).
- **Seed:** 42. **Python:** 3.13.9.

## Features and target

- **Target:** `next_main_category` — the next click's category within the same session. Classes: 1=trousers, 2=skirts, 3=blouses, 4=sale.
- **Predictors:** current click attributes (current category, colour, location, pose, price, price-vs-average, page, country) plus past-only session history (previous category, clicks so far, distinct categories so far, count of the current category so far). The raw session id is used for grouping only, never as an input.

## Metrics

- **Validation macro F1 (selection):** 0.8130.
- **Test macro F1 (held-out, once):** 0.8195.
- **Test accuracy:** 82.4%.
- Full per-class metrics and confusion matrices are in `artifacts/scientist/metrics.json`.

## Limitations

- **2008 data.** Behaviour, catalogue, and pricing are from April–August 2008; none of it reflects present-day shopping.
- **One shop, one dominant market.** Results are specific to this retailer and a single dominant country.
- **Browsing only.** No purchases, revenue, or customer identity — the model says nothing about conversion.
- **Category-sticky signal.** A strong same-category baseline is hard to beat; gains over it are modest, and no statistical-significance claim is made.

## Ethical considerations

- Descriptive, non-causal predictions; the model must not be read as explaining *why* shoppers move between categories.
- No demographic, personal, or sensitive attributes are used; country is a coded IP-origin id only.
- No fairness certification or present-day validity is claimed.

## Security and trust

- **`model.joblib` is pickle-based.** Loading a pickle from an untrusted source can execute arbitrary code. Load **only** the trusted, repo-produced, hash-verified artifact.
- **Artifact hash:** `b3a442546a75debcf6dadd33bb957a209185ea0a2a8a7d7934b3e8894701e080` (model.joblib).
- **Round-trip verified:** True (predict + probability parity after reload).
- This is a **non-production**, workflow-demonstration artifact from 2008 data; do not deploy it or use it for real decisions.
