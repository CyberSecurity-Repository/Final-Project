# Model Evaluation Report

Independent evaluation of next-main-category models for one online clothing shop's 2008 clickstream. Every number below is computed by the deterministic pipeline and cited to `artifacts/scientist/metrics.json`; nothing is estimated.

## Problem and target

- **Task:** predict the **next** main product category a session will view, given the current click and prior clicks only.
- **Target:** `next_main_category` — the following click's category within the same verified session (`shift(-1)`); each session's final click is dropped. Classes: 1=trousers, 2=skirts, 3=blouses, 4=sale.
- **Primary selection metric:** macro_f1 (validation).

## Data splits

Chronological, session-safe split (a random row split is forbidden — it would leak future clicks of a session into training):

| Split | Months | Rows | Sessions |
| --- | --- | --- | --- |
| Train | April–June | 99,064 | 13,399 |
| Validation | July | 30,160 | 4,033 |
| Test | August | 12,224 | 1,552 |

All 4 target classes are present in every split.

## Leakage controls

- Target is the next click **within the same session**; session boundaries are never crossed.
- Each session's final click is removed (it has no successor to label).
- Predictors use the current click and prior clicks only; strictly-past aggregates are shifted before any running calculation.
- Encoders and estimators are fit on **training rows only**; the winner is chosen on validation and the test month is evaluated exactly once.
- The raw session id/key is kept for auditing only — never a training input.
- A deterministic leakage audit (`leakage_audit.json`) must pass before any model is trained.

## Baseline and candidate comparison

Validation metrics for every candidate (one baseline + two model variations):

| Candidate | Macro F1 | Accuracy | Weighted F1 | Log loss |
| --- | --- | --- | --- | --- |
| `baseline_transition` | 0.8130 | 0.8159 | 0.8160 | 0.6749 |
| `logistic_regression` | 0.8125 | 0.8154 | 0.8154 | 0.6567 |
| `random_forest` | 0.8063 | 0.8093 | 0.8093 | 0.6797 |

## Model selection

- **Selection rule:** highest validation macro_f1; tie-break: weighted_f1 desc, accuracy desc, log_loss asc, candidate_id asc.
- **Selected model:** `baseline_transition` (`current_category_transition_baseline`), chosen by the highest **validation** macro F1 = 0.8130.
- The winner was locked from validation results **before** the test set was touched; the test evaluation below was performed exactly once.

## Held-out test results

Performance of the locked winner on the **August test month** (used once):

- **Macro F1:** 0.8195
- **Accuracy:** 82.4%
- **Weighted F1:** 0.8238
- **Log loss:** 0.6527

Per-class test performance:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| 1=trousers | 0.801 | 0.868 | 0.833 | 3,326 |
| 2=skirts | 0.791 | 0.797 | 0.794 | 2,345 |
| 3=blouses | 0.804 | 0.777 | 0.790 | 2,891 |
| 4=sale | 0.886 | 0.838 | 0.861 | 3,662 |

## Confusion-matrix interpretation

The test confusion matrix is in `metrics.json#test.confusion_matrix` (figure: `figures/confusion_matrix_test.png`). The weakest class is **3=blouses**, with a test F1 of 0.790 — the hardest category to predict as the next click, and the main source of error.
Because browsing is strongly category-sticky, most correct predictions sit on the diagonal (next click stays in the current category); the informative errors are the category *switches*.

## Failure cases and next experiments

- **Failure mode:** minority/rare next-category transitions (category switches) are under-predicted relative to the dominant same-category case.
- **Next experiments:** add short session-window transition features; try class-weighting or calibrated probabilities; evaluate a sequence model (e.g. a simple RNN) on longer sessions; segment metrics by session depth.
- **Not claimed:** no statistical-significance testing was performed; differences between candidates are reported as-is.
