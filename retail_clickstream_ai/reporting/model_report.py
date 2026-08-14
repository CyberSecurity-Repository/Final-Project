"""Deterministic renderers for the Scientist reports.

Turns the machine-computed model metrics into the two required Scientist
documents — ``evaluation_report.md`` and ``model_card.md``. Every number is
emitted through the :class:`~retail_clickstream_ai.reporting.evidence.Evidence`
recorder, so each figure in the prose is backed by (and cited to) a key in
``metrics.json`` or ``model_metadata.json``. The renderers validate their claims
before returning, so an unsupported number raises rather than reaching the file.

This is the deterministic core the Governance Reviewer's
``render_scientist_reports`` tool calls: the agent decides *what* to say, the
renderer guarantees every stated number is machine-backed.
"""

from __future__ import annotations

from typing import Any

from retail_clickstream_ai.paths import SCIENTIST_METADATA_REF, SCIENTIST_METRICS_REF
from retail_clickstream_ai.reporting.evidence import Evidence

METRICS_REF = SCIENTIST_METRICS_REF
METADATA_REF = SCIENTIST_METADATA_REF

# Section markers the Scientist artifact validator asserts are present.
REQUIRED_EVAL_REPORT_MARKERS: tuple[str, ...] = (
    "# Model Evaluation Report",
    "## Problem and target",
    "## Data splits",
    "## Leakage controls",
    "## Baseline and candidate comparison",
    "## Model selection",
    "## Held-out test results",
    "## Confusion-matrix interpretation",
    "## Failure cases and next experiments",
)
REQUIRED_MODEL_CARD_MARKERS: tuple[str, ...] = (
    "# Model Card",
    "## Purpose and intended users",
    "## Training data and provenance",
    "## Features and target",
    "## Metrics",
    "## Limitations",
    "## Ethical considerations",
    "## Security and trust",
    "2008",
    "not for production",
    "pickle",
)

_MONTH_NAMES = {4: "April", 5: "May", 6: "June", 7: "July", 8: "August"}
_CANDIDATE_ORDER = ("baseline_transition", "logistic_regression", "random_forest")


def _weakest_test_class(metrics: dict[str, Any]) -> str:
    """Return the class code (as str) with the lowest test F1 (deterministic)."""
    per_class = metrics["test"]["per_class"]
    return min(sorted(per_class), key=lambda c: (per_class[c]["f1"], c))


def render_evaluation_report(metrics: dict[str, Any]) -> tuple[str, Evidence]:
    """Render ``evaluation_report.md`` and return ``(markdown, evidence)``."""
    ev = Evidence(metrics, METRICS_REF)

    def pct(dotted: str) -> str:
        return ev.number(dotted, fmt="{:.1%}")

    def n(dotted: str) -> str:
        return ev.number(dotted, fmt="{:,}")

    winner = str(metrics["selected_candidate_id"])
    labels = metrics["class_labels"]
    lines: list[str] = []
    add = lines.append

    add("# Model Evaluation Report")
    add("")
    add(
        "Independent evaluation of next-main-category models for one online "
        "clothing shop's 2008 clickstream. Every number below is computed by the "
        f"deterministic pipeline and cited to `{METRICS_REF}`; nothing is estimated."
    )
    add("")

    add("## Problem and target")
    add("")
    add(
        "- **Task:** predict the **next** main product category a session will view, "
        "given the current click and prior clicks only."
    )
    add(
        f"- **Target:** `next_main_category` — the following click's category within the "
        f"same verified session (`shift(-1)`); each session's final click is dropped. "
        f"Classes: {', '.join(f'{c}={labels[c]}' for c in sorted(labels))}."
    )
    add(f"- **Primary selection metric:** {ev.value('primary_metric')} (validation).")
    add("")

    add("## Data splits")
    add("")
    add(
        "Chronological, session-safe split (a random row split is forbidden — it would "
        "leak future clicks of a session into training):"
    )
    add("")
    add("| Split | Months | Rows | Sessions |")
    add("| --- | --- | --- | --- |")
    add(
        f"| Train | April–June | {n('split_counts.train_rows')} | {n('split_counts.train_sessions')} |"
    )
    add(
        f"| Validation | July | {n('split_counts.validation_rows')} | "
        f"{n('split_counts.validation_sessions')} |"
    )
    add(f"| Test | August | {n('split_counts.test_rows')} | {n('split_counts.test_sessions')} |")
    add("")
    add(f"All {len(labels)} target classes are present in every split.")
    add("")

    add("## Leakage controls")
    add("")
    add(
        "- Target is the next click **within the same session**; session boundaries are never crossed."
    )
    add("- Each session's final click is removed (it has no successor to label).")
    add(
        "- Predictors use the current click and prior clicks only; strictly-past aggregates are shifted before any running calculation."
    )
    add(
        "- Encoders and estimators are fit on **training rows only**; the winner is chosen on validation and the test month is evaluated exactly once."
    )
    add("- The raw session id/key is kept for auditing only — never a training input.")
    add(
        "- A deterministic leakage audit (`leakage_audit.json`) must pass before any model is trained."
    )
    add("")

    add("## Baseline and candidate comparison")
    add("")
    add("Validation metrics for every candidate (one baseline + two model variations):")
    add("")
    add("| Candidate | Macro F1 | Accuracy | Weighted F1 | Log loss |")
    add("| --- | --- | --- | --- | --- |")
    for cid in _CANDIDATE_ORDER:
        if cid not in metrics["candidates"]:
            continue
        ll_key = f"candidates.{cid}.log_loss"
        ll = (
            "n/a"
            if metrics["candidates"][cid]["log_loss"] is None
            else ev.number(ll_key, fmt="{:.4f}")
        )
        add(
            f"| `{cid}` | {ev.number(f'candidates.{cid}.macro_f1', fmt='{:.4f}')} | "
            f"{ev.number(f'candidates.{cid}.accuracy', fmt='{:.4f}')} | "
            f"{ev.number(f'candidates.{cid}.weighted_f1', fmt='{:.4f}')} | {ll} |"
        )
    add("")

    add("## Model selection")
    add("")
    add(f"- **Selection rule:** {ev.value('selection_rule')}.")
    add(
        f"- **Selected model:** `{winner}` (`{ev.value('selected_model')}`), chosen by the "
        f"highest **validation** macro F1 = {ev.number('validation.macro_f1', fmt='{:.4f}')}."
    )
    add(
        "- The winner was locked from validation results **before** the test set was "
        "touched; the test evaluation below was performed exactly once."
    )
    add("")

    add("## Held-out test results")
    add("")
    add("Performance of the locked winner on the **August test month** (used once):")
    add("")
    add(f"- **Macro F1:** {ev.number('test.macro_f1', fmt='{:.4f}')}")
    add(f"- **Accuracy:** {pct('test.accuracy')}")
    add(f"- **Weighted F1:** {ev.number('test.weighted_f1', fmt='{:.4f}')}")
    if metrics["test"]["log_loss"] is not None:
        add(f"- **Log loss:** {ev.number('test.log_loss', fmt='{:.4f}')}")
    add("")
    add("Per-class test performance:")
    add("")
    add("| Class | Precision | Recall | F1 | Support |")
    add("| --- | --- | --- | --- | --- |")
    for c in sorted(labels):
        add(
            f"| {c}={labels[c]} | {ev.number(f'test.per_class.{c}.precision', fmt='{:.3f}')} | "
            f"{ev.number(f'test.per_class.{c}.recall', fmt='{:.3f}')} | "
            f"{ev.number(f'test.per_class.{c}.f1', fmt='{:.3f}')} | "
            f"{ev.number(f'test.per_class.{c}.support', fmt='{:,}')} |"
        )
    add("")

    add("## Confusion-matrix interpretation")
    add("")
    weak = _weakest_test_class(metrics)
    add(
        f"The test confusion matrix is in `metrics.json#test.confusion_matrix` "
        f"(figure: `figures/confusion_matrix_test.png`). The weakest class is "
        f"**{weak}={labels[weak]}**, with a test F1 of "
        f"{ev.number(f'test.per_class.{weak}.f1', fmt='{:.3f}')} — the hardest category to "
        "predict as the next click, and the main source of error."
    )
    add(
        "Because browsing is strongly category-sticky, most correct predictions sit on the "
        "diagonal (next click stays in the current category); the informative errors are the "
        "category *switches*."
    )
    add("")

    add("## Failure cases and next experiments")
    add("")
    add(
        "- **Failure mode:** minority/rare next-category transitions (category switches) are under-predicted relative to the dominant same-category case."
    )
    add(
        "- **Next experiments:** add short session-window transition features; try class-weighting or calibrated probabilities; evaluate a sequence model (e.g. a simple RNN) on longer sessions; segment metrics by session depth."
    )
    add(
        "- **Not claimed:** no statistical-significance testing was performed; differences between candidates are reported as-is."
    )
    add("")

    markdown = "\n".join(lines)
    ev.validate()
    return markdown, ev


def render_model_card(metrics: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, Evidence]:
    """Render ``model_card.md`` and return ``(markdown, evidence)``.

    Uses two recorders — metric numbers cite ``metrics.json``, provenance facts
    cite ``model_metadata.json`` — and both are validated before returning.
    """
    ev = Evidence(metrics, METRICS_REF)
    evm = Evidence(metadata, METADATA_REF)
    labels = metrics["class_labels"]
    winner = str(metrics["selected_candidate_id"])
    lines: list[str] = []
    add = lines.append

    add("# Model Card")
    add("")
    add(f"**Model:** `{winner}` (`{ev.value('selected_model')}`) — next-main-category classifier.")
    add("")

    add("## Purpose and intended users")
    add("")
    add(
        "- **Purpose:** demonstrate a leakage-safe workflow that predicts the next main "
        "product category in an online-shopping session."
    )
    add(
        "- **Intended users:** the project's own analysts and reviewers, as a **teaching / "
        "workflow** artifact. It is a demonstration, **not for production** use or real "
        "business decisions."
    )
    add("")

    add("## Training data and provenance")
    add("")
    add(
        "- **Source:** *Clickstream Data for Online Shopping* (UCI dataset 553), one Polish "
        "online clothing shop, **April–August 2008**."
    )
    add(
        f"- **Training rows:** {ev.number('split_counts.train_rows', fmt='{:,}')} labelled "
        f"click-to-next-click examples across "
        f"{ev.number('split_counts.train_sessions', fmt='{:,}')} sessions (April–June)."
    )
    add(f"- **Training-data hash:** `{evm.value('training_data_sha256')}` (clean_data.csv).")
    add(f"- **Seed:** {evm.value('seed')}. **Python:** {evm.value('python_version')}.")
    add("")

    add("## Features and target")
    add("")
    add(
        f"- **Target:** `next_main_category` — the next click's category within the same "
        f"session. Classes: {', '.join(f'{c}={labels[c]}' for c in sorted(labels))}."
    )
    add(
        "- **Predictors:** current click attributes (current category, colour, location, "
        "pose, price, price-vs-average, page, country) plus past-only session history "
        "(previous category, clicks so far, distinct categories so far, count of the current "
        "category so far). The raw session id is used for grouping only, never as an input."
    )
    add("")

    add("## Metrics")
    add("")
    add(f"- **Validation macro F1 (selection):** {ev.number('validation.macro_f1', fmt='{:.4f}')}.")
    add(f"- **Test macro F1 (held-out, once):** {ev.number('test.macro_f1', fmt='{:.4f}')}.")
    add(f"- **Test accuracy:** {ev.number('test.accuracy', fmt='{:.1%}')}.")
    add(f"- Full per-class metrics and confusion matrices are in `{METRICS_REF}`.")
    add("")

    add("## Limitations")
    add("")
    add(
        "- **2008 data.** Behaviour, catalogue, and pricing are from April–August 2008; none of it reflects present-day shopping."
    )
    add(
        "- **One shop, one dominant market.** Results are specific to this retailer and a single dominant country."
    )
    add(
        "- **Browsing only.** No purchases, revenue, or customer identity — the model says nothing about conversion."
    )
    add(
        "- **Category-sticky signal.** A strong same-category baseline is hard to beat; gains over it are modest, and no statistical-significance claim is made."
    )
    add("")

    add("## Ethical considerations")
    add("")
    add(
        "- Descriptive, non-causal predictions; the model must not be read as explaining *why* shoppers move between categories."
    )
    add(
        "- No demographic, personal, or sensitive attributes are used; country is a coded IP-origin id only."
    )
    add("- No fairness certification or present-day validity is claimed.")
    add("")

    add("## Security and trust")
    add("")
    add(
        "- **`model.joblib` is pickle-based.** Loading a pickle from an untrusted source can "
        "execute arbitrary code. Load **only** the trusted, repo-produced, hash-verified "
        "artifact."
    )
    add(f"- **Artifact hash:** `{evm.value('artifact_sha256')}` (model.joblib).")
    add(
        f"- **Round-trip verified:** {evm.value('round_trip_validated')} (predict + probability parity after reload)."
    )
    add(
        "- This is a **non-production**, workflow-demonstration artifact from 2008 data; do "
        "not deploy it or use it for real decisions."
    )
    add("")

    markdown = "\n".join(lines)
    ev.validate()
    evm.validate()
    return markdown, ev
