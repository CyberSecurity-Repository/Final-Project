"""Deterministic renderer for ``insights.md``.

Turns the machine-computed EDA metrics into a scannable, business-facing
insights note. Every number is emitted through the :class:`Evidence` recorder,
so each figure in the prose is backed by — and cited to — a metric key. The
document always separates *observation* (what the data shows) from
*interpretation* (a labelled, plausible reading) and *limitation* (why it may
not generalize), and it always carries the 2008 time-period caveat.
"""

from __future__ import annotations

from typing import Any

from retail_clickstream_ai.paths import ANALYST_EDA_METRICS_REF
from retail_clickstream_ai.reporting.evidence import Evidence

METRICS_REF = ANALYST_EDA_METRICS_REF


def _stickiness_word(pct: float) -> str:
    """Qualitative label for how category-sticky browsing is (interpretation)."""
    if pct >= 70:
        return "strongly"
    if pct >= 50:
        return "moderately"
    return "weakly"


def render_insights(metrics: dict[str, Any]) -> tuple[str, Evidence]:
    """Render ``insights.md`` and return ``(markdown, evidence)``.

    The returned :class:`Evidence` carries every numeric claim; the renderer
    validates it before returning, so an unsupported number raises rather than
    reaching the file.
    """
    ev = Evidence(metrics, METRICS_REF)
    h = metrics["headline"]

    n = ev.number  # count formatter

    def p(key: str) -> str:  # percentage formatter
        return ev.number(key, fmt="{:.1f}%")

    lines: list[str] = []
    add = lines.append

    add("# Retail Clickstream — Business Insights")
    add("")
    add(
        "Evidence-first read of one online clothing shop's clickstream. Every number "
        f"below is computed by the deterministic pipeline and cited to "
        f"`{METRICS_REF}`; nothing here is estimated. Percentages are rounded for "
        "display and link to their exact source value."
    )
    add("")

    # -- Scope ------------------------------------------------------------- #
    add("## What this is")
    add("")
    add(
        f"- **Scale:** {n('headline.rows')} clicks across {n('headline.sessions')} "
        "shopping sessions, from April to August 2008."
    )
    add(
        f"- **Prediction target:** the next main category viewed in a session — "
        f"{n('headline.target_rows')} labelled click-to-next-click examples "
        "(each session's final click has no successor and is dropped)."
    )
    add(
        f"- **Catalogue reach:** {n('headline.distinct_products')} distinct product "
        f"codes in four main categories, browsed from "
        f"{n('headline.distinct_countries')} coded countries of origin."
    )
    add(
        "- **Caveat up front:** this is **2008** data from a **single** European "
        "shop. It demonstrates a workflow; it is **not** evidence about present-day "
        "shoppers or any other retailer."
    )
    add("")

    # -- Data quality ------------------------------------------------------ #
    add("## Data quality")
    add("")
    add(
        f"- No missing values ({n('headline.null_total')} nulls) and no full-row "
        f"duplicates ({n('headline.duplicate_rows')}) — verified against the dataset "
        f"contract. Evidence: `{ev.ref('data_quality')}`."
    )
    add(
        f"- One computed cross-column anomaly: {n('headline.first_letter_mismatch_count')} "
        "row is labelled *skirts* yet carries a product code from the trousers code "
        "family. It is recorded, not silently fixed. Evidence: "
        f"`{ev.ref('data_quality.cross_column_first_letter')}`."
    )
    add(
        "- Click order within every session was verified as a contiguous sequence "
        "before any analysis, so the next-category signal is trustworthy."
    )
    add("")

    # -- Findings ---------------------------------------------------------- #
    add("## Key findings")
    add("")

    add("### Finding — Browsing is category-sticky")
    add(
        f"- **Observation:** after any click, the next click stays in the *same* main "
        f"category {p('headline.next_same_category_pct')} of the time. Evidence: "
        f"`{ev.ref('transitions.next_same_category_pct')}`."
    )
    add(
        f"- **Interpretation:** sessions are {_stickiness_word(h['next_same_category_pct'])} "
        "focused — shoppers dig within a category rather than hop between them."
    )
    add(
        "- **Relevance:** the next-category target is highly learnable; a naive "
        '"same category" baseline will already be hard to beat, so models must '
        "earn their keep on the *switch* cases."
    )
    add("")

    add('### Finding — Trousers lead, but "sale" is a large bucket')
    add(
        f"- **Observation:** the most-viewed category is "
        f"**{ev.value('headline.top_category_label')}** at "
        f"{p('headline.top_category_share_pct')} of clicks, while the *sale* "
        f"category alone accounts for {p('headline.sale_category_share_pct')}. "
        f"Evidence: `{ev.ref('main_category.by_category')}`."
    )
    add(
        "- **Interpretation:** demand is spread across all four categories with no "
        "single one dominating; the promotional *sale* bucket pulls meaningful "
        "attention."
    )
    add(
        "- **Relevance:** merchandising and the model both have to handle a genuine "
        "multi-class mix, not a near-constant majority class."
    )
    add("")

    add("### Finding — The strongest cross-category move")
    add(
        f"- **Observation:** the biggest *switch* between categories is "
        f"**{ev.value('headline.strongest_transition_from_label')} → "
        f"{ev.value('headline.strongest_transition_to_label')}** at "
        f"{p('headline.strongest_transition_prob_pct')} of that category's next "
        f"clicks. Evidence: `{ev.ref('transitions.strongest_offdiagonal')}`."
    )
    add(
        "- **Interpretation (cautious):** this is a descriptive co-view pattern, not "
        "a causal cross-sell effect."
    )
    add(
        "- **Relevance:** a candidate hypothesis for adjacency/recommendation tests — "
        "to be validated experimentally, not assumed."
    )
    add("")

    add("### Finding — Sessions are short but long-tailed")
    add(
        f"- **Observation:** the median session is "
        f"{ev.number('headline.clicks_per_session_median', fmt='{:.0f}')} clicks and "
        f"the mean {ev.number('headline.clicks_per_session_mean', fmt='{:.1f}')}, yet "
        f"the deepest reaches {n('headline.clicks_per_session_max')} clicks; "
        f"{p('headline.single_click_session_pct')} of sessions are a single click. "
        f"Evidence: `{ev.ref('clicks_per_session')}`."
    )
    add(
        "- **Interpretation:** most visits are quick scans with a minority of deep, "
        "engaged sessions — a classic long tail."
    )
    add(
        "- **Relevance:** many sessions offer little context before the next click, "
        "so early-session prediction is the hard, valuable case."
    )
    add("")

    add("### Finding — Traffic is geographically concentrated")
    add(
        f"- **Observation:** **{ev.value('headline.top_country_label')}** accounts for "
        f"{p('headline.top_country_share_pct')} of all clicks. Evidence: "
        f"`{ev.ref('country.top')}`."
    )
    add(
        "- **Interpretation:** this is effectively a single-market shop in this "
        "window; other countries are thin slivers."
    )
    add(
        "- **Limitation:** country-level patterns for minor markets rest on very few "
        "sessions and should not be over-read."
    )
    add("")

    add("### Finding — Pricing sits mid-range and near the category average")
    add(
        f"- **Observation:** listed prices run from "
        f"US${n('headline.price_min')} to US${n('headline.price_max')} with a median "
        f"of US${ev.number('headline.price_median', fmt='{:.0f}')}; "
        f"{p('headline.price_above_category_avg_pct')} of clicks are on items priced "
        f"above their category average. Evidence: `{ev.ref('price')}`."
    )
    add(
        "- **Interpretation:** the assortment is balanced around its own category "
        "averages rather than skewed to premium or clearance pricing."
    )
    add(
        '- **Relevance:** the "above category average" flag is a ready, leakage-safe '
        "feature for the modelling stage."
    )
    add("")

    add("### Finding — Activity peaks early in the window")
    add(
        f"- **Observation:** the busiest month is month "
        f"{ev.number('headline.busiest_month', fmt='{:d}')} with "
        f"{n('headline.busiest_month_clicks')} clicks "
        f"({p('headline.busiest_month_share_pct')} of the total). Evidence: "
        f"`{ev.ref('date_coverage.by_month')}`."
    )
    add(
        "- **Interpretation:** traffic is front-loaded in the covered season; the "
        "chronological train/validation/test split is built to respect this drift "
        "rather than shuffle across it."
    )
    add("")

    # -- Limitations ------------------------------------------------------- #
    add("## Limitations")
    add("")
    add(
        "- **2008 data.** Behaviour, catalogue, and pricing are from April–August "
        "2008; none of it should be read as current-market truth."
    )
    add(
        "- **Description, not causation.** These are observed co-occurrence patterns. "
        "EDA cannot establish why shoppers move between categories."
    )
    add(
        "- **One shop, one dominant market.** Findings are specific to this retailer "
        "and are dominated by a single country."
    )
    add(
        "- **Coded fields.** Country, colour, location, and pose are categorical "
        "codes; labels come from the display-only codebook and carry no ordering."
    )
    add(
        "- **No purchases or revenue.** The data is browsing clicks only — it says "
        "nothing about conversion, baskets, or money."
    )
    add("")

    # -- Evidence index ---------------------------------------------------- #
    add("## Evidence index")
    add("")
    add(f"- Computed metrics: `{METRICS_REF}`")
    add(
        "- Figures + manifest: `artifacts/analyst/eda_report.html`, "
        "`artifacts/analyst/eda/figure_manifest.json`"
    )
    add("- Dataset contract: `artifacts/analyst/dataset_contract.json`")
    add("- Cleaned data: `artifacts/analyst/clean_data.csv`")
    add("")

    markdown = "\n".join(lines)
    ev.validate()  # deterministic guarantee: no unsupported number reaches the file.
    return markdown, ev
