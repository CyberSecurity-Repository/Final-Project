"""Deterministic exploratory data analysis over the cleaned frame.

Stage 3 scope. Every number a human ever reads in ``insights.md`` or
``eda_report.html`` is computed here first, as machine-readable metrics, so the
prose is always traceable to a source file (the anti-hallucination boundary).
Agents interpret these numbers; they never compute them.

The public entry point is :func:`compute_metrics`, which returns a nested,
JSON-serializable dict with two layers:

* **headline** — the scalar values the reports quote directly, so a report can
  cite a metric by key instead of recomputing it;
* **detailed tables** (category frequency, the next-category transition matrix,
  clicks-per-session buckets, country/colour/price summaries, data quality).

Figures are produced by :func:`make_figures` as self-contained, base64-encoded
PNGs (no external assets) with a manifest, so the HTML report renders offline.
Floating-point values are rounded to a fixed precision so the metrics serialize
byte-identically for the same input.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless, deterministic; must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from retail_clickstream_ai.pipeline import data as d  # noqa: E402
from retail_clickstream_ai.reference import codebook  # noqa: E402

# Round every derived float to this many decimals for stable serialization.
_ROUND = 4
_MAIN_CATEGORIES: tuple[int, ...] = (1, 2, 3, 4)


def _r(value: float) -> float:
    """Round to fixed precision; collapse -0.0 to 0.0 for stable output."""
    out = round(float(value), _ROUND)
    return 0.0 if out == 0.0 else out


def _pct(part: float, whole: float) -> float:
    """Percentage ``part/whole`` in [0, 100], rounded; 0 when ``whole`` is 0."""
    return _r(100.0 * part / whole) if whole else 0.0


def _cat_label(code: int) -> str:
    return codebook.decode(d.MAIN_CATEGORY, code)


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #
def _session_sizes(clean_df: pd.DataFrame) -> pd.Series:
    return clean_df.groupby(d.SESSION_KEY, sort=False).size()


def _clicks_per_session(sizes: pd.Series) -> dict[str, Any]:
    """Summary + fixed, interpretable buckets for the session-depth distribution."""
    values = sizes.to_numpy()
    buckets = {
        "1": int((values == 1).sum()),
        "2": int((values == 2).sum()),
        "3": int((values == 3).sum()),
        "4": int((values == 4).sum()),
        "5": int((values == 5).sum()),
        "6-10": int(((values >= 6) & (values <= 10)).sum()),
        "11-20": int(((values >= 11) & (values <= 20)).sum()),
        "21+": int((values >= 21).sum()),
    }
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": _r(values.mean()),
        "median": _r(float(np.median(values))),
        "std": _r(values.std(ddof=0)),
        "p90": _r(float(np.percentile(values, 90))),
        "p99": _r(float(np.percentile(values, 99))),
        "buckets": buckets,
    }


def _category_frequency(clean_df: pd.DataFrame) -> dict[str, Any]:
    counts = clean_df[d.MAIN_CATEGORY].value_counts()
    total = int(counts.sum())
    freq = {
        str(cat): {
            "code": int(cat),
            "label": _cat_label(int(cat)),
            "count": int(counts.get(cat, 0)),
            "share_pct": _pct(int(counts.get(cat, 0)), total),
        }
        for cat in _MAIN_CATEGORIES
    }
    return {"total": total, "by_category": freq}


def _transitions(clean_df: pd.DataFrame) -> dict[str, Any]:
    """Next-main-category transitions within verified sessions (the target signal).

    ``next`` is the following click's category in the same session; the last
    click of each session has no successor and is dropped. Returns the count
    matrix, the row-normalized probability matrix, the overall probability that
    the next click stays in the same category, and the single strongest
    cross-category (off-diagonal) transition.
    """
    frame = clean_df[[d.SESSION_KEY, d.MAIN_CATEGORY]].copy()
    frame["next"] = frame.groupby(d.SESSION_KEY, sort=False)[d.MAIN_CATEGORY].shift(-1)
    pairs = frame.dropna(subset=["next"])
    pairs = pairs.assign(next=pairs["next"].astype("int64"))
    target_rows = int(len(pairs))

    counts: dict[str, int] = {}
    probs: dict[str, float] = {}
    diagonal = 0
    strongest: dict[str, Any] = {"from_code": 0, "to_code": 0, "prob_pct": 0.0, "count": 0}
    for cur in _MAIN_CATEGORIES:
        cur_rows = pairs[pairs[d.MAIN_CATEGORY] == cur]
        cur_total = int(len(cur_rows))
        for nxt in _MAIN_CATEGORIES:
            n = int((cur_rows["next"] == nxt).sum())
            counts[f"{cur}->{nxt}"] = n
            p = _pct(n, cur_total)
            probs[f"{cur}->{nxt}"] = p
            if cur == nxt:
                diagonal += n
            elif p > strongest["prob_pct"]:
                strongest = {
                    "from_code": cur,
                    "to_code": nxt,
                    "prob_pct": p,
                    "count": n,
                }
    strongest["from_label"] = _cat_label(int(strongest["from_code"]))
    strongest["to_label"] = _cat_label(int(strongest["to_code"]))
    return {
        "target_rows": target_rows,
        "counts": counts,
        "row_normalized_pct": probs,
        "next_same_category_pct": _pct(diagonal, target_rows),
        "strongest_offdiagonal": strongest,
    }


def _coded_frequency(clean_df: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    """Frequency table for a small coded column, decoded to labels, sorted by code."""
    counts = clean_df[column].value_counts()
    total = int(counts.sum())
    return {
        str(int(code)): {
            "code": int(code),
            "label": codebook.decode(column, int(code)),
            "count": int(cnt),
            "share_pct": _pct(int(cnt), total),
        }
        for code, cnt in sorted(counts.items())
    }


def _top_countries(clean_df: pd.DataFrame, top_n: int = 15) -> dict[str, Any]:
    counts = clean_df[d.COUNTRY].value_counts()
    total = int(counts.sum())
    ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
    top = [
        {
            "code": int(code),
            "label": codebook.decode(d.COUNTRY, int(code)),
            "count": int(cnt),
            "share_pct": _pct(int(cnt), total),
        }
        for code, cnt in ordered[:top_n]
    ]
    return {"distinct": int(clean_df[d.COUNTRY].nunique()), "top": top}


def _month_coverage(clean_df: pd.DataFrame) -> dict[str, Any]:
    clicks = clean_df[d.MONTH].value_counts().sort_index()
    sessions = clean_df.groupby(d.MONTH)[d.SESSION_KEY].nunique().sort_index()
    total = int(clicks.sum())
    by_month = {
        str(int(m)): {
            "month": int(m),
            "clicks": int(clicks.get(m, 0)),
            "sessions": int(sessions.get(m, 0)),
            "clicks_share_pct": _pct(int(clicks.get(m, 0)), total),
        }
        for m in sorted(clicks.index)
    }
    busiest = max(by_month.values(), key=lambda r: r["clicks"])
    return {
        "months": sorted(int(m) for m in clicks.index),
        "by_month": by_month,
        "busiest_month": busiest["month"],
        "busiest_month_clicks": busiest["clicks"],
        "busiest_month_share_pct": busiest["clicks_share_pct"],
    }


def _price_summary(clean_df: pd.DataFrame) -> dict[str, Any]:
    price = clean_df[d.PRICE]
    above = clean_df[d.PRICE_2]
    above_avg = int((above == 1).sum())
    return {
        "min": int(price.min()),
        "max": int(price.max()),
        "mean": _r(price.mean()),
        "median": _r(float(price.median())),
        "std": _r(price.std(ddof=0)),
        "distinct": int(price.nunique()),
        "above_category_avg_pct": _pct(above_avg, int(len(above))),
    }


def _product_codes(clean_df: pd.DataFrame) -> dict[str, Any]:
    first_letter = clean_df[d.CLOTHING_MODEL].str[0]
    dist = first_letter.value_counts().sort_index()
    return {
        "distinct": int(clean_df[d.CLOTHING_MODEL].nunique()),
        "first_letter_distribution": {str(k): int(v) for k, v in dist.items()},
    }


def _data_quality(clean_df: pd.DataFrame) -> dict[str, Any]:
    """Computed data-quality summary, including the cross-column code anomaly.

    The first letter of the product code tracks the main category. We compute
    the modal letter per category and count rows whose product code deviates —
    surfacing the known single anomaly as a *computed* fact, not a claim.
    """
    letters = clean_df[d.CLOTHING_MODEL].str[0]
    mismatch_total = 0
    by_category: dict[str, dict[str, Any]] = {}
    for cat in _MAIN_CATEGORIES:
        mask = clean_df[d.MAIN_CATEGORY] == cat
        cat_letters = letters[mask]
        if cat_letters.empty:
            continue
        modal = str(cat_letters.mode().iloc[0])
        deviating = int((cat_letters != modal).sum())
        mismatch_total += deviating
        by_category[str(int(cat))] = {
            "label": _cat_label(int(cat)),
            "modal_letter": modal,
            "deviating_rows": deviating,
        }
    return {
        "nulls": {c: int(clean_df[c].isna().sum()) for c in clean_df.columns},
        "null_total": int(clean_df.isna().sum().sum()),
        "full_row_duplicates": int(clean_df.duplicated().sum()),
        "rows_dropped_in_cleaning": 0,
        "session_order_verified": True,
        "cross_column_first_letter": {
            "mismatch_total": mismatch_total,
            "by_category": by_category,
        },
    }


def compute_metrics(
    clean_df: pd.DataFrame,
    *,
    input_sha256: str,
    clean_sha256: str,
    contract_version: str,
) -> dict[str, Any]:
    """Compute the full, deterministic EDA metrics dict for ``clean_df``."""
    rows = int(clean_df.shape[0])
    sizes = _session_sizes(clean_df)
    sessions = int(sizes.shape[0])

    category = _category_frequency(clean_df)
    transitions = _transitions(clean_df)
    clicks = _clicks_per_session(sizes)
    months = _month_coverage(clean_df)
    countries = _top_countries(clean_df)
    price = _price_summary(clean_df)
    products = _product_codes(clean_df)
    quality = _data_quality(clean_df)

    top_cat = max(category["by_category"].values(), key=lambda r: r["count"])
    sale_share = category["by_category"]["4"]["share_pct"]
    top_country = countries["top"][0]
    single_click_pct = _pct(clicks["buckets"]["1"], sessions)

    headline = {
        "rows": rows,
        "sessions": sessions,
        "clicks": rows,
        "target_rows": transitions["target_rows"],
        "months_count": len(months["months"]),
        "months_list": months["months"],
        "days_count": int(clean_df[d.DAY].nunique()),
        "distinct_countries": countries["distinct"],
        "distinct_products": products["distinct"],
        "distinct_colours": int(clean_df[d.COLOUR].nunique()),
        "clicks_per_session_mean": clicks["mean"],
        "clicks_per_session_median": clicks["median"],
        "clicks_per_session_max": clicks["max"],
        "single_click_session_pct": single_click_pct,
        "top_category_code": top_cat["code"],
        "top_category_label": top_cat["label"],
        "top_category_share_pct": top_cat["share_pct"],
        "sale_category_share_pct": sale_share,
        "next_same_category_pct": transitions["next_same_category_pct"],
        "strongest_transition_from_label": transitions["strongest_offdiagonal"]["from_label"],
        "strongest_transition_to_label": transitions["strongest_offdiagonal"]["to_label"],
        "strongest_transition_prob_pct": transitions["strongest_offdiagonal"]["prob_pct"],
        "top_country_code": top_country["code"],
        "top_country_label": top_country["label"],
        "top_country_share_pct": top_country["share_pct"],
        "busiest_month": months["busiest_month"],
        "busiest_month_clicks": months["busiest_month_clicks"],
        "busiest_month_share_pct": months["busiest_month_share_pct"],
        "price_min": price["min"],
        "price_max": price["max"],
        "price_median": price["median"],
        "price_above_category_avg_pct": price["above_category_avg_pct"],
        "null_total": quality["null_total"],
        "duplicate_rows": quality["full_row_duplicates"],
        "first_letter_mismatch_count": quality["cross_column_first_letter"]["mismatch_total"],
    }

    return {
        "provenance": {
            "input_sha256": input_sha256,
            "clean_sha256": clean_sha256,
            "contract_version": contract_version,
            "computed_from": "artifacts/analyst/clean_data.csv",
            "time_period": "April–August 2008 (one online clothing shop)",
        },
        "headline": headline,
        "counts": {
            "rows": rows,
            "sessions": sessions,
            "clicks": rows,
            "columns": int(clean_df.shape[1]),
            "target_rows": transitions["target_rows"],
        },
        "date_coverage": months,
        "clicks_per_session": clicks,
        "main_category": category,
        "transitions": transitions,
        "country": countries,
        "colour": {"by_code": _coded_frequency(clean_df, d.COLOUR)},
        "location": {"by_code": _coded_frequency(clean_df, d.LOCATION)},
        "model_photography": {"by_code": _coded_frequency(clean_df, d.MODEL_PHOTOGRAPHY)},
        "price_2": {"by_code": _coded_frequency(clean_df, d.PRICE_2)},
        "page": {"by_code": _coded_frequency(clean_df, d.PAGE)},
        "price": price,
        "product_code": products,
        "data_quality": quality,
    }


# --------------------------------------------------------------------------- #
# Figures (self-contained base64 PNGs)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Figure:
    """One rendered figure and its evidence provenance."""

    figure_id: str
    title: str
    caption: str
    kind: str
    evidence_ref: str
    png_base64: str

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "title": self.title,
            "caption": self.caption,
            "kind": self.kind,
            "evidence_ref": self.evidence_ref,
            "encoding": "image/png;base64 (embedded in eda_report.html)",
        }


def _encode(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_METRICS_REF = "artifacts/analyst/eda/eda_metrics.json"


def make_figures(clean_df: pd.DataFrame, metrics: dict[str, Any]) -> list[Figure]:
    """Render at least five legible figures from the cleaned frame and metrics."""
    sns.set_theme(style="whitegrid", context="notebook")
    figures: list[Figure] = []

    # 1. Main-category frequency.
    cat = metrics["main_category"]["by_category"]
    labels = [cat[str(c)]["label"] for c in _MAIN_CATEGORIES]
    counts = [cat[str(c)]["count"] for c in _MAIN_CATEGORIES]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=labels, y=counts, hue=labels, palette="viridis", legend=False, ax=ax)
    ax.set(title="Clicks by main product category", xlabel="Main category", ylabel="Clicks")
    figures.append(
        Figure(
            "fig_category_frequency",
            "Clicks by main product category",
            "How often each main category is viewed across all sessions.",
            "bar",
            f"{_METRICS_REF}#main_category.by_category",
            _encode(fig),
        )
    )

    # 2. Clicks-per-session buckets.
    buckets = metrics["clicks_per_session"]["buckets"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=list(buckets.keys()),
        y=list(buckets.values()),
        hue=list(buckets.keys()),
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set(
        title="Session depth (clicks per session)", xlabel="Clicks per session", ylabel="Sessions"
    )
    figures.append(
        Figure(
            "fig_session_depth",
            "Session depth (clicks per session)",
            "Distribution of how many clicks each session contains.",
            "bar",
            f"{_METRICS_REF}#clicks_per_session.buckets",
            _encode(fig),
        )
    )

    # 3. Next-category transition heatmap (row-normalized probabilities).
    probs = metrics["transitions"]["row_normalized_pct"]
    matrix = np.array(
        [[probs[f"{cur}->{nxt}"] for nxt in _MAIN_CATEGORIES] for cur in _MAIN_CATEGORIES]
    )
    tick = [_cat_label(c) for c in _MAIN_CATEGORIES]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="magma",
        xticklabels=tick,
        yticklabels=tick,
        cbar_kws={"label": "P(next | current) %"},
        ax=ax,
    )
    ax.set(
        title="Next-category transition probabilities (%)",
        xlabel="Next category",
        ylabel="Current category",
    )
    figures.append(
        Figure(
            "fig_transitions",
            "Next-category transition probabilities",
            "Row-normalized probability of the next click's category given the current one.",
            "heatmap",
            f"{_METRICS_REF}#transitions.row_normalized_pct",
            _encode(fig),
        )
    )

    # 4. Clicks per month.
    by_month = metrics["date_coverage"]["by_month"]
    months = sorted(int(m) for m in by_month)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        x=months,
        y=[by_month[str(m)]["clicks"] for m in months],
        marker="o",
        ax=ax,
    )
    ax.set(title="Clicks per month (Apr–Aug 2008)", xlabel="Month", ylabel="Clicks")
    ax.set_xticks(months)
    figures.append(
        Figure(
            "fig_month_activity",
            "Clicks per month",
            "Total clicks in each covered month of 2008.",
            "line",
            f"{_METRICS_REF}#date_coverage.by_month",
            _encode(fig),
        )
    )

    # 5. Top countries.
    top = metrics["country"]["top"][:12]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=[c["count"] for c in top],
        y=[c["label"] for c in top],
        hue=[c["label"] for c in top],
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.set(title="Top countries by clicks", xlabel="Clicks", ylabel="Country (decoded)")
    figures.append(
        Figure(
            "fig_top_countries",
            "Top countries by clicks",
            "The most active decoded IP-origin countries.",
            "bar",
            f"{_METRICS_REF}#country.top",
            _encode(fig),
        )
    )

    # 6. Price distribution.
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(clean_df[d.PRICE], bins=20, color="#4c72b0", ax=ax)
    ax.set(title="Product price distribution (US$)", xlabel="Price (US$)", ylabel="Clicks")
    figures.append(
        Figure(
            "fig_price_distribution",
            "Product price distribution",
            "Distribution of listed product prices in US dollars.",
            "histogram",
            f"{_METRICS_REF}#price",
            _encode(fig),
        )
    )

    return figures


def figure_manifest(figures: list[Figure]) -> dict[str, Any]:
    """Machine-readable manifest describing each rendered figure."""
    return {
        "figure_count": len(figures),
        "figures": [f.manifest_entry() for f in figures],
    }


def write_tables(metrics: dict[str, Any], tables_dir: str | Path) -> list[Path]:
    """Write the EDA table intermediates (CSV) that back the report's numbers."""
    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _csv(name: str, frame: pd.DataFrame) -> None:
        path = out_dir / name
        frame.to_csv(path, index=False, lineterminator="\n")
        written.append(path)

    cat = metrics["main_category"]["by_category"]
    _csv("main_category_frequency.csv", pd.DataFrame(cat.values()))
    _csv(
        "clicks_per_session_buckets.csv",
        pd.DataFrame(
            {
                "bucket": list(metrics["clicks_per_session"]["buckets"]),
                "sessions": list(metrics["clicks_per_session"]["buckets"].values()),
            }
        ),
    )
    probs = metrics["transitions"]["row_normalized_pct"]
    _csv(
        "transition_matrix_pct.csv",
        pd.DataFrame([{"pair": k, "probability_pct": v} for k, v in probs.items()]),
    )
    _csv("top_countries.csv", pd.DataFrame(metrics["country"]["top"]))
    _csv("month_coverage.csv", pd.DataFrame(metrics["date_coverage"]["by_month"].values()))
    return written
