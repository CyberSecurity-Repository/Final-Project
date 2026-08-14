"""Deterministic renderer for the self-contained ``eda_report.html``.

Produces a single HTML file with inline CSS and base64-embedded figures — no
external assets — so it renders offline anywhere. The report always contains the
required sections (methodology, dataset & provenance, data quality, descriptive
tables, figures, findings, limitations), and all free-text is HTML-escaped. Every
number in the narrative is emitted through the :class:`Evidence` recorder and
validated before the file is returned; table cells are rendered directly from the
computed metrics, so they are traceable to the metrics file by construction.
"""

from __future__ import annotations

import html
from typing import Any

from retail_clickstream_ai.paths import ANALYST_EDA_METRICS_REF
from retail_clickstream_ai.reporting.evidence import Evidence
from retail_clickstream_ai.validation.contract import DatasetContract

METRICS_REF = ANALYST_EDA_METRICS_REF

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  margin: 0; color: #1b1f24; background: #f7f8fa; line-height: 1.55; }
main { max-width: 980px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.9rem; margin: 0 0 .25rem; }
h2 { font-size: 1.3rem; margin: 2.25rem 0 .75rem; padding-bottom: .3rem;
  border-bottom: 2px solid #e2e5ea; }
h3 { font-size: 1.05rem; margin: 1.25rem 0 .4rem; }
.subtitle { color: #566; margin: 0 0 1rem; }
.callout { background: #fff8e6; border: 1px solid #f0d98c; border-radius: 8px;
  padding: .8rem 1rem; margin: 1rem 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .75rem; margin: 1rem 0; }
.card { background: #fff; border: 1px solid #e2e5ea; border-radius: 8px; padding: .8rem; }
.card .k { font-size: 1.4rem; font-weight: 700; }
.card .l { color: #667; font-size: .82rem; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; background: #fff;
  font-size: .92rem; }
th, td { border: 1px solid #e2e5ea; padding: .4rem .6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #eef1f5; }
figure { margin: 1.5rem 0; background: #fff; border: 1px solid #e2e5ea;
  border-radius: 8px; padding: 1rem; }
figure img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
figcaption { color: #566; font-size: .88rem; margin-top: .5rem; text-align: center; }
code { background: #eef1f5; padding: .1rem .3rem; border-radius: 4px; font-size: .85em; }
footer { margin-top: 3rem; color: #889; font-size: .82rem; }
.tag { display: inline-block; font-size: .72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .04em; color: #556; background: #eef1f5; border-radius: 4px;
  padding: .1rem .4rem; margin-right: .35rem; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _stat_cards(cards: list[tuple[str, str]]) -> str:
    items = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="l">{_esc(label)}</div></div>'
        for k, label in cards
    )
    return f'<div class="grid">{items}</div>'


def render_eda_report(
    metrics: dict[str, Any],
    figures: list[Any],
    *,
    contract: DatasetContract,
) -> tuple[str, Evidence]:
    """Render the HTML report and return ``(html_string, evidence)``."""
    ev = Evidence(metrics, METRICS_REF)
    n = ev.number

    def p(key: str) -> str:
        return ev.number(key, fmt="{:.1f}%")

    raw = contract.data["dataset"]["raw_file"]
    cleaned = contract.cleaned_data
    ds = contract.data["dataset"]

    parts: list[str] = []
    out = parts.append

    out("<main>")
    out("<h1>Retail Clickstream — Exploratory Data Analysis</h1>")
    out(
        '<p class="subtitle">Deterministic EDA of one online clothing shop, '
        "April–August 2008. Predicting the next main product category in a session.</p>"
    )
    out(
        '<div class="callout"><span class="tag">Caveat</span>'
        "This is <strong>2008</strong> data from a <strong>single</strong> shop. "
        "It demonstrates a workflow and is <strong>not</strong> evidence about "
        "present-day shoppers, other retailers, purchases, or revenue. Findings are "
        "descriptive only — no causal claims.</div>"
    )

    out(
        _stat_cards(
            [
                (n("headline.rows"), "clicks"),
                (n("headline.sessions"), "sessions"),
                (n("headline.target_rows"), "labelled examples"),
                (n("headline.distinct_products"), "products"),
                (n("headline.distinct_countries"), "countries"),
                (f"{p('headline.next_same_category_pct')}", "next-click stays in category"),
            ]
        )
    )

    # -- Methodology ------------------------------------------------------- #
    out('<h2 id="methodology">Methodology</h2>')
    out(
        "<p>The raw file is parsed with its verified delimiter and encoding, then "
        "validated column-by-column against a machine-readable dataset contract "
        "(types, ranges, category domains, session integrity). Cleaning is "
        "non-destructive: nothing is imputed, dropped, coerced, or clipped — invalid "
        "data would stop the run. A composite <code>session_key</code> is derived and "
        "rows are sorted into canonical click order. All descriptive numbers below "
        "are computed by deterministic Python and written to "
        f"<code>{_esc(METRICS_REF)}</code> before any prose is written, so every "
        "figure here is traceable to a computed source.</p>"
    )

    # -- Dataset & provenance --------------------------------------------- #
    out('<h2 id="dataset-provenance">Dataset &amp; provenance</h2>')
    out(
        _table(
            ["Field", "Value"],
            [
                ["Dataset", ds["name"]],
                ["Publisher", ds["publisher"]],
                ["License", ds["license"]],
                ["Documented period", ds["documented_time_period"]],
                ["Raw file", raw["name"]],
                ["Raw SHA-256", raw["sha256"]],
                ["Raw rows × cols", f"{raw['row_count']:,} × {raw['column_count']}"],
                ["Cleaned file", cleaned["file"]],
                ["Cleaned SHA-256", cleaned["sha256"]],
                ["Cleaned rows × cols", f"{cleaned['row_count']:,} × {cleaned['column_count']}"],
            ],
        )
    )

    # -- Data quality ------------------------------------------------------ #
    dq = metrics["data_quality"]
    out('<h2 id="data-quality">Data quality</h2>')
    out(
        f"<p>Nulls: <strong>{n('headline.null_total')}</strong>. Full-row duplicates: "
        f"<strong>{n('headline.duplicate_rows')}</strong>. Rows dropped in cleaning: "
        f"<strong>{n('data_quality.rows_dropped_in_cleaning')}</strong>. Session click "
        "order was verified before analysis.</p>"
    )
    out(
        f"<p>One computed cross-column anomaly: "
        f"<strong>{n('headline.first_letter_mismatch_count')}</strong> row is labelled "
        "<em>skirts</em> but carries a trousers-family product code. It is recorded, "
        "not corrected.</p>"
    )
    out(
        _table(
            ["Category", "Modal code letter", "Deviating rows"],
            [
                [v["label"], v["modal_letter"], v["deviating_rows"]]
                for v in dq["cross_column_first_letter"]["by_category"].values()
            ],
        )
    )

    # -- Descriptive tables ----------------------------------------------- #
    out('<h2 id="descriptive-tables">Descriptive tables</h2>')

    out("<h3>Main-category frequency</h3>")
    cat = metrics["main_category"]["by_category"]
    out(
        _table(
            ["Category", "Code", "Clicks", "Share %"],
            [[v["label"], v["code"], f"{v['count']:,}", v["share_pct"]] for v in cat.values()],
        )
    )

    out("<h3>Next-category transition probabilities (%)</h3>")
    probs = metrics["transitions"]["row_normalized_pct"]
    cats = [1, 2, 3, 4]
    labels = {c: cat[str(c)]["label"] for c in cats}
    trans_rows = [[labels[cur]] + [probs[f"{cur}->{nxt}"] for nxt in cats] for cur in cats]
    out(_table(["Current \\ Next"] + [labels[c] for c in cats], trans_rows))

    out("<h3>Clicks per session</h3>")
    cps = metrics["clicks_per_session"]
    out(
        _table(
            ["Bucket", "Sessions"],
            [[k, f"{v:,}"] for k, v in cps["buckets"].items()],
        )
    )
    out(
        f"<p>min {cps['min']}, median {cps['median']}, mean {cps['mean']}, "
        f"p90 {cps['p90']}, p99 {cps['p99']}, max {cps['max']}.</p>"
    )

    out("<h3>Top countries by clicks</h3>")
    out(
        _table(
            ["Country", "Code", "Clicks", "Share %"],
            [
                [c["label"], c["code"], f"{c['count']:,}", c["share_pct"]]
                for c in metrics["country"]["top"][:12]
            ],
        )
    )

    out("<h3>Month coverage</h3>")
    out(
        _table(
            ["Month", "Clicks", "Sessions", "Clicks share %"],
            [
                [v["month"], f"{v['clicks']:,}", f"{v['sessions']:,}", v["clicks_share_pct"]]
                for v in metrics["date_coverage"]["by_month"].values()
            ],
        )
    )

    # -- Figures ----------------------------------------------------------- #
    out('<h2 id="figures">Figures</h2>')
    for fig in figures:
        out("<figure>")
        out(f'<img alt="{_esc(fig.title)}" src="data:image/png;base64,{fig.png_base64}">')
        out(
            f"<figcaption><strong>{_esc(fig.title)}.</strong> {_esc(fig.caption)} "
            f"<br><code>{_esc(fig.evidence_ref)}</code></figcaption>"
        )
        out("</figure>")

    # -- Findings ---------------------------------------------------------- #
    out('<h2 id="findings">Findings</h2>')
    out("<ul>")
    out(
        f"<li><strong>Category-sticky browsing.</strong> The next click stays in the "
        f"same category {p('headline.next_same_category_pct')} of the time — a highly "
        "learnable target where the value is in the switch cases.</li>"
    )
    out(
        f"<li><strong>Balanced category mix.</strong> "
        f"{_esc(ev.value('headline.top_category_label'))} leads at "
        f"{p('headline.top_category_share_pct')}; the <em>sale</em> bucket is "
        f"{p('headline.sale_category_share_pct')}.</li>"
    )
    out(
        f"<li><strong>Strongest cross-category move:</strong> "
        f"{_esc(ev.value('headline.strongest_transition_from_label'))} → "
        f"{_esc(ev.value('headline.strongest_transition_to_label'))} at "
        f"{p('headline.strongest_transition_prob_pct')} (descriptive, not causal).</li>"
    )
    out(
        f"<li><strong>Short, long-tailed sessions.</strong> median "
        f"{n('headline.clicks_per_session_median')} clicks; "
        f"{p('headline.single_click_session_pct')} are single-click.</li>"
    )
    out(
        f"<li><strong>One dominant market.</strong> "
        f"{_esc(ev.value('headline.top_country_label'))} is "
        f"{p('headline.top_country_share_pct')} of clicks.</li>"
    )
    out("</ul>")

    # -- Limitations ------------------------------------------------------- #
    out('<h2 id="limitations">Limitations</h2>')
    out("<ul>")
    out("<li><strong>2008 data</strong> from a single shop — not current-market truth.</li>")
    out(
        "<li><strong>Description, not causation</strong> — EDA cannot explain why "
        "shoppers switch categories.</li>"
    )
    out(
        "<li><strong>One dominant country</strong> — minor-market patterns rest on "
        "few sessions.</li>"
    )
    out(
        "<li><strong>Coded categorical fields</strong> — labels are display-only and "
        "carry no ordering.</li>"
    )
    out("<li><strong>Clicks only</strong> — no purchases, baskets, or revenue.</li>")
    out("</ul>")

    out(
        "<footer>Generated deterministically from the dataset contract and computed "
        f"EDA metrics. Contract version {_esc(contract.data['contract_version'])}. "
        "All numbers trace to <code>" + _esc(METRICS_REF) + "</code>.</footer>"
    )
    out("</main>")

    body = "\n".join(parts)
    document = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Retail Clickstream — EDA Report</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    ev.validate()
    return document, ev
