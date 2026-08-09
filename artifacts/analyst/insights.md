# Retail Clickstream — Business Insights

Evidence-first read of one online clothing shop's clickstream. Every number below is computed by the deterministic pipeline and cited to `artifacts/analyst/eda/eda_metrics.json`; nothing here is estimated. Percentages are rounded for display and link to their exact source value.

## What this is

- **Scale:** 165,474 clicks across 24,026 shopping sessions, from April to August 2008.
- **Prediction target:** the next main category viewed in a session — 141,448 labelled click-to-next-click examples (each session's final click has no successor and is dropped).
- **Catalogue reach:** 217 distinct product codes in four main categories, browsed from 47 coded countries of origin.
- **Caveat up front:** this is **2008** data from a **single** European shop. It demonstrates a workflow; it is **not** evidence about present-day shoppers or any other retailer.

## Data quality

- No missing values (0 nulls) and no full-row duplicates (0) — verified against the dataset contract. Evidence: `artifacts/analyst/eda/eda_metrics.json#data_quality`.
- One computed cross-column anomaly: 1 row is labelled *skirts* yet carries a product code from the trousers code family. It is recorded, not silently fixed. Evidence: `artifacts/analyst/eda/eda_metrics.json#data_quality.cross_column_first_letter`.
- Click order within every session was verified as a contiguous sequence before any analysis, so the next-category signal is trustworthy.

## Key findings

### Finding — Browsing is category-sticky
- **Observation:** after any click, the next click stays in the *same* main category 81.2% of the time. Evidence: `artifacts/analyst/eda/eda_metrics.json#transitions.next_same_category_pct`.
- **Interpretation:** sessions are strongly focused — shoppers dig within a category rather than hop between them.
- **Relevance:** the next-category target is highly learnable; a naive "same category" baseline will already be hard to beat, so models must earn their keep on the *switch* cases.

### Finding — Trousers lead, but "sale" is a large bucket
- **Observation:** the most-viewed category is **trousers** at 30.1% of clicks, while the *sale* category alone accounts for 23.4%. Evidence: `artifacts/analyst/eda/eda_metrics.json#main_category.by_category`.
- **Interpretation:** demand is spread across all four categories with no single one dominating; the promotional *sale* bucket pulls meaningful attention.
- **Relevance:** merchandising and the model both have to handle a genuine multi-class mix, not a near-constant majority class.

### Finding — The strongest cross-category move
- **Observation:** the biggest *switch* between categories is **skirts → blouses** at 9.3% of that category's next clicks. Evidence: `artifacts/analyst/eda/eda_metrics.json#transitions.strongest_offdiagonal`.
- **Interpretation (cautious):** this is a descriptive co-view pattern, not a causal cross-sell effect.
- **Relevance:** a candidate hypothesis for adjacency/recommendation tests — to be validated experimentally, not assumed.

### Finding — Sessions are short but long-tailed
- **Observation:** the median session is 4 clicks and the mean 6.9, yet the deepest reaches 195 clicks; 21.0% of sessions are a single click. Evidence: `artifacts/analyst/eda/eda_metrics.json#clicks_per_session`.
- **Interpretation:** most visits are quick scans with a minority of deep, engaged sessions — a classic long tail.
- **Relevance:** many sessions offer little context before the next click, so early-session prediction is the hard, valuable case.

### Finding — Traffic is geographically concentrated
- **Observation:** **Poland** accounts for 81.0% of all clicks. Evidence: `artifacts/analyst/eda/eda_metrics.json#country.top`.
- **Interpretation:** this is effectively a single-market shop in this window; other countries are thin slivers.
- **Limitation:** country-level patterns for minor markets rest on very few sessions and should not be over-read.

### Finding — Pricing sits mid-range and near the category average
- **Observation:** listed prices run from US$18 to US$82 with a median of US$43; 51.2% of clicks are on items priced above their category average. Evidence: `artifacts/analyst/eda/eda_metrics.json#price`.
- **Interpretation:** the assortment is balanced around its own category averages rather than skewed to premium or clearance pricing.
- **Relevance:** the "above category average" flag is a ready, leakage-safe feature for the modelling stage.

### Finding — Activity peaks early in the window
- **Observation:** the busiest month is month 4 with 48,199 clicks (29.1% of the total). Evidence: `artifacts/analyst/eda/eda_metrics.json#date_coverage.by_month`.
- **Interpretation:** traffic is front-loaded in the covered season; the chronological train/validation/test split is built to respect this drift rather than shuffle across it.

## Limitations

- **2008 data.** Behaviour, catalogue, and pricing are from April–August 2008; none of it should be read as current-market truth.
- **Description, not causation.** These are observed co-occurrence patterns. EDA cannot establish why shoppers move between categories.
- **One shop, one dominant market.** Findings are specific to this retailer and are dominated by a single country.
- **Coded fields.** Country, colour, location, and pose are categorical codes; labels come from the display-only codebook and carry no ordering.
- **No purchases or revenue.** The data is browsing clicks only — it says nothing about conversion, baskets, or money.

## Evidence index

- Computed metrics: `artifacts/analyst/eda/eda_metrics.json`
- Figures + manifest: `artifacts/analyst/eda_report.html`, `artifacts/analyst/eda/figure_manifest.json`
- Dataset contract: `artifacts/analyst/dataset_contract.json`
- Cleaned data: `artifacts/analyst/clean_data.csv`
