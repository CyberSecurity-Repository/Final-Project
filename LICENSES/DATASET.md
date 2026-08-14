# Dataset license and attribution

The software in this repository is licensed under **MIT** (see the top-level
[`LICENSE`](../LICENSE)). The **dataset** is licensed separately and is **not
distributed** in this repository — it is downloaded by the user (see
[`data/README.md`](../data/README.md)). This file provides the required attribution.

## Dataset

**Clickstream Data for Online Shopping**

- **Source (primary):** UCI Machine Learning Repository, dataset **#553** —
  <https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping>
- **Source (mirror):** Kaggle — `tunguz/clickstream-data-for-online-shopping` —
  <https://www.kaggle.com/datasets/tunguz/clickstream-data-for-online-shopping>
- **Authors / donors:** Mariusz Łapczyński and Sylwester Białowąs
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)** —
  <https://creativecommons.org/licenses/by/4.0/>
- **Coverage:** ~165,474 rows of clickstream from one online clothing shop,
  **April–August 2008**.

## Citation

> Łapczyński, M., & Białowąs, S. (2013). *Discovering Patterns of Users' Behaviour
> in an E-shop — Comparison of Consumer Buying Behaviours in Poland and Other European
> Countries.* "Studia Ekonomiczne", nr 151, 144–153.

Dataset donated to the UCI ML Repository (dataset #553).

## Terms (CC BY 4.0, summary)

You may share and adapt the material for any purpose, **provided you give appropriate
credit**, link to the license, and indicate if changes were made. This repository:

- **attributes** the authors and source (this file + `data/README.md` + README §4);
- **indicates changes**: the raw file is deterministically cleaned/normalized into
  `artifacts/analyst/clean_data.csv` and engineered into `artifacts/scientist/features.csv`;
  transformations are recorded in the dataset contract and run artifacts.

## Disclaimer

The data is from **2008** and is used here **only** to demonstrate an AI product
workflow. It is **not** evidence of current shopping behaviour and carries no warranty.
See the model card (`artifacts/scientist/model_card.md`) for limitations and ethics.
