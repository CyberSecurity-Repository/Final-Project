"""Deterministic report and figure writers.

Renders the self-contained ``eda_report.html``, ``insights.md``,
``evaluation_report.md``, and ``model_card.md`` from machine-readable metrics so
every number in prose is traceable to a computed source. Implemented in
Stage 3 (Analyst) and Stage 4 (Scientist).
"""

from __future__ import annotations
