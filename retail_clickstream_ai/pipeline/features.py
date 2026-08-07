"""Leakage-safe feature engineering — implemented in Stage 4.

Builds next-main-category targets within verified sessions, drops each session's
final click, restricts predictors to the current and prior clicks, and produces
the chronological (month-based) train/validation/test split. No random row
splits: rows in a session are related and random splitting would leak.
"""

from __future__ import annotations
