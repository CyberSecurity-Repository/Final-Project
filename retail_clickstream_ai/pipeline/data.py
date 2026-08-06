"""Data acquisition, inspection, cleaning, and contract generation.

Implemented across Stage 2 (raw inspection + dataset contract + validators) and
Stage 3 (deterministic cleaning + EDA intermediates). Handles the semicolon-
delimited clickstream CSV, deterministic column normalization, the verified
session key, and stable ordering by session and click order.
"""

from __future__ import annotations
