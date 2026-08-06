"""Deterministic data/feature/model pipelines.

All numerically exact work (parsing, cleaning, feature building, training,
metrics, file writes) lives here as tested Python — never inside LLM prompts.
Agents interpret the outputs of these functions; they do not compute them.
"""

from __future__ import annotations
