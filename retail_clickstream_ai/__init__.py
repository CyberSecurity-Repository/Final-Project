"""Retail Clickstream AI — industry-simulated AI product workflow.

Predicts the next main product category within an online-shopping session using
a CrewAI Flow that orchestrates a Data Analyst crew and a Data Scientist crew
over deterministic Python pipelines and validators.

Importing this package performs no network access and requires no secrets.
OpenAI credentials are validated lazily, only when an LLM-backed run starts
(see :func:`retail_clickstream_ai.config.require_openai_config`).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
