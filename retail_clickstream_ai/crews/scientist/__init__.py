"""Scientist crew — implemented in Stage 4.

Three sequential agents: Contract & Feature Engineer, Model Trainer, and
Evaluation & Governance Reviewer. Produces the four required Scientist
artifacts: ``features.csv``, ``model.joblib``, ``evaluation_report.md``,
``model_card.md``.
"""

from __future__ import annotations

from retail_clickstream_ai.crews.scientist.crew import (
    ScientistCrewBundle,
    build_scientist_crew,
    run_scientist_crew,
)
from retail_clickstream_ai.crews.scientist.specs import SCIENTIST_SPECS
from retail_clickstream_ai.crews.scientist.tools import build_scientist_tools

__all__ = [
    "SCIENTIST_SPECS",
    "ScientistCrewBundle",
    "build_scientist_crew",
    "build_scientist_tools",
    "run_scientist_crew",
]
