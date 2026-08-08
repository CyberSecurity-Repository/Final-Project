"""Analyst crew — Stage 3.

Three sequential agents: Source & Quality Analyst, Data Engineer, and
EDA & Business Analyst. Produces the four required Analyst artifacts:
``clean_data.csv``, ``eda_report.html``, ``insights.md``, ``dataset_contract.json``.

Only the Pydantic handoff ``models`` are re-exported here — importing this
package must stay light (no CrewAI). Import the crew/tools submodules directly
(``from retail_clickstream_ai.crews.analyst.crew import build_analyst_crew``)
when you actually need the LLM wiring.
"""

from __future__ import annotations

from retail_clickstream_ai.crews.analyst.models import (
    AnalystCrewHandoff,
    DataEngineeringHandoff,
    SourceQualityReview,
)

__all__ = [
    "AnalystCrewHandoff",
    "DataEngineeringHandoff",
    "SourceQualityReview",
]
