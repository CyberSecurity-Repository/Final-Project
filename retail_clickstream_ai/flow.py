"""CrewAI Flow orchestration — implemented in Stage 5.

Will define the typed Flow state and wire the pipeline:

    prepare run -> Analyst crew -> Analyst gate -> Scientist crew -> model gate -> manifest

Deterministic validators (not the LLM) own every pass/fail gate. Any failed
validation routes to a terminal failure handler and blocks downstream work.
"""

from __future__ import annotations
