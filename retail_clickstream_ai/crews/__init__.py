"""CrewAI crew configuration.

The Analyst crew (Stage 3) and Scientist crew (Stage 4) are configured here as
sequential CrewAI processes. Each agent's role/goal/backstory, task boundaries,
allowed tools, stop conditions, and structured outputs come from the runtime
prompt files under ``prompts/crewai/``.
"""

from __future__ import annotations
