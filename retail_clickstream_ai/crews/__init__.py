"""CrewAI crew configuration.

The Analyst crew (Stage 3) and Scientist crew (Stage 4) are configured here as
sequential CrewAI processes. Each agent's role/goal/backstory, task boundaries,
allowed tools, stop conditions, and structured outputs come from the inline
runtime prompt specs in ``analyst/specs.py`` and ``scientist/specs.py`` (shared
rules in ``prompts.py``).
"""

from __future__ import annotations
