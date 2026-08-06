"""Deterministic validators.

Contract validation, artifact presence/hash checks, leakage audits, and the
Flow's pass/fail gates live here. An LLM never decides whether a schema,
artifact, hash, feature, model, or metric is valid — these functions do, and a
failed validator ends the task and blocks downstream work.
"""

from __future__ import annotations
