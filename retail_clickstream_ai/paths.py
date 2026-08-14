"""Canonical filesystem paths for the project.

Every module imports paths from here instead of hard-coding strings, so the
repository layout has a single source of truth. Reading these values performs
no I/O and requires no configuration. The artifact location may be overridden
with the ``ARTIFACT_ROOT`` environment variable (useful for tests).
"""

from __future__ import annotations

import os
from pathlib import Path

# This file lives at ``retail_clickstream_ai/paths.py`` (flat layout); the
# project root is the package's parent directory.
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_ROOT.parent

# Static, layout-fixed directories.
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"

DOCS_DIR: Path = PROJECT_ROOT / "docs"

DEFAULT_ARTIFACT_ROOT = "artifacts"

# Canonical evidence-reference strings embedded verbatim in generated reports so
# every reported number cites its machine-readable source file. They name the
# committed artifact layout and are display-only: unlike ``artifact_root()`` they
# intentionally do not follow the ``ARTIFACT_ROOT`` override (which only changes
# where a run writes), because a published report should cite the canonical path.
ANALYST_EDA_METRICS_REF = "artifacts/analyst/eda/eda_metrics.json"
SCIENTIST_METRICS_REF = "artifacts/scientist/metrics.json"
SCIENTIST_METADATA_REF = "artifacts/scientist/model_metadata.json"


def artifact_root() -> Path:
    """Return the artifact root, honoring the ``ARTIFACT_ROOT`` override.

    A relative override is resolved against the project root; an absolute
    override is used as-is.
    """
    raw = (os.environ.get("ARTIFACT_ROOT") or DEFAULT_ARTIFACT_ROOT).strip()
    raw = raw or DEFAULT_ARTIFACT_ROOT
    root = Path(raw)
    return root if root.is_absolute() else PROJECT_ROOT / root


def analyst_artifacts() -> Path:
    """Directory for the four required Analyst-crew artifacts."""
    return artifact_root() / "analyst"


def scientist_artifacts() -> Path:
    """Directory for the four required Scientist-crew artifacts."""
    return artifact_root() / "scientist"


def runs_artifacts() -> Path:
    """Directory for per-run logs, manifests, and failure reports."""
    return artifact_root() / "runs"
