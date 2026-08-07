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

PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
PROMPTS_CREWAI: Path = PROMPTS_DIR / "crewai"

DOCS_DIR: Path = PROJECT_ROOT / "docs"

_DEFAULT_ARTIFACT_ROOT = "artifacts"


def artifact_root() -> Path:
    """Return the artifact root, honoring the ``ARTIFACT_ROOT`` override.

    A relative override is resolved against the project root; an absolute
    override is used as-is.
    """
    raw = (os.environ.get("ARTIFACT_ROOT") or _DEFAULT_ARTIFACT_ROOT).strip()
    raw = raw or _DEFAULT_ARTIFACT_ROOT
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
