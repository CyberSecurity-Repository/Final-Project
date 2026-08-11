"""Deterministic serialization helpers shared by the pipeline and the crew tools.

Kept free of any CrewAI import so both the offline deterministic pipeline and the
LLM-backed crew tools can stamp and persist handoff records the same way. A
record's self-reference hash is computed over its content *excluding* its own
path/hash fields, so it is well-defined (no circular dependency) and identical
whichever code path produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retail_clickstream_ai import paths
from retail_clickstream_ai.pipeline import data as d


def rel(path: Path) -> str:
    """Repository-relative POSIX path, or the plain path if it escapes the root."""
    try:
        return Path(path).resolve().relative_to(paths.PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as stable JSON (sorted keys, trailing newline)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def missing_prerequisites(*paths_: Path) -> list[Path]:
    """Return the subset of ``paths_`` that do not exist or are empty.

    Used by a terminal ``write_*`` tool to detect that an agent called it before
    an earlier tool in its own chain produced the artifact it needs, so the tool
    can return an actionable message instead of raising on a missing file.
    """
    return [p for p in paths_ if not p.exists() or p.stat().st_size == 0]


def content_sha(model: Any, self_fields: set[str]) -> str:
    """SHA-256 of a Pydantic record's content, excluding its self-reference fields."""
    content = model.model_dump(exclude=self_fields)
    payload = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return d.sha256_bytes(payload)


def stamp_and_write(model: Any, path: Path, path_field: str, sha_field: str) -> tuple[Any, str]:
    """Stamp a record with its repo-relative path + content hash, re-validate, write.

    Re-validation re-runs the model's PASS invariant now that the self-reference
    fields are populated, so a stamped ``PASS`` record is provably well-formed.
    """
    sha = content_sha(model, {path_field, sha_field})
    data = model.model_dump()
    data[path_field] = rel(path)
    data[sha_field] = sha
    validated = type(model).model_validate(data)
    write_json(path, validated.model_dump())
    return validated, sha
