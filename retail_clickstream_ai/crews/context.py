"""Run context shared by the Analyst crew and its tools.

An :class:`AnalystRunContext` captures the trusted, run-scoped paths and the
input hash for a single Analyst run. It is the single place that resolves the
runtime placeholders (``{run_id}``, ``{artifact_root}``, ``{source_metadata_path}``
…) and enforces the shared-rules boundary that every path must stay inside the
repository root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from retail_clickstream_ai import paths
from retail_clickstream_ai.crews.io_helpers import rel
from retail_clickstream_ai.pipeline import data as d


@dataclass(frozen=True)
class AnalystRunContext:
    """Trusted, run-scoped locations + identity for one Analyst run."""

    run_id: str
    input_path: Path
    input_sha256: str
    model_name: str | None
    repository_root: Path
    artifact_root: Path
    analyst_dir: Path
    eda_dir: Path
    run_dir: Path
    # Run policy: relaxed only for the synthetic fixture, which intentionally
    # differs from the pinned production data.
    pin_hash: bool = True
    require_all_months: bool = True

    # -- derived agent-input file locations -------------------------------- #
    @property
    def source_metadata_path(self) -> Path:
        return self.run_dir / "source_metadata.json"

    @property
    def raw_validation_report_path(self) -> Path:
        return self.run_dir / "raw_validation_report.json"

    @property
    def raw_profile_path(self) -> Path:
        return self.run_dir / "raw_profile.json"

    def within_root(self, path: Path) -> bool:
        """True when ``path`` resolves inside the trusted repository root."""
        try:
            Path(path).resolve().relative_to(self.repository_root)
            return True
        except ValueError:
            return False

    def placeholder_map(self) -> dict[str, str]:
        """Runtime placeholder substitutions for the task prompts."""
        return {
            "run_id": self.run_id,
            "repository_root": str(self.repository_root),
            "artifact_root": rel(self.artifact_root),
            "input_sha256": self.input_sha256,
            "openai_model_name": self.model_name or "(resolved from OPENAI_MODEL_NAME)",
            "source_metadata_path": rel(self.source_metadata_path),
            "raw_validation_report_path": rel(self.raw_validation_report_path),
            "raw_profile_path": rel(self.raw_profile_path),
            "raw_data_path": rel(self.input_path),
        }

    @classmethod
    def build(
        cls,
        input_path: str | Path,
        *,
        run_id: str | None = None,
        model_name: str | None = None,
        pin_hash: bool = True,
        require_all_months: bool = True,
    ) -> AnalystRunContext:
        """Build a context: hash the input and resolve every run-scoped path."""
        path = Path(input_path)
        sha = d.sha256_file(path)
        rid = run_id or f"analyst-{sha[:12]}"
        analyst_dir = paths.analyst_artifacts()
        return cls(
            run_id=rid,
            input_path=path,
            input_sha256=sha,
            model_name=model_name,
            repository_root=paths.PROJECT_ROOT,
            artifact_root=paths.artifact_root(),
            analyst_dir=analyst_dir,
            eda_dir=analyst_dir / "eda",
            run_dir=paths.runs_artifacts() / rid,
            pin_hash=pin_hash,
            require_all_months=require_all_months,
        )
