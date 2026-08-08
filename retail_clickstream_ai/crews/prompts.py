"""Loader for the versioned CrewAI runtime prompt files.

The six agents' role/goal/backstory, task boundaries, and expected structured
output are authored in ``prompts/crewai/`` and are the source of truth — the
crew must load *those* exact prompts, not paraphrased one-liners. This module
parses an agent prompt file into its parts and exposes the shared runtime rules
so every task can be built as ``shared rules + role-specific task prompt``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from retail_clickstream_ai import paths

SHARED_RULES_FILE = paths.PROMPTS_CREWAI / "00_shared_runtime_rules.md"


@dataclass(frozen=True)
class AgentPrompt:
    """The parsed parts of one runtime agent prompt file."""

    role: str
    goal: str
    backstory: str
    task_prompt: str
    expected_output: str
    raw: str


def load_shared_rules() -> str:
    """Return the shared runtime rules text applied to every task."""
    return SHARED_RULES_FILE.read_text(encoding="utf-8").strip()


def _sections(text: str) -> dict[str, str]:
    """Split markdown into ``## `` top-level sections (heading -> body)."""
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = line[3:].strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    return sections


def _bold_field(text: str, label: str) -> str:
    """Extract the inline value after a ``**Label:**`` marker."""
    match = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", text)
    if not match:
        raise ValueError(f"prompt is missing the '{label}' field")
    return match.group(1).strip()


def load_agent_prompt(path: str | Path) -> AgentPrompt:
    """Parse a runtime agent prompt file into an :class:`AgentPrompt`."""
    raw = Path(path).read_text(encoding="utf-8")
    sections = _sections(raw)
    definition = sections.get("Agent definition", "")
    task_prompt = sections.get("Task prompt", "")
    expected = sections.get("Expected output", "")
    if not definition or not task_prompt:
        raise ValueError(f"prompt file missing required sections: {path}")
    return AgentPrompt(
        role=_bold_field(definition, "Role"),
        goal=_bold_field(definition, "Goal"),
        backstory=_bold_field(definition, "Backstory"),
        task_prompt=task_prompt,
        expected_output=expected,
        raw=raw,
    )


def fill_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Replace ``{key}`` placeholders from ``mapping``; leave unknown braces intact."""
    for key, value in mapping.items():
        text = text.replace("{" + key + "}", value)
    return text


def build_task_description(prompt: AgentPrompt, shared_rules: str, mapping: dict[str, str]) -> str:
    """Compose a task description: shared rules + the role's filled task prompt."""
    filled = fill_placeholders(prompt.task_prompt, mapping)
    return f"{shared_rules}\n\n---\n\n{filled}"


# Repository-relative locations of the three Analyst runtime prompt files.
ANALYST_PROMPT_FILES: tuple[Path, ...] = (
    paths.PROMPTS_CREWAI / "analyst" / "01_source_quality_analyst.md",
    paths.PROMPTS_CREWAI / "analyst" / "02_data_engineer.md",
    paths.PROMPTS_CREWAI / "analyst" / "03_eda_business_analyst.md",
)
