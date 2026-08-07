"""Structured validation issues and reports.

A validator never raises on the first problem it finds. It collects
:class:`ValidationIssue` records — each naming the offending *column*, the
*rule* that failed, the *observed* value(s), and the *expected* condition — into
a :class:`ValidationReport`. Callers decide whether to raise
:class:`ContractValidationError` (which carries the full report) or to render
the report for a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    """How serious a validation issue is.

    ``FATAL`` issues block the handoff; ``WARNING`` and ``INFO`` are recorded
    but do not, on their own, fail the dataset.
    """

    FATAL = "fatal"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    """A single, self-describing validation failure.

    Every field is meant to make the problem actionable without re-running the
    checker: which rule, which column, what was seen, and what was required.
    """

    rule: str
    message: str
    column: str | None = None
    observed: str | None = None
    expected: str | None = None
    severity: Severity = Severity.FATAL

    def render(self) -> str:
        """One-line human-readable rendering of the issue."""
        where = f" [{self.column}]" if self.column else ""
        parts = [f"{self.severity.value.upper()}{where} {self.rule}: {self.message}"]
        if self.observed is not None:
            parts.append(f"observed={self.observed}")
        if self.expected is not None:
            parts.append(f"expected={self.expected}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, str | None]:
        """JSON-serializable view of the issue."""
        return {
            "severity": self.severity.value,
            "rule": self.rule,
            "column": self.column,
            "observed": self.observed,
            "expected": self.expected,
            "message": self.message,
        }


@dataclass
class ValidationReport:
    """The accumulated result of running one or more validators."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        """Append one issue to the report."""
        self.issues.append(issue)

    @property
    def fatal_issues(self) -> list[ValidationIssue]:
        """Only the issues that block the handoff."""
        return [i for i in self.issues if i.severity is Severity.FATAL]

    @property
    def ok(self) -> bool:
        """True when no fatal issue was recorded (warnings/info are allowed)."""
        return not self.fatal_issues

    def raise_if_failed(self) -> None:
        """Raise :class:`ContractValidationError` when a fatal issue exists."""
        if not self.ok:
            raise ContractValidationError(self)

    def render(self) -> str:
        """Multi-line human-readable rendering of the whole report."""
        if not self.issues:
            return "PASS: no issues."
        status = "PASS (warnings only)" if self.ok else "FAIL"
        lines = [f"{status}: {len(self.issues)} issue(s)"]
        lines.extend(f"  - {issue.render()}" for issue in self.issues)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable view of the report."""
        return {
            "ok": self.ok,
            "fatal_count": len(self.fatal_issues),
            "issue_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
        }


class ContractValidationError(RuntimeError):
    """Raised when a dataset fails a fatal contract rule.

    The full :class:`ValidationReport` is attached so callers can inspect or
    render every issue, not just the first.
    """

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(report.render())
