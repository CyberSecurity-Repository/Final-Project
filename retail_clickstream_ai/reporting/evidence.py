"""Traceable numeric claims for the Analyst reports.

The reports (``insights.md`` and ``eda_report.html``) must never contain a
number that a reader cannot trace back to a computed metric. This module makes
that structurally impossible: prose is assembled through an :class:`Evidence`
recorder whose :meth:`Evidence.number` method can only emit a value it just read
out of the metrics dict, and it records a :class:`Claim` linking the rendered
text to its metric path. :func:`validate_claims` re-resolves every claim against
the metrics and raises :class:`UnsupportedClaimError` on any mismatch — the
deterministic equivalent of the runtime renderer "rejecting an unsupported
claim". A test injects a fabricated claim to prove the rejection fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class UnsupportedClaimError(ValueError):
    """Raised when a report claim does not match its cited metric value."""


@dataclass(frozen=True)
class Claim:
    """A rendered number linked to the metric it was read from.

    ``ref`` is an evidence reference of the form ``path#dotted.key``; ``stated``
    is the raw value read from the metrics (so a claim built through
    :class:`Evidence` is correct by construction); ``text`` is exactly how the
    value appears in the prose (used to scan the rendered document).
    """

    ref: str
    stated: Any
    text: str


def resolve_metric(metrics: dict[str, Any], ref: str) -> Any:
    """Resolve a dotted metric path, ignoring any ``path#`` file prefix."""
    dotted = ref.split("#", 1)[1] if "#" in ref else ref
    node: Any = metrics
    for segment in dotted.split("."):
        if not isinstance(node, dict) or segment not in node:
            raise UnsupportedClaimError(f"metric path not found: {ref}")
        node = node[segment]
    return node


def validate_claims(
    claims: list[Claim], metrics: dict[str, Any], *, tolerance: float = 1e-9
) -> None:
    """Raise :class:`UnsupportedClaimError` if any claim disagrees with metrics."""
    for claim in claims:
        resolved = resolve_metric(metrics, claim.ref)
        if isinstance(resolved, bool) or isinstance(claim.stated, bool):
            matches = resolved == claim.stated
        elif isinstance(resolved, int | float) and isinstance(claim.stated, int | float):
            matches = abs(float(resolved) - float(claim.stated)) <= tolerance
        else:
            matches = str(resolved) == str(claim.stated)
        if not matches:
            raise UnsupportedClaimError(
                f"claim '{claim.text}' cites {claim.ref} but the metric is {resolved!r}"
            )


@dataclass
class Evidence:
    """A recorder that lets prose emit only metric-backed numbers.

    ``metrics_ref`` is the repository-relative path of the metrics file, used to
    build full evidence references. Every :meth:`number` / :meth:`value` call
    reads from the metrics and records a :class:`Claim`, so the rendered text can
    never contain an unsupported figure.
    """

    metrics: dict[str, Any]
    metrics_ref: str
    claims: list[Claim] = field(default_factory=list)

    def _record(self, dotted: str, stated: Any, text: str) -> str:
        self.claims.append(Claim(f"{self.metrics_ref}#{dotted}", stated, text))
        return text

    def number(self, dotted: str, fmt: str = "{:,}") -> str:
        """Read a numeric metric and return it formatted, recording the claim."""
        value = resolve_metric(self.metrics, dotted)
        return self._record(dotted, value, fmt.format(value))

    def value(self, dotted: str) -> str:
        """Read a metric (e.g. a label) and return it as text, recording the claim."""
        value = resolve_metric(self.metrics, dotted)
        return self._record(dotted, value, str(value))

    def ref(self, dotted: str) -> str:
        """Return the full evidence reference for a metric path (no claim)."""
        return f"{self.metrics_ref}#{dotted}"

    def validate(self) -> None:
        """Confirm every recorded claim still matches its metric."""
        validate_claims(self.claims, self.metrics)
