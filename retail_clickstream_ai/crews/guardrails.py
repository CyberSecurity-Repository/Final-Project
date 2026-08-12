"""Task-level completion guardrails shared by the Analyst and Scientist crews.

CrewAI's native tool-calling agents sometimes emit a Final Answer after only
part of their assigned tool sequence: nothing in the framework's native
tool-calling prompt tells the model it must exhaust its tools before
stopping — the only steering text is a generic, per-tool-call nudge ("Analyze
the tool result. If requirements are met, provide the Final Answer. Otherwise,
call the next tool.") with "requirements" left entirely to the model's own
inference from the task description.

:func:`build_handoff_guardrail` is the backstop for that gap: it checks
whether a role's terminal ``write_*`` tool actually persisted its handoff
record under the current run before CrewAI accepts the task's output. If not,
CrewAI automatically re-runs the agent (bounded by ``Task.guardrail_max_retries``)
with the returned message appended to its original instructions.

Scope: this checks only *whether the terminal tool ran*, never *what it
found*. A handoff stamped ``BLOCKED`` still means the agent completed its
assigned procedure and must be allowed through — the ``write_*`` tool itself
remains the sole authority on PASS/BLOCKED (see ``io_helpers.stamp_and_write``).
A guardrail retry is therefore unrelated to retrying a deterministic
contract/schema failure: it fires only because no verdict was ever recorded,
never because a validator returned one it didn't like.

Deliberately does NOT use ``from __future__ import annotations``: CrewAI's
``Task.guardrail`` field validator inspects the closure's live signature
(``inspect.signature``) and rejects a stringified return annotation — it
requires the real ``tuple[bool, Any]`` type object, so ``TaskOutput`` and the
return type are imported and annotated concretely here.
"""

import json
from collections.abc import Callable
from typing import Any

from crewai import TaskOutput
from crewai.lite_agent_output import LiteAgentOutput

from retail_clickstream_ai.crews.context import AnalystRunContext, ScientistRunContext

HandoffRunContext = AnalystRunContext | ScientistRunContext


def build_handoff_guardrail(
    context: HandoffRunContext,
    *,
    handoff_filename: str,
    tool_name: str,
) -> Callable[[TaskOutput | LiteAgentOutput], tuple[bool, Any]]:
    """Return a ``Task.guardrail`` that fails until ``tool_name`` writes ``handoff_filename``.

    Satisfies CrewAI's guardrail signature contract
    (``crewai.task.Task.validate_guardrail_function``): exactly one positional
    parameter, return type ``tuple[bool, Any]``. Never inspects the agent's own
    output text — only the trusted on-disk artifact under ``context.run_dir`` —
    so a well-written summary cannot substitute for an actual tool call, and a
    plain/empty summary cannot fail it either.
    """
    handoff_path = context.run_dir / handoff_filename
    run_id = context.run_id

    def _handoff_guardrail(output: TaskOutput | LiteAgentOutput) -> tuple[bool, Any]:
        if not handoff_path.exists() or handoff_path.stat().st_size == 0:
            return False, (
                f"`{handoff_filename}` has not been written yet, so `{tool_name}` has not "
                f"completed in this run. Continue your required procedure and call "
                f"`{tool_name}` — it is the last required tool call. Do not submit a Final "
                f"Answer before it returns."
            )
        try:
            payload = json.loads(handoff_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, (
                f"`{handoff_filename}` exists but is not valid JSON. Call `{tool_name}` "
                f"again so it can rebuild the record from the current on-disk evidence, "
                f"then submit your Final Answer."
            )
        if payload.get("run_id") != run_id:
            return False, (
                f"`{handoff_filename}` exists but belongs to a different run "
                f"(found run_id={payload.get('run_id')!r}, expected {run_id!r}). "
                f"Call `{tool_name}` to write this run's own handoff before submitting a "
                f"Final Answer."
            )
        return True, output

    return _handoff_guardrail


__all__ = ["build_handoff_guardrail", "HandoffRunContext"]
