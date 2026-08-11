"""The CrewAI Scientist crew — three sequential agents.

Wires the inline runtime prompts (``scientist/specs.py`` plus the shared rules in
``crews/prompts.py``) to three agents — Contract & Feature Engineer, Model
Trainer, and Evaluation & Governance Reviewer — each restricted to the tools its
prompt allows, each writing a validated Pydantic handoff to disk through its tools,
running as a ``Process.sequential`` crew with explicit task context references. The
heavy lifting is done by the deterministic tools; the agents interpret and assemble
the handoffs, and the one-time held-out test evaluation lives entirely inside the
governance tool.

Tasks intentionally do **not** set ``output_pydantic``: the ``write_*`` tools already
persist a fully validated handoff (via ``stamp_and_write``), so coercing each agent's
free-text final answer back into the strict model is redundant and makes weak models
crash on machine-only fields (content hashes, sizes) they cannot author. The
deterministic validators/gates remain the sole authority on pass/fail.

Building a crew needs no API key (so tests can inspect it offline); credentials
are required only by :func:`run_scientist_crew`, which starts a real LLM run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from retail_clickstream_ai.crews import prompts as prompt_loader
from retail_clickstream_ai.crews.context import ScientistRunContext
from retail_clickstream_ai.crews.scientist.specs import SCIENTIST_SPECS
from retail_clickstream_ai.crews.scientist.tools import build_scientist_tools

# Fixed per-role settings taken from the runtime prompt specs.
_ROLE_ORDER = ("contract_feature_engineer", "model_trainer", "evaluation_governance")
_MAX_ITER = {
    "contract_feature_engineer": 7,
    "model_trainer": 7,
    "evaluation_governance": 9,
}


@dataclass
class ScientistCrewBundle:
    """A built crew plus the agents/tasks, so callers/tests can inspect wiring."""

    crew: Crew
    agents: dict[str, Agent]
    tasks: dict[str, Task]
    context: ScientistRunContext


def _make_agent(
    role_key: str, spec: prompt_loader.AgentSpec, llm: Any, tools: list[BaseTool], verbose: bool
) -> Agent:
    return Agent(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory,
        tools=tools,
        llm=llm,
        allow_delegation=False,
        allow_code_execution=False,
        memory=False,
        max_iter=_MAX_ITER[role_key],
        max_retry_limit=1,
        respect_context_window=True,
        verbose=verbose,
    )


def build_scientist_crew(
    context: ScientistRunContext,
    *,
    llm: Any = None,
    tools: dict[str, list[BaseTool]] | None = None,
    verbose: bool = False,
) -> ScientistCrewBundle:
    """Assemble the three-agent sequential Scientist crew (no API key required)."""
    shared_rules = prompt_loader.SHARED_RUNTIME_RULES
    mapping = context.placeholder_map()
    role_tools = tools or build_scientist_tools(context)

    agents: dict[str, Agent] = {}
    tasks: dict[str, Task] = {}
    ordered_tasks: list[Task] = []

    for role_key in _ROLE_ORDER:
        spec = SCIENTIST_SPECS[role_key]
        agent = _make_agent(role_key, spec, llm, role_tools[role_key], verbose)
        task = Task(
            description=prompt_loader.build_task_description(spec, shared_rules, mapping),
            expected_output=prompt_loader.fill_placeholders(spec.expected_output, mapping),
            agent=agent,
            tools=role_tools[role_key],
            context=list(ordered_tasks),  # explicit upstream context references
            name=f"scientist_{role_key}",
        )
        agents[role_key] = agent
        tasks[role_key] = task
        ordered_tasks.append(task)

    crew = Crew(
        agents=[agents[r] for r in _ROLE_ORDER],
        tasks=ordered_tasks,
        process=Process.sequential,
        memory=False,
        verbose=verbose,
    )
    return ScientistCrewBundle(crew=crew, agents=agents, tasks=tasks, context=context)


def run_scientist_crew(
    *,
    run_id: str | None = None,
    verbose: bool = False,
) -> tuple[Any, ScientistRunContext]:
    """Prepare a run, build the crew, and kick off a real LLM-backed Scientist run.

    Requires OpenAI configuration (validated inside :func:`make_llm`) and the
    validated Analyst artifacts on disk. The immutable experiment configuration is
    written first, so the Model Trainer's tools have something to read.
    """
    from retail_clickstream_ai.config import load_settings
    from retail_clickstream_ai.crews.llm import make_llm
    from retail_clickstream_ai.pipeline import modeling as model_pipeline

    settings = load_settings()
    llm = make_llm(settings)  # validates credentials; raises if missing

    context = ScientistRunContext.build(run_id=run_id, model_name=settings.openai_model_name)
    context.scientist_dir.mkdir(parents=True, exist_ok=True)
    context.run_dir.mkdir(parents=True, exist_ok=True)
    model_pipeline.write_experiment_config(context.experiment_config_path)

    bundle = build_scientist_crew(context, llm=llm, verbose=verbose)
    result = bundle.crew.kickoff(inputs=context.placeholder_map())
    return result, context


if __name__ == "__main__":  # pragma: no cover
    _result, _ctx = run_scientist_crew()
    print(f"Scientist crew finished for run {_ctx.run_id}")
