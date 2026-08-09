"""The CrewAI Analyst crew — three sequential agents.

Wires the inline runtime prompts (``analyst/specs.py`` plus the shared rules in
``crews/prompts.py``) to three agents — Source & Quality Analyst, Data Engineer,
and EDA & Business Analyst — each restricted to the tools its prompt allows, each
returning a validated Pydantic structured output, running as a
``Process.sequential`` crew with explicit task context references. The heavy
lifting is done by the deterministic tools; the agents interpret and assemble the
handoffs.

Building a crew needs no API key (so tests can inspect it offline); credentials
are required only by :func:`run_analyst_crew`, which starts a real LLM run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from retail_clickstream_ai.crews import prompts as prompt_loader
from retail_clickstream_ai.crews.analyst import models as m
from retail_clickstream_ai.crews.analyst.specs import ANALYST_SPECS
from retail_clickstream_ai.crews.analyst.tools import build_analyst_tools
from retail_clickstream_ai.crews.context import AnalystRunContext

# Fixed per-role settings taken from the runtime prompt specs.
_ROLE_ORDER = ("source_quality", "data_engineer", "eda_business")
_MAX_ITER = {"source_quality": 6, "data_engineer": 6, "eda_business": 8}
_OUTPUT_MODEL: dict[str, type] = {
    "source_quality": m.SourceQualityReview,
    "data_engineer": m.DataEngineeringHandoff,
    "eda_business": m.AnalystCrewHandoff,
}


@dataclass
class AnalystCrewBundle:
    """A built crew plus the agents/tasks, so callers/tests can inspect wiring."""

    crew: Crew
    agents: dict[str, Agent]
    tasks: dict[str, Task]
    context: AnalystRunContext


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


def build_analyst_crew(
    context: AnalystRunContext,
    *,
    llm: Any = None,
    tools: dict[str, list[BaseTool]] | None = None,
    verbose: bool = False,
) -> AnalystCrewBundle:
    """Assemble the three-agent sequential Analyst crew (no API key required)."""
    shared_rules = prompt_loader.SHARED_RUNTIME_RULES
    mapping = context.placeholder_map()
    role_tools = tools or build_analyst_tools(context)

    agents: dict[str, Agent] = {}
    tasks: dict[str, Task] = {}
    ordered_tasks: list[Task] = []

    for role_key in _ROLE_ORDER:
        spec = ANALYST_SPECS[role_key]
        agent = _make_agent(role_key, spec, llm, role_tools[role_key], verbose)
        task = Task(
            description=prompt_loader.build_task_description(spec, shared_rules, mapping),
            expected_output=prompt_loader.fill_placeholders(spec.expected_output, mapping),
            agent=agent,
            tools=role_tools[role_key],
            output_pydantic=_OUTPUT_MODEL[role_key],
            context=list(ordered_tasks),  # explicit upstream context references
            name=f"analyst_{role_key}",
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
    return AnalystCrewBundle(crew=crew, agents=agents, tasks=tasks, context=context)


def run_analyst_crew(
    input_path: str,
    *,
    run_id: str | None = None,
    pin_hash: bool = True,
    require_all_months: bool = True,
    verbose: bool = False,
) -> tuple[Any, AnalystRunContext]:
    """Prepare a run, build the crew, and kick off a real LLM-backed Analyst run.

    Requires OpenAI configuration (validated inside :func:`make_llm`). The three
    agent-input files are written first, so the Source & Quality agent's tools
    have something to read.
    """
    from retail_clickstream_ai.config import load_settings
    from retail_clickstream_ai.crews.llm import make_llm
    from retail_clickstream_ai.pipeline.analyst_pipeline import prepare_run
    from retail_clickstream_ai.validation.contract import DatasetContract

    settings = load_settings()
    llm = make_llm(settings)  # validates credentials; raises if missing

    context = AnalystRunContext.build(
        input_path,
        run_id=run_id,
        model_name=settings.openai_model_name,
        pin_hash=pin_hash,
        require_all_months=require_all_months,
    )
    context.run_dir.mkdir(parents=True, exist_ok=True)
    prepare_run(
        input_path,
        DatasetContract.build(),
        context.run_dir,
        require_all_months=require_all_months,
    )

    bundle = build_analyst_crew(context, llm=llm, verbose=verbose)
    result = bundle.crew.kickoff(inputs=context.placeholder_map())
    return result, context


if __name__ == "__main__":  # pragma: no cover
    import sys

    _result, _ctx = run_analyst_crew(
        sys.argv[1] if len(sys.argv) > 1 else "data/raw/e-shop clothing 2008.csv"
    )
    print(f"Analyst crew finished for run {_ctx.run_id}")
