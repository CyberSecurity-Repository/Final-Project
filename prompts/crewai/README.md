# CrewAI Runtime Agent Prompt Index

These files are the prompts used by the six agents at application runtime. They are not Claude Code implementation prompts.

## Execution order

| Order | Crew | Agent prompt | Receives context from | Main result |
|---:|---|---|---|---|
| 1 | Analyst | `analyst/01_source_quality_analyst.md` | Flow inputs + raw validation artifacts | Source/quality readiness decision |
| 2 | Analyst | `analyst/02_data_engineer.md` | Agent 1 | Validated cleaned-data handoff |
| 3 | Analyst | `analyst/03_eda_business_analyst.md` | Agents 1–2 | Analyst crew handoff + required Analyst reports |
| 4 | Scientist | `scientist/01_contract_feature_engineer.md` | Flow-validated Analyst handoff | Leakage-audited feature handoff |
| 5 | Scientist | `scientist/02_model_trainer.md` | Agent 4 | Validation-only candidate comparison |
| 6 | Scientist | `scientist/03_evaluation_governance_reviewer.md` | Agents 4–5 | Locked model, test evaluation, reports, Scientist handoff |

Every task also receives `00_shared_runtime_rules.md`. Prepend those rules to the task description or inject them through the project's system-prompt boundary.

## CrewAI wiring requirements

- Use `Process.sequential` for both crews.
- Assign every task to its named agent.
- Set explicit task `context` references; do not rely on accidental full-context carryover.
- Pass the final Analyst handoff into the Scientist crew through Flow inputs only after the deterministic Analyst gate passes.
- Use Pydantic or JSON structured task outputs. Validate each output before the next task receives it.
- Keep `allow_delegation: false`, `allow_code_execution: false`, `memory: false`, and bounded iterations/retries.
- Resolve the LLM from `OPENAI_MODEL_NAME`; never hard-code a model or API key in these files.
- Give agents only the tools listed in their prompt. Tool names may be adapted to the implementation, but permissions and behavior must remain equivalent.
- Keep raw CSV rows and model binaries out of LLM context. Agents receive compact, machine-produced summaries and references.

## Runtime input contract

Supply these common placeholders to both crews:

- `{run_id}` — unique run identifier.
- `{repository_root}` — trusted repository root.
- `{artifact_root}` — run-scoped artifact root.
- `{input_sha256}` — SHA-256 of the raw input.
- `{openai_model_name}` — model resolved from environment configuration.

The Analyst crew also receives:

- `{source_metadata_path}`
- `{raw_validation_report_path}`
- `{raw_profile_path}`
- `{raw_data_path}` — a reference for deterministic tools; do not insert file contents into the prompt.

The Scientist crew also receives:

- `{analyst_handoff_path}`
- `{clean_data_path}`
- `{dataset_contract_path}`
- `{experiment_config_path}`

If any placeholder is missing, malformed, outside the trusted repository root, or inconsistent with the current run, return `BLOCKED`. Do not guess a path or value.

## Output handling

The agent's final response is a structured control record, not a prose report. Deterministic writer tools create the required `.md`, `.html`, `.csv`, `.json`, and `.joblib` artifacts. This separation prevents an agent response from silently becoming an unvalidated repository artifact.

Recommended implementation schemas:

- `SourceQualityReview`
- `DataEngineeringHandoff`
- `AnalystCrewHandoff`
- `FeatureEngineeringHandoff`
- `TrainingRunHandoff`
- `ScientistCrewHandoff`

Exact schema fields are specified in each prompt file.

## Non-negotiable gate

An LLM may interpret validated results, but it never decides whether a schema, artifact, hash, feature, model, or metric is technically valid. Only deterministic validators set those pass/fail facts. A failed validator ends the task with `FAIL`; a failed upstream task makes the next task `BLOCKED`.

