"""OpenAI LLM factory for the crews.

The model is resolved from ``OPENAI_MODEL_NAME`` at run time — never hard-coded —
and credentials are validated only here, at the start of an LLM-backed run, via
:func:`retail_clickstream_ai.config.require_openai_config`. Importing this module
performs no network call and needs no key; CrewAI is imported lazily inside the
factory so the rest of the package stays light.
"""

from __future__ import annotations

from typing import Any

from retail_clickstream_ai.config import Settings, require_openai_config


def make_llm(settings: Settings | None = None) -> Any:
    """Build a CrewAI ``LLM`` bound to the configured OpenAI model.

    Raises :class:`retail_clickstream_ai.config.ConfigurationError` if the
    required OpenAI settings are absent — call this only when starting a real run.
    """
    resolved = require_openai_config(settings)
    from crewai import LLM

    model_name = resolved.openai_model_name
    assert model_name is not None  # guaranteed by require_openai_config
    return LLM(model=model_name, api_key=resolved.openai_api_key)
