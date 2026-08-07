"""Application configuration.

Settings are read from environment variables, optionally seeded from a local
``.env`` file. Two hard rules hold:

1. Importing this module (or the package) never requires a secret and never
   performs network access. ``load_dotenv`` only reads a local file if present.
2. OpenAI credentials are validated lazily, only when an LLM-backed run begins,
   via :func:`require_openai_config`. Secrets are never logged or echoed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Seed os.environ from a local .env if one exists. This is a no-op when the file
# is absent and never overrides values already set in the real environment.
load_dotenv(override=False)


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Resolved application settings.

    None of these fields are required to import the package or run the offline
    test suite. ``openai_api_key`` and ``openai_model_name`` are only needed
    when an LLM-backed run starts; enforce them with :func:`require_openai_config`.
    """

    openai_api_key: str | None
    openai_model_name: str | None
    log_level: str
    artifact_root: str

    @property
    def has_openai_config(self) -> bool:
        """True only when both OpenAI settings are present and non-empty."""
        return bool(self.openai_api_key) and bool(self.openai_model_name)


def _clean(value: str | None) -> str | None:
    """Return a stripped value, or ``None`` for missing/empty/whitespace."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load_settings() -> Settings:
    """Read the current environment into a :class:`Settings` snapshot."""
    return Settings(
        openai_api_key=_clean(os.environ.get("OPENAI_API_KEY")),
        openai_model_name=_clean(os.environ.get("OPENAI_MODEL_NAME")),
        log_level=(_clean(os.environ.get("LOG_LEVEL")) or "INFO").upper(),
        artifact_root=_clean(os.environ.get("ARTIFACT_ROOT")) or "artifacts",
    )


def require_openai_config(settings: Settings | None = None) -> Settings:
    """Validate that OpenAI settings are present; raise otherwise.

    Call this at the start of an LLM-backed run only — never at import time.
    The error message lists which variables are missing but never their values.
    """
    resolved = settings or load_settings()
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", resolved.openai_api_key),
            ("OPENAI_MODEL_NAME", resolved.openai_model_name),
        )
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Missing required OpenAI configuration: "
            + ", ".join(missing)
            + ". Set these in your environment or local .env before starting an "
            "LLM-backed run (see .env.example)."
        )
    return resolved
