"""Stage 1 smoke tests.

Prove that the package imports offline and without secrets, that OpenAI
configuration is validated only lazily, and that the canonical path module
honors its override. These tests must not require an API key or network access.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _reimport_package() -> None:
    """Drop the package (and submodules) from the import cache and re-import it."""
    for name in list(sys.modules):
        if name == "retail_clickstream_ai" or name.startswith("retail_clickstream_ai."):
            del sys.modules[name]
    importlib.import_module("retail_clickstream_ai")
    importlib.import_module("retail_clickstream_ai.config")
    importlib.import_module("retail_clickstream_ai.paths")


def test_package_exposes_version() -> None:
    pkg = importlib.import_module("retail_clickstream_ai")
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__


def test_import_makes_no_network_call(no_network: None) -> None:
    """Re-importing the package with the network blocked must succeed."""
    _reimport_package()
    pkg = importlib.import_module("retail_clickstream_ai")
    assert pkg.__version__


def test_package_imports_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
    _reimport_package()
    pkg = importlib.import_module("retail_clickstream_ai")
    assert pkg.__version__


def test_require_openai_config_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
    from retail_clickstream_ai.config import (
        ConfigurationError,
        load_settings,
        require_openai_config,
    )

    settings = load_settings()
    assert settings.openai_api_key is None
    assert settings.has_openai_config is False
    with pytest.raises(ConfigurationError) as excinfo:
        require_openai_config(settings)
    # The error names the missing variables but never leaks a value.
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert "OPENAI_MODEL_NAME" in str(excinfo.value)


def test_require_openai_config_passes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-model")
    from retail_clickstream_ai.config import load_settings, require_openai_config

    resolved = require_openai_config(load_settings())
    assert resolved.has_openai_config is True
    assert resolved.openai_model_name == "test-model"


def test_artifact_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    from retail_clickstream_ai import paths

    assert paths.artifact_root() == tmp_path
    assert paths.analyst_artifacts() == tmp_path / "analyst"
    assert paths.scientist_artifacts() == tmp_path / "scientist"
    assert paths.runs_artifacts() == tmp_path / "runs"
