"""Shared pytest fixtures.

Includes a network guard so the offline test suite fails loudly if any code
attempts a real socket connection. Combined with fresh re-imports in the smoke
tests, this proves the import and configuration paths stay fully offline.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


class NoNetworkError(RuntimeError):
    """Raised when code attempts network access during an offline test."""


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block outbound socket connections for the duration of a test."""

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise NoNetworkError("Network access is not allowed in offline tests.")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    yield
