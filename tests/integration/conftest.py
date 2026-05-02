"""Integration test configuration.

Enforces environment isolation for all integration tests so that a caller's
shell environment cannot accidentally trigger ClickHouse mutations or an
unexpected ML backend variant.

Overrides applied for every test in this directory:
- ``USE_CLICKHOUSE`` → ``"false"``   (prevents real ClickHouse connections)
- ``ML_BACKEND``     → ``"default"`` (prevents LoRA/external model paths)
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force safe env defaults for every integration test."""
    monkeypatch.setenv("USE_CLICKHOUSE", "false")
    monkeypatch.setenv("ML_BACKEND", "default")
