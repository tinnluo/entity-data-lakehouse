"""Unit test configuration.

Enforces environment isolation for all unit tests so that a caller's shell
environment cannot accidentally trigger ClickHouse connections or an
unexpected ML backend variant.

Overrides applied for every test in this directory:
- ``USE_CLICKHOUSE`` → ``"false"``   (prevents real ClickHouse connections)
- ``ML_BACKEND``     → ``"default"`` (prevents LoRA/external model paths)
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force safe env defaults for every unit test."""
    monkeypatch.setenv("USE_CLICKHOUSE", "false")
    monkeypatch.setenv("ML_BACKEND", "default")
