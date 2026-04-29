"""Unit tests for the observability module.

Verifies:
  - _NoOpLangfuse stub is returned and a UserWarning is emitted when env vars absent.
  - Warning fires only once per process (singleton sentinel).
  - Real Langfuse client is constructed when both keys are present.
  - _NoOpLangfuse methods are callable and return without error.
  - _NoOpTrace / _NoOpGeneration / _NoOpSpan chain correctly.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_warned_flag() -> None:
    """Reset the module-level _WARNED_ONCE sentinel and singleton between tests."""
    import entity_data_lakehouse.observability as obs_mod

    obs_mod._WARNED_ONCE = False
    obs_mod._LANGFUSE_INSTANCE = None


# ---------------------------------------------------------------------------
# No-op path (credentials absent)
# ---------------------------------------------------------------------------


def test_get_langfuse_returns_noop_when_creds_absent(monkeypatch) -> None:
    """get_langfuse() must return a _NoOpLangfuse when keys are unset."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    _clear_warned_flag()

    from entity_data_lakehouse.observability import _NoOpLangfuse, get_langfuse

    client = get_langfuse()
    assert isinstance(client, _NoOpLangfuse)


def test_get_langfuse_emits_warning_when_creds_absent(monkeypatch) -> None:
    """A UserWarning must be emitted on the first no-creds call."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    _clear_warned_flag()

    from entity_data_lakehouse.observability import get_langfuse

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_langfuse()

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "Langfuse credentials not set" in str(user_warnings[0].message)


def test_get_langfuse_warning_fires_only_once(monkeypatch) -> None:
    """The UserWarning must not repeat on subsequent calls."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    _clear_warned_flag()

    from entity_data_lakehouse.observability import get_langfuse

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_langfuse()
        get_langfuse()
        get_langfuse()

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1, "Warning should fire exactly once"


# ---------------------------------------------------------------------------
# Real Langfuse path (credentials present, package mocked)
# ---------------------------------------------------------------------------


def test_get_langfuse_constructs_real_client_when_creds_present(monkeypatch) -> None:
    """get_langfuse() must instantiate Langfuse with the provided credentials."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-123")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-456")
    monkeypatch.setenv("LANGFUSE_HOST", "https://example.langfuse.com")
    _clear_warned_flag()

    fake_client = MagicMock()
    FakeLangfuse = MagicMock(return_value=fake_client)

    import sys

    fake_langfuse_mod = MagicMock()
    fake_langfuse_mod.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse_mod)

    from entity_data_lakehouse.observability import get_langfuse

    result = get_langfuse()

    FakeLangfuse.assert_called_once_with(
        public_key="pk-test-123",
        secret_key="sk-test-456",
        host="https://example.langfuse.com",
    )
    assert result is fake_client


def test_get_langfuse_falls_back_to_noop_when_package_missing(monkeypatch) -> None:
    """get_langfuse() must return _NoOpLangfuse if langfuse package is absent."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-123")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-456")
    _clear_warned_flag()

    import sys

    # Remove langfuse from modules so the import inside get_langfuse raises ImportError.
    monkeypatch.setitem(sys.modules, "langfuse", None)  # None causes ImportError

    from entity_data_lakehouse.observability import _NoOpLangfuse, get_langfuse

    result = get_langfuse()
    assert isinstance(result, _NoOpLangfuse)


# ---------------------------------------------------------------------------
# No-op object interface
# ---------------------------------------------------------------------------


def test_noop_langfuse_interface() -> None:
    """All _NoOpLangfuse methods must be callable without error."""
    from entity_data_lakehouse.observability import _NoOpLangfuse

    lf = _NoOpLangfuse()

    trace = lf.trace(name="test")
    assert trace is not None

    gen = lf.generation(name="gen")
    gen.end(output={"result": 42})
    gen.score(name="acc", value=0.9)
    gen.update(metadata={"x": 1})

    span = lf.span(name="span")
    span.end()
    span.update(metadata={})

    inner_gen = trace.generation(name="inner_gen")
    inner_gen.end()

    inner_span = trace.span(name="inner_span")
    inner_span.end()
    inner_gen2 = inner_span.generation(name="gen_in_span")
    inner_gen2.end()

    trace.score(name="metric", value=1.0)
    trace.update(metadata={})

    dataset = lf.get_dataset("my-dataset")
    item = dataset.upsert_item(input={"x": 1}, expected_output="planning")
    item.link(run_name="test-run")

    lf.flush()
    lf.shutdown()


def test_noop_trace_generation_chain() -> None:
    """_NoOpTrace.generation() must return a _NoOpGeneration."""
    from entity_data_lakehouse.observability import _NoOpGeneration, _NoOpTrace

    trace = _NoOpTrace()
    gen = trace.generation(name="x")
    assert isinstance(gen, _NoOpGeneration)
    # Chaining must not raise.
    gen.end().score(name="s", value=0.5).update(metadata={})
