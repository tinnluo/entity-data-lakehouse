"""Optional Langfuse observability wrapper.

Provides a uniform interface for emitting traces and generation spans to
Langfuse.  When ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are not
set, all calls become no-ops and a one-time ``warnings.warn`` is emitted so the
absence of credentials is visible without being fatal.

Usage::

    from entity_data_lakehouse.observability import get_langfuse

    lf = get_langfuse()
    trace = lf.trace(name="pipeline_run")
    gen = trace.generation(name="sklearn_predict", input={"rows": 5})
    gen.end(output={"rows_written": 5})
    lf.flush()

All objects returned by the no-op client implement the same interface as the
real Langfuse objects so call-sites need no conditional logic.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel so the warning fires at most once per process.
_WARNED_ONCE: bool = False

# Module-level singleton — created lazily on first get_langfuse() call.
# Caching avoids per-prediction client construction overhead and ensures
# flush() on the returned object actually drains the shared queue.
_LANGFUSE_INSTANCE: Any = None


# ---------------------------------------------------------------------------
# No-op stubs
# ---------------------------------------------------------------------------


class _NoOpGeneration:
    """No-op stand-in for a Langfuse ``Generation`` span."""

    def end(self, **kwargs: Any) -> "_NoOpGeneration":
        return self

    def score(self, **kwargs: Any) -> "_NoOpGeneration":
        return self

    def update(self, **kwargs: Any) -> "_NoOpGeneration":
        return self


class _NoOpTrace:
    """No-op stand-in for a Langfuse ``Trace``."""

    def generation(self, **kwargs: Any) -> _NoOpGeneration:
        return _NoOpGeneration()

    def span(self, **kwargs: Any) -> "_NoOpSpan":
        return _NoOpSpan()

    def score(self, **kwargs: Any) -> "_NoOpTrace":
        return self

    def update(self, **kwargs: Any) -> "_NoOpTrace":
        return self


class _NoOpSpan:
    """No-op stand-in for a Langfuse ``Span``."""

    def end(self, **kwargs: Any) -> "_NoOpSpan":
        return self

    def generation(self, **kwargs: Any) -> _NoOpGeneration:
        return _NoOpGeneration()

    def update(self, **kwargs: Any) -> "_NoOpSpan":
        return self


class _NoOpDataset:
    """No-op stand-in for a Langfuse ``Dataset``."""

    def upsert_item(self, **kwargs: Any) -> "_NoOpDatasetItem":
        return _NoOpDatasetItem()


class _NoOpDatasetItem:
    """No-op stand-in for a Langfuse ``DatasetItem``."""

    def link(self, **kwargs: Any) -> "_NoOpDatasetItem":
        return self


class _NoOpLangfuse:
    """No-op Langfuse client returned when credentials are absent."""

    def trace(self, **kwargs: Any) -> _NoOpTrace:
        return _NoOpTrace()

    def generation(self, **kwargs: Any) -> _NoOpGeneration:
        return _NoOpGeneration()

    def span(self, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def score(self, **kwargs: Any) -> "_NoOpLangfuse":
        return self

    def get_dataset(self, name: str) -> _NoOpDataset:
        return _NoOpDataset()

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_langfuse() -> Any:
    """Return a configured ``Langfuse`` client or a no-op stub.

    Returns the real ``langfuse.Langfuse`` client when both
    ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` environment variables
    are set.  Otherwise returns a ``_NoOpLangfuse`` instance and emits a
    ``UserWarning`` on the first call to make the silent degradation visible.

    The same instance is returned on every call within a process so that a
    ``lf.flush()`` at batch boundaries reliably drains all queued events.
    Re-reading credentials on every call is intentionally avoided: if
    credentials change mid-process a new session should be started.

    The ``langfuse`` package is imported lazily so this module can be imported
    without ``langfuse`` installed.
    """
    import os

    global _WARNED_ONCE, _LANGFUSE_INSTANCE

    if _LANGFUSE_INSTANCE is not None:
        return _LANGFUSE_INSTANCE

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()

    if not public_key or not secret_key:
        if not _WARNED_ONCE:
            warnings.warn(
                "Langfuse credentials not set (LANGFUSE_PUBLIC_KEY / "
                "LANGFUSE_SECRET_KEY); tracing disabled.",
                UserWarning,
                stacklevel=2,
            )
            _WARNED_ONCE = True
        _LANGFUSE_INSTANCE = _NoOpLangfuse()
        return _LANGFUSE_INSTANCE

    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]

        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        _LANGFUSE_INSTANCE = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        return _LANGFUSE_INSTANCE
    except ImportError:
        logger.warning(
            "langfuse package not installed; tracing disabled. "
            "Install with: pip install 'entity-data-lakehouse[observability]'"
        )
        _LANGFUSE_INSTANCE = _NoOpLangfuse()
        return _LANGFUSE_INSTANCE
