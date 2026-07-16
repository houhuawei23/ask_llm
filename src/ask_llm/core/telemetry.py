"""Structured observability primitives for Ask LLM.

Provides error classification, request/task log context, and helpers for
injecting correlation IDs into Loguru logs without leaking CLI details into
core modules.

Error classification consumes the single keyword rule table in
``ask_llm.core.error_keywords`` (P4.8); ``ErrorCategory`` is re-exported here
for backward compatibility.
"""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from ask_llm.core.error_keywords import ErrorCategory, classify_error_message

__all__ = [
    "ErrorCategory",
    "LogContext",
    "bind_context",
    "classify_error",
    "should_fallback_for_error",
]


class LogContext(BaseModel):
    """Immutable correlation context for a single request or task attempt.

    Fields are intentionally flat so they serialize cleanly to JSON log records
    and execution reports.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: int | None = None
    provider: str | None = None
    model: str | None = None
    attempt: int = Field(default=1, ge=1)
    phase: str | None = None

    def with_attempt(self, attempt: int) -> LogContext:
        """Return a copy with the attempt counter updated."""
        return self.model_copy(update={"attempt": attempt})

    def with_provider_model(self, provider: str, model: str) -> LogContext:
        """Return a copy with provider/model filled in."""
        return self.model_copy(update={"provider": provider, "model": model})

    def to_extra(self) -> dict[str, Any]:
        """Return fields suitable for ``logger.bind(**...)``."""
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "provider": self.provider,
            "model": self.model,
            "attempt": self.attempt,
            "phase": self.phase,
        }


def classify_error(error_message: str | None) -> ErrorCategory:
    """Classify a raw error message into a stable :class:`ErrorCategory`.

    Delegates to the single keyword rule table in
    ``ask_llm.core.error_keywords`` (first matching rule in table order wins).
    The heuristic is intentionally conservative: we only categorize errors that
    are clearly identifiable from common provider/HTTP signatures. Everything
    else falls back to ``UNKNOWN`` so users are not misled.
    """
    return classify_error_message(error_message)


def should_fallback_for_error(category: ErrorCategory) -> bool:
    """Return whether a failed attempt should be allowed to try the next fallback config.

    Authentication and content-filter failures are unlikely to be resolved by a
    different provider/model, so we skip fallback for those categories.
    Validation errors (e.g. provider not in cache) are also terminal.
    """
    terminal = {
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.CONTENT_FILTER,
        ErrorCategory.VALIDATION_ERROR,
    }
    return category not in terminal


def bind_context(ctx: LogContext | None = None, **kwargs: Any) -> Any:
    """Bind structured context to Loguru's logger.

    Args:
        ctx: Optional pre-built log context.
        **kwargs: Additional key/value pairs merged into the bound logger.

    Returns:
        A Loguru bound logger.
    """
    if ctx is None:
        return logger.bind(**kwargs)
    return logger.bind(**ctx.to_extra(), **kwargs)
