"""Structured observability primitives for Ask LLM.

Provides error classification, request/task log context, and helpers for
injecting correlation IDs into Loguru logs without leaking CLI details into
core modules.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    """High-level failure category for an API call or task attempt."""

    SUCCESS = "success"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTENT_FILTER = "content_filter"
    MODEL_ERROR = "model_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


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

    The heuristic is intentionally conservative: we only categorize errors that
    are clearly identifiable from common provider/HTTP signatures. Everything
    else falls back to ``UNKNOWN`` so users are not misled.
    """
    if not error_message:
        return ErrorCategory.UNKNOWN

    text = error_message.lower()

    auth_keywords = (
        "401",
        "403",
        "authentication",
        "unauthorized",
        "invalid api key",
        "api key invalid",
        "authentication_error",
        "access denied",
        "invalid token",
    )
    if any(k in text for k in auth_keywords):
        return ErrorCategory.AUTHENTICATION

    rate_limit_keywords = (
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "throttled",
        "quota exceeded",
        "insufficient_quota",
    )
    if any(k in text for k in rate_limit_keywords):
        return ErrorCategory.RATE_LIMIT

    timeout_keywords = (
        "timeout",
        "timed out",
        "time out",
        "deadline exceeded",
    )
    if any(k in text for k in timeout_keywords):
        return ErrorCategory.TIMEOUT

    content_filter_keywords = (
        "content filter",
        "content_filter",
        "content policy",
        "moderation",
        "safety",
        "blocked",
        "inappropriate content",
        "content rejected",
    )
    if any(k in text for k in content_filter_keywords):
        return ErrorCategory.CONTENT_FILTER

    model_error_keywords = (
        "model not found",
        "invalid model",
        "model error",
        "bad request",
        "invalid_request_error",
        "context length",
        "too long",
        "maximum context",
    )
    if any(k in text for k in model_error_keywords):
        return ErrorCategory.MODEL_ERROR

    network_keywords = (
        "connection",
        "connect",
        "network",
        "dns",
        "unreachable",
        "refused",
        "ssl",
        "certificate",
        "proxy",
    )
    if any(k in text for k in network_keywords):
        return ErrorCategory.NETWORK_ERROR

    validation_keywords = (
        "validation",
        "invalid",
        "required",
        "missing",
        "not found in cache",
    )
    if any(k in text for k in validation_keywords):
        return ErrorCategory.VALIDATION_ERROR

    return ErrorCategory.UNKNOWN


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
