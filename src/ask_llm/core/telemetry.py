"""Structured observability primitives for Ask LLM.

Provides request/task log context and helpers for injecting correlation IDs
into Loguru logs without leaking CLI details into core modules. Error
classification lives in ``ask_llm.core.error_keywords`` (the single rule
table); import it from there directly.
"""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


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
