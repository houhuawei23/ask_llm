"""Shared orchestration skeleton for chunked LLM jobs (P3.3).

``HeadingFormatter`` and ``BodyFormatter`` implement the same pipeline:
resolve config with built-in fallbacks → load prompt template → build work
units → run through the shared bounded runner → collect ordered results →
save a checkpoint when units fail. This base owns that skeleton; subclasses
only define their work units, per-unit LLM call, and result assembly.

Resume is symmetric: any subclass gets ``resume_from_checkpoint`` semantics
through :meth:`_retry_failed_units`, ending the historical asymmetry where
title checkpoints were written but could never be resumed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from loguru import logger

from ask_llm.core.concurrent import run_bounded_with_retries
from ask_llm.core.format_checkpoint import (
    CHECKPOINT_VERSION,
    FailedChunkInfo,
    FormatCheckpoint,
    SuccessfulChunkInfo,
    generate_checkpoint_path,
)
from ask_llm.core.processor import RequestProcessor

UnitT = TypeVar("UnitT")
WorkerResultT = TypeVar("WorkerResultT")


class ChunkedLLMJob:
    """Base class for chunked LLM format jobs (heading / body)."""

    def __init__(
        self,
        processor: RequestProcessor,
        *,
        prompt_template: str | None = None,
        prompt_file: str | None = None,
        concurrency: int,
        retries: int,
        retry_delay: float,
        retry_delay_max: float,
    ):
        """Initialize the shared job skeleton.

        Subclasses resolve their own config-section fallbacks and pass the
        final values here.
        """
        self.processor = processor
        self.prompt_template = prompt_template
        self.prompt_file = prompt_file
        self.concurrency = concurrency
        self.retries = retries
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max

        # Load prompt from file if specified
        if prompt_file:
            self.prompt_template = self._load_prompt_from_file(prompt_file)

    @staticmethod
    def _pick(override: Any, fallback: Any) -> Any:
        """Return *override* unless it is None, else *fallback*."""
        return override if override is not None else fallback

    def _resolve_template(self, error_msg: str) -> str:
        """Return the effective prompt template, loading from file if needed.

        Raises:
            ValueError: If no template is available.
        """
        template = self.prompt_template
        if not template and self.prompt_file:
            template = self._load_prompt_from_file(self.prompt_file)
        if not template:
            raise ValueError(error_msg)
        self.prompt_template = template
        return template

    @staticmethod
    def _load_prompt_from_file(prompt_path: str) -> str:
        """Load prompt template from file.

        Supports @ prefix for relative paths from project root.

        Raises:
            FileNotFoundError: If prompt file not found
            OSError: If file cannot be read
        """
        from ask_llm.utils.prompt_resolver import load_prompt_template

        return load_prompt_template(prompt_path)

    def _run_units(
        self,
        units: list[UnitT],
        worker: Callable[[UnitT, int], WorkerResultT],
        *,
        is_failed: Callable[[WorkerResultT], bool],
        error_message: Callable[[WorkerResultT], str],
        retry_count_from_result: Callable[[WorkerResultT], int],
        order_key: Callable[[WorkerResultT], Any],
    ) -> list[WorkerResultT]:
        """Run work units through the shared bounded runner (ordered results)."""
        max_workers = min(self.concurrency, len(units)) if units else 1
        return run_bounded_with_retries(
            units,
            worker,
            max_workers=max_workers,
            max_retries=self.retries,
            retry_delay=self.retry_delay,
            retry_delay_max=self.retry_delay_max,
            is_failed=is_failed,
            error_message=error_message,
            retry_count_from_result=retry_count_from_result,
            order_key=order_key,
        )

    def _save_checkpoint(
        self,
        *,
        source_file: str | None,
        format_type: str,
        model: str,
        prompt_template: str,
        max_chunk_tokens: int | None,
        failed_chunks: list[FailedChunkInfo],
        successful_chunks: list[SuccessfulChunkInfo],
        checkpoint_path: str | None = None,
    ) -> str | None:
        """Save a checkpoint for failed units; returns its path (or None).

        No-op when nothing failed or no source file was given.
        """
        if not failed_chunks or not source_file:
            return None
        path = checkpoint_path or str(generate_checkpoint_path(source_file, format_type))
        checkpoint = FormatCheckpoint(
            version=CHECKPOINT_VERSION,
            source_file=source_file,
            format_type=format_type,
            model=model,
            prompt_template=prompt_template,
            max_chunk_tokens=max_chunk_tokens,
            created_at=datetime.now().isoformat(),
            failed_chunks=failed_chunks,
            successful_chunks=successful_chunks,
        )
        checkpoint.save(path)
        return path

    def _retry_failed_units(
        self,
        checkpoint: FormatCheckpoint,
        units: list[UnitT],
        worker: Callable[[UnitT, int], WorkerResultT],
        *,
        is_failed: Callable[[WorkerResultT], bool],
        error_message: Callable[[WorkerResultT], str],
        retry_count_from_result: Callable[[WorkerResultT], int],
        order_key: Callable[[WorkerResultT], Any],
    ) -> list[WorkerResultT]:
        """Re-run checkpoint-failed work units through the shared runner."""
        logger.info(
            f"[{type(self).__name__}] Resuming from checkpoint: "
            f"failed_units={len(checkpoint.failed_chunks)}, "
            f"successful_units={len(checkpoint.successful_chunks)}"
        )
        return self._run_units(
            units,
            worker,
            is_failed=is_failed,
            error_message=error_message,
            retry_count_from_result=retry_count_from_result,
            order_key=order_key,
        )
