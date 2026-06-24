"""Execution report generation for batch, translation, and paper workflows.

A report captures per-task attempt histories, aggregate statistics, and failure
category breakdowns so users can diagnose production runs without re-running them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ask_llm import __version__
from ask_llm.core.batch_models import BatchResult, TaskStatus
from ask_llm.core.telemetry import ErrorCategory


class AttemptRecord(BaseModel):
    """Record of a single provider/model attempt for one task."""

    provider: str
    model: str
    status: TaskStatus
    error: str | None = None
    error_category: ErrorCategory | None = None
    latency: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    timestamp: datetime


class TaskRecord(BaseModel):
    """Record aggregating all attempts for one task."""

    task_id: int
    final_status: TaskStatus
    primary_provider: str
    primary_model: str
    attempts: list[AttemptRecord] = Field(default_factory=list)
    final_error: str | None = None
    final_error_category: ErrorCategory | None = None


class FailureSummary(BaseModel):
    """Aggregate failure breakdown by category."""

    total_failed_tasks: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)


class TokenSummary(BaseModel):
    """Aggregate token consumption summary."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0


class ExecutionReport(BaseModel):
    """Machine-readable report for a complete execution."""

    version: str = Field(default=__version__, description="Ask LLM version")
    command: str = Field(..., description="CLI command that produced the report")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    token_summary: TokenSummary = Field(default_factory=TokenSummary)
    failure_summary: FailureSummary = Field(default_factory=FailureSummary)
    tasks: list[TaskRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional command-specific metadata (e.g. input files)",
    )

    def finalize(self) -> None:
        """Compute aggregate summaries after all task records are added."""
        self.completed_at = datetime.now()
        self.total_tasks = len(self.tasks)
        self.successful_tasks = sum(1 for t in self.tasks if t.final_status == TaskStatus.SUCCESS)
        self.failed_tasks = self.total_tasks - self.successful_tasks

        self.token_summary = TokenSummary()
        self.failure_summary = FailureSummary()
        for task in self.tasks:
            for attempt in task.attempts:
                if attempt.status == TaskStatus.SUCCESS:
                    self.token_summary.total_input_tokens += attempt.input_tokens or 0
                    self.token_summary.total_output_tokens += attempt.output_tokens or 0
            if task.final_status == TaskStatus.FAILED:
                self.failure_summary.total_failed_tasks += 1
                category = task.final_error_category or ErrorCategory.UNKNOWN
                key = category.value
                self.failure_summary.by_category[key] = (
                    self.failure_summary.by_category.get(key, 0) + 1
                )

    def to_json_file(self, path: str) -> None:
        """Serialize the report to a JSON file."""
        from pathlib import Path

        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str) -> ExecutionReport:
        """Load a report from a JSON file."""
        from pathlib import Path

        data = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(data)


def _batch_result_to_attempt_records(result: BatchResult) -> list[AttemptRecord]:
    """Convert a BatchResult (and its attempt_history) into AttemptRecords."""
    records: list[AttemptRecord] = []
    # If a task went through a fallback chain, the final BatchResult carries the
    # full attempt_history; avoid double-counting the final result.
    sources = list(result.attempt_history) if result.attempt_history else [result]

    for src in sources:
        metadata = src.metadata
        records.append(
            AttemptRecord(
                provider=src.model_settings.provider,
                model=src.model_settings.model,
                status=src.status,
                error=src.error,
                error_category=src.error_category,
                latency=metadata.latency if metadata else None,
                input_tokens=metadata.input_tokens if metadata else None,
                output_tokens=metadata.output_tokens if metadata else None,
                timestamp=src.timestamp,
            )
        )
    return records


def build_report_from_batch_results(
    command: str,
    results: list[BatchResult],
    *,
    metadata: dict[str, Any] | None = None,
) -> ExecutionReport:
    """Build an execution report from a list of BatchResults."""
    report = ExecutionReport(command=command, metadata=metadata or {})

    grouped: dict[int, list[BatchResult]] = {}
    for result in results:
        grouped.setdefault(result.task_id, []).append(result)

    for task_id, task_results in sorted(grouped.items()):
        # Each task may appear once per model in legacy BatchProcessor mode, or
        # once total in GlobalBatchProcessor mode. Use the successful result if
        # any, otherwise the last one.
        success_result = next((r for r in task_results if r.status == TaskStatus.SUCCESS), None)
        representative = success_result or task_results[-1]
        attempts: list[AttemptRecord] = []
        for r in task_results:
            attempts.extend(_batch_result_to_attempt_records(r))

        report.tasks.append(
            TaskRecord(
                task_id=task_id,
                final_status=representative.status,
                primary_provider=representative.model_settings.provider,
                primary_model=representative.model_settings.model,
                attempts=attempts,
                final_error=representative.error,
                final_error_category=representative.error_category,
            )
        )

    report.finalize()
    return report
