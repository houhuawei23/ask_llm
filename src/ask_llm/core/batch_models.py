"""Batch processing data models; processors live in :mod:`ask_llm.core.batch_processor`."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ask_llm.core.models import RequestMetadata
from ask_llm.core.telemetry import ErrorCategory
from ask_llm.utils.token_counter import TokenCounter


class TaskStatus(str, Enum):
    """Task processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class ModelConfig(BaseModel):
    """Model configuration for batch processing."""

    provider: str
    model: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)


class BatchTask(BaseModel):
    """A single batch processing task."""

    task_id: int
    prompt: str
    content: str
    output_filename: str | None = None  # Optional output filename for split mode
    model_settings: ModelConfig | None = None  # Optional per-task model configuration
    fallback_model_configs: list[ModelConfig] = Field(
        default_factory=list,
        description="Ordered list of fallback provider/model configs to try on failure",
    )
    task_kind: Literal["translation_chunk", "paper_explain"] = "translation_chunk"
    paper_mode: bool = False  # legacy; if True, task_kind is coerced to paper_explain
    return_reasoning: bool = False

    @model_validator(mode="before")
    @classmethod
    def _legacy_paper_mode(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("paper_mode")
            and data.get("task_kind", "translation_chunk") == "translation_chunk"
        ):
            return {**data, "task_kind": "paper_explain"}
        return data


def sort_batch_tasks_by_estimated_input(
    tasks: list[BatchTask],
    default_model: str,
) -> list[BatchTask]:
    """
    Sort tasks by descending estimated full prompt tokens.

    When concurrent workers are fewer than tasks, heavy requests start earlier and reduce
    wall-clock tail latency.
    """

    def _estimate(t: BatchTask) -> int:
        model = t.model_settings.model if t.model_settings else default_model
        if "{content}" in t.prompt:
            full = t.prompt.replace("{content}", t.content)
        else:
            full = f"{t.prompt}\n\n{t.content}"
        return int(TokenCounter.estimate_tokens(full, model)["token_count"])

    return sorted(tasks, key=_estimate, reverse=True)


class AttemptRecord(BaseModel):
    """Flat, non-recursive record of a single provider/model attempt for one task.

    ``BatchResult.attempt_history`` is a ``list[AttemptRecord]`` (not
    ``list[BatchResult]``) so the object graph is acyclic by construction. The
    prior self-referential type caused the v2.15.1 circular-reference crash
    during checkpoint serialization; a flat record type makes that class of bug
    structurally impossible. See ARCHITECTURE_REVIEW.md bug B7.
    """

    provider: str
    model: str
    status: TaskStatus
    error: str | None = None
    error_category: ErrorCategory | None = None
    latency: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_result(cls, result: "BatchResult") -> "AttemptRecord":
        """Project a :class:`BatchResult` into a flat attempt record."""
        metadata = result.metadata
        return cls(
            provider=result.model_settings.provider,
            model=result.model_settings.model,
            status=result.status,
            error=result.error,
            error_category=result.error_category,
            latency=metadata.latency if metadata else None,
            input_tokens=metadata.input_tokens if metadata else None,
            output_tokens=metadata.output_tokens if metadata else None,
            timestamp=result.timestamp,
        )


class BatchResult(BaseModel):
    """Result of a batch processing task."""

    model_config = ConfigDict(protected_namespaces=())

    task_id: int
    prompt: str
    content: str
    output_filename: str | None = None  # Optional output filename for split mode
    model_settings: ModelConfig  # Model configuration used for this task
    response: str | None = None
    metadata: RequestMetadata | None = None
    reasoning: str | None = None  # e.g. DeepSeek reasoner when paper_mode + return_reasoning
    status: TaskStatus = TaskStatus.PENDING
    error: str | None = None
    error_category: ErrorCategory | None = Field(
        default=None,
        description="Classified failure category when status is FAILED",
    )
    attempt_history: list[AttemptRecord] = Field(
        default_factory=list,
        description=(
            "Preceding attempts for this task (e.g. earlier configs in a fallback "
            "chain). Flat AttemptRecords, never the final result itself, so the "
            "object graph stays acyclic."
        ),
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = 0

    def project(self) -> dict[str, Any]:
        """Single projection of this result for exporters and reports (P4.7).

        The one canonical dict shape for a ``BatchResult``; exporters
        (``BatchResultExporter._prepare_data``) build on it instead of
        hand-assembling per-result dicts.
        """
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "content": self.content,
            "model_settings": {
                "provider": self.model_settings.provider,
                "model": self.model_settings.model,
                "temperature": self.model_settings.temperature,
                "top_p": self.model_settings.top_p,
            },
            "response": self.response,
            "status": self.status.value,
            "error": self.error,
            "metadata": {
                "provider": self.metadata.provider,
                "model": self.metadata.model,
                "temperature": self.metadata.temperature,
                "input_tokens": self.metadata.input_tokens,
                "output_tokens": self.metadata.output_tokens,
                "latency": self.metadata.latency,
                "timestamp": self.metadata.timestamp.isoformat(),
            }
            if self.metadata
            else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "retry_count": self.retry_count,
        }


class BatchStatistics(BaseModel):
    """Statistics for batch processing."""

    total_tasks: int = Field(default=0, description="Total number of tasks")
    successful_tasks: int = Field(default=0, description="Number of successful tasks")
    failed_tasks: int = Field(default=0, description="Number of failed tasks")
    total_latency: float = Field(default=0.0, description="Total latency in seconds")
    average_latency: float = Field(default=0.0, description="Average latency per task")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")

    @classmethod
    def from_results(cls, results: list[BatchResult]) -> dict[str, "BatchStatistics"]:
        """Aggregate per-``(provider, model)`` statistics from batch results.

        Single source of truth for batch statistics. Replaces the previously
        duplicated aggregators that lived in ``batch_processor`` and
        ``batch_service`` (ARCHITECTURE_REVIEW.md §4.1.2, P1.5). Pure function.
        """
        grouped: dict[str, list[BatchResult]] = {}
        for result in results:
            key = f"{result.model_settings.provider}/{result.model_settings.model}"
            grouped.setdefault(key, []).append(result)

        statistics: dict[str, BatchStatistics] = {}
        for key, model_results in grouped.items():
            stats = cls(total_tasks=len(model_results))
            successful = [r for r in model_results if r.status == TaskStatus.SUCCESS]
            stats.successful_tasks = len(successful)
            stats.failed_tasks = len(model_results) - stats.successful_tasks
            if successful:
                latencies = [r.metadata.latency for r in successful if r.metadata]
                if latencies:
                    stats.total_latency = sum(latencies)
                    stats.average_latency = stats.total_latency / len(latencies)
                stats.total_input_tokens = sum(
                    r.metadata.input_tokens for r in successful if r.metadata
                )
                stats.total_output_tokens = sum(
                    r.metadata.output_tokens for r in successful if r.metadata
                )
            statistics[key] = stats
        return statistics
