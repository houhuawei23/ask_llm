"""Batch processing data models; processors live in :mod:`ask_llm.core.batch_processor`."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    pass


from ask_llm.core.models import RequestMetadata
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
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)


class BatchTask(BaseModel):
    """A single batch processing task."""

    task_id: int
    prompt: str
    content: str
    output_filename: Optional[str] = None  # Optional output filename for split mode
    task_model_config: Optional[ModelConfig] = (
        None  # Optional for backward compatibility (renamed from model_config to avoid Pydantic reserved keyword)
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
    tasks: List[BatchTask],
    default_model: str,
) -> List[BatchTask]:
    """
    Sort tasks by descending estimated full prompt tokens.

    When concurrent workers are fewer than tasks, heavy requests start earlier and reduce
    wall-clock tail latency.
    """

    def _estimate(t: BatchTask) -> int:
        model = t.task_model_config.model if t.task_model_config else default_model
        if "{content}" in t.prompt:
            full = t.prompt.replace("{content}", t.content)
        else:
            full = f"{t.prompt}\n\n{t.content}"
        return int(TokenCounter.estimate_tokens(full, model)["token_count"])

    return sorted(tasks, key=_estimate, reverse=True)


class BatchResult(BaseModel):
    """Result of a batch processing task."""

    model_config = ConfigDict(protected_namespaces=())

    task_id: int
    prompt: str
    content: str
    output_filename: Optional[str] = None  # Optional output filename for split mode
    model_settings: (
        ModelConfig  # Renamed from model_config to avoid conflict with Pydantic's reserved field
    )
    response: Optional[str] = None
    metadata: Optional[RequestMetadata] = None
    reasoning: Optional[str] = None  # e.g. DeepSeek reasoner when paper_mode + return_reasoning
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = 0


class BatchStatistics(BaseModel):
    """Statistics for batch processing."""

    total_tasks: int = Field(default=0, description="Total number of tasks")
    successful_tasks: int = Field(default=0, description="Number of successful tasks")
    failed_tasks: int = Field(default=0, description="Number of failed tasks")
    total_latency: float = Field(default=0.0, description="Total latency in seconds")
    average_latency: float = Field(default=0.0, description="Average latency per task")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")


from ask_llm.core.batch_processor import BatchProcessor, GlobalBatchProcessor  # noqa: E402

__all__ = [
    "BatchProcessor",
    "BatchResult",
    "BatchStatistics",
    "BatchTask",
    "GlobalBatchProcessor",
    "ModelConfig",
    "TaskStatus",
    "sort_batch_tasks_by_estimated_input",
]
