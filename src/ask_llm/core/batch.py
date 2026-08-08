"""Backward-compatible re-exports for batch data models and processors.

New code should import directly from :mod:`ask_llm.core.batch_models` for data models
and :mod:`ask_llm.core.batch_processor` for processors.
"""

from ask_llm.core.batch_models import (
    BatchResult,
    BatchStatistics,
    BatchTask,
    ModelConfig,
    TaskStatus,
    sort_batch_tasks_by_estimated_input,
)
from ask_llm.core.batch_processor import GlobalBatchProcessor

__all__ = [
    "BatchResult",
    "BatchStatistics",
    "BatchTask",
    "GlobalBatchProcessor",
    "ModelConfig",
    "TaskStatus",
    "sort_batch_tasks_by_estimated_input",
]
