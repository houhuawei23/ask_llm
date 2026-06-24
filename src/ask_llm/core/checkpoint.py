"""Generic checkpoint persistence for resumable commands.

Provides a base class that concrete checkpoints (batch, translation) can extend
by implementing serialization hooks for their task and result types.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from loguru import logger

TTask = TypeVar("TTask")
TResult = TypeVar("TResult")

CHECKPOINT_VERSION = 1


@dataclass
class BaseCheckpoint(ABC, Generic[TTask, TResult]):
    """Generic checkpoint for commands that can resume after interruption.

    Attributes:
        version: Checkpoint schema version.
        command: CLI command name (e.g., ``batch``, ``trans``).
        created_at: ISO timestamp when the checkpoint was created.
        config_digest: Hash/digest of the relevant configuration for consistency checks.
        completed_task_ids: Task IDs that have finished successfully.
        failed_tasks: Task contexts that failed and may be retried.
        successful_results: Results from completed tasks.
    """

    version: int
    command: str
    created_at: str
    config_digest: str
    completed_task_ids: list[int] = field(default_factory=list)
    failed_tasks: list[TTask] = field(default_factory=list)
    successful_results: list[TResult] = field(default_factory=list)

    @abstractmethod
    def task_to_dict(self, task: TTask) -> dict[str, Any]:
        """Serialize a task to a dictionary."""

    @abstractmethod
    def result_to_dict(self, result: TResult) -> dict[str, Any]:
        """Serialize a result to a dictionary."""

    @abstractmethod
    def task_from_dict(self, data: dict[str, Any]) -> TTask:
        """Deserialize a task from a dictionary."""

    @abstractmethod
    def result_from_dict(self, data: dict[str, Any]) -> TResult:
        """Deserialize a result from a dictionary."""

    @abstractmethod
    def result_task_id(self, result: TResult) -> int:
        """Return the task ID associated with a result."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "version": self.version,
            "command": self.command,
            "created_at": self.created_at,
            "config_digest": self.config_digest,
            "completed_task_ids": self.completed_task_ids,
            "failed_tasks": [self.task_to_dict(t) for t in self.failed_tasks],
            "successful_results": [self.result_to_dict(r) for r in self.successful_results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseCheckpoint[TTask, TResult]:
        """Deserialize checkpoint from dictionary.

        Note:
            Concrete subclasses should override this to call their own
            serialization hooks and return the correct subclass instance.
        """
        raise NotImplementedError(
            "Concrete checkpoints must implement from_dict() with their own serializers"
        )

    def save(self, path: str | Path) -> None:
        """Atomically save checkpoint to JSON file."""
        path = Path(path)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> BaseCheckpoint[TTask, TResult]:
        """Load checkpoint from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def is_completed(self, task_id: int) -> bool:
        """Return True if *task_id* has already completed successfully."""
        return task_id in self.completed_task_ids

    def merge(self, new_results: list[TResult]) -> None:
        """Merge new successful results into the checkpoint."""
        for result in new_results:
            task_id = self.result_task_id(result)
            if task_id not in self.completed_task_ids:
                self.completed_task_ids.append(task_id)
            self.successful_results.append(result)

    def add_failed_task(self, task: TTask, task_id: int) -> None:
        """Add a failed task context for retry.

        The task is only added if it is not already marked completed.
        """
        if task_id in self.completed_task_ids:
            return
        self.failed_tasks.append(task)

    def mark_all_failed_for_retry(self, failed_results: list[TResult]) -> None:
        """Replace the failed task list with tasks derived from failed results.

        Subclasses must map results back to tasks via :meth:`result_to_task`.
        """
        self.failed_tasks = [self.result_to_task(r) for r in failed_results]

    @abstractmethod
    def result_to_task(self, result: TResult) -> TTask:
        """Reconstruct a retryable task from a failed result."""
