"""Checkpoint persistence for batch and translation commands.

Uses the generic :class:`BaseCheckpoint` with ``BatchTask`` / ``BatchResult``
serialization. Both ``ask-llm batch`` and ``ask-llm trans`` can use this
concrete checkpoint type because they share the same task/result model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ask_llm.core.batch import BatchResult, BatchTask
from ask_llm.core.checkpoint import CHECKPOINT_VERSION, BaseCheckpoint


@dataclass
class BatchCheckpoint(BaseCheckpoint[BatchTask, BatchResult]):
    """Checkpoint for resumable batch/translation runs."""

    @classmethod
    def create(
        cls,
        *,
        command: str,
        config_digest: str,
    ) -> BatchCheckpoint:
        """Create a fresh checkpoint for a command run."""
        from datetime import datetime

        return cls(
            version=CHECKPOINT_VERSION,
            command=command,
            created_at=datetime.now().isoformat(),
            config_digest=config_digest,
        )

    def task_to_dict(self, task: BatchTask) -> dict[str, Any]:
        return task.model_dump(mode="json")

    def result_to_dict(self, result: BatchResult) -> dict[str, Any]:
        return result.model_dump(mode="json")

    def task_from_dict(self, data: dict[str, Any]) -> BatchTask:
        return BatchTask.model_validate(data)

    def result_from_dict(self, data: dict[str, Any]) -> BatchResult:
        return BatchResult.model_validate(data)

    def result_task_id(self, result: BatchResult) -> int:
        return result.task_id

    def result_to_task(self, result: BatchResult) -> BatchTask:
        return BatchTask(
            task_id=result.task_id,
            prompt=result.prompt,
            content=result.content,
            output_filename=result.output_filename,
            model_settings=result.model_settings,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchCheckpoint:
        """Deserialize a batch checkpoint from a dictionary."""
        return cls(
            version=data.get("version", CHECKPOINT_VERSION),
            command=data["command"],
            created_at=data.get("created_at", ""),
            config_digest=data.get("config_digest", ""),
            completed_task_ids=list(data.get("completed_task_ids", [])),
            failed_tasks=[BatchTask.model_validate(t) for t in data.get("failed_tasks", [])],
            successful_results=[
                BatchResult.model_validate(r) for r in data.get("successful_results", [])
            ],
        )

    @classmethod
    def load(cls, path: str | Path) -> BatchCheckpoint:
        """Load a batch checkpoint from JSON file."""
        import json

        path_obj = Path(path)
        data = json.loads(path_obj.read_text(encoding="utf-8"))
        return cls.from_dict(data)
