"""Unit tests for generic checkpoint persistence."""

from __future__ import annotations

from pathlib import Path

from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig
from ask_llm.core.batch_checkpoint import BatchCheckpoint
from ask_llm.core.checkpoint import CHECKPOINT_VERSION


def _make_task(task_id: int = 0) -> BatchTask:
    return BatchTask(
        task_id=task_id,
        prompt="Translate: {content}",
        content="hello",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
    )


def _make_result(task_id: int = 0) -> BatchResult:
    return BatchResult(
        task_id=task_id,
        prompt="Translate: {content}",
        content="hello",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
        response="bonjour",
        status="success",
    )


def test_batch_checkpoint_roundtrip(tmp_path):
    checkpoint = BatchCheckpoint.create(command="batch", config_digest="abc")
    task = _make_task(task_id=1)
    result = _make_result(task_id=0)
    checkpoint.merge([result])
    checkpoint.add_failed_task(task, task_id=1)

    path = tmp_path / "checkpoint.json"
    checkpoint.save(path)
    loaded = BatchCheckpoint.load(path)

    assert loaded.version == CHECKPOINT_VERSION
    assert loaded.command == "batch"
    assert loaded.config_digest == "abc"
    assert loaded.completed_task_ids == [0]
    assert len(loaded.successful_results) == 1
    assert loaded.successful_results[0].response == "bonjour"
    assert len(loaded.failed_tasks) == 1
    assert loaded.failed_tasks[0].task_id == 1


def test_is_completed():
    checkpoint = BatchCheckpoint.create(command="batch", config_digest="abc")
    checkpoint.completed_task_ids = [1, 3]
    assert checkpoint.is_completed(1)
    assert not checkpoint.is_completed(2)


def test_merge_skips_duplicate_task_ids():
    checkpoint = BatchCheckpoint.create(command="batch", config_digest="abc")
    result1 = _make_result(task_id=0)
    result2 = _make_result(task_id=0)
    checkpoint.merge([result1, result2])
    assert checkpoint.completed_task_ids == [0]
    assert len(checkpoint.successful_results) == 2


def test_add_failed_task_skips_completed():
    checkpoint = BatchCheckpoint.create(command="batch", config_digest="abc")
    checkpoint.completed_task_ids = [0]
    checkpoint.add_failed_task(_make_task(task_id=0), task_id=0)
    assert len(checkpoint.failed_tasks) == 0


def test_result_to_task():
    checkpoint = BatchCheckpoint.create(command="batch", config_digest="abc")
    result = _make_result(task_id=5)
    task = checkpoint.result_to_task(result)
    assert task.task_id == 5
    assert task.prompt == result.prompt
    assert task.model_settings == result.model_settings


def test_save_is_atomic(tmp_path):
    checkpoint = BatchCheckpoint.create(command="batch", config_digest="abc")
    path = tmp_path / "checkpoint.json"
    checkpoint.save(path)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
