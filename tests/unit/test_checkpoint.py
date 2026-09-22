"""Unit tests for generic checkpoint persistence."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ask_llm.core.batch_models import BatchResult, BatchTask, ModelConfig
from ask_llm.core.batch_checkpoint import BatchCheckpoint
from ask_llm.core.checkpoint import CHECKPOINT_VERSION, atomic_write_text


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
    checkpoint.failed_tasks.append(task)

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
    # Unique tmp names: no leftover tmp file of any naming scheme.
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_concurrent_writers_no_corruption(tmp_path):
    """Two concurrent savers of the same target must not share a tmp file."""
    path = tmp_path / "checkpoint.json"
    errors: list[Exception] = []

    def write(tag: str) -> None:
        try:
            atomic_write_text(path, f"payload-{tag}")
        except Exception as e:  # pragma: no cover - only on failure
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    content = path.read_text(encoding="utf-8")
    assert content.startswith("payload-")
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_failure_leaves_no_tmp(tmp_path, monkeypatch):
    """A failure after tmp creation must clean up and leave the original intact."""
    target = tmp_path / "secret.txt"
    target.write_text("original", encoding="utf-8")

    def exploding_chmod(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("ask_llm.core.checkpoint.os.chmod", exploding_chmod)
    with pytest.raises(RuntimeError, match="boom"):
        atomic_write_text(target, "new", mode=0o600)
    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_mode_sets_permissions(tmp_path):
    target = tmp_path / "secret.txt"
    atomic_write_text(target, "secret", mode=0o600)
    assert (target.stat().st_mode & 0o777) == 0o600
