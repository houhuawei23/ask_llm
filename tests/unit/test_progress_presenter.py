"""Unit tests for the extracted progress presenter (P1.3 / ProgressPresenter)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ask_llm.core.progress_presenter import NullProgressPresenter, ProgressPresenter


def test_progress_presenter_creates_one_bar_per_slot():
    with patch("ask_llm.core.progress_presenter.Progress") as mock_progress_cls:
        progress = MagicMock()
        progress.add_task.return_value = "task-id"
        mock_progress_cls.return_value = progress

        presenter = ProgressPresenter(task_meta={}, num_slots=4)

    # Exactly num_slots bars, independent of task count.
    assert progress.add_task.call_count == 4
    assert len(presenter.slot_bars) == 4
    assert presenter.active is True


def test_acquire_relabels_bar_and_release_returns_slot():
    task_meta = {7: (123, 456, "prov/model")}
    with patch("ask_llm.core.progress_presenter.Progress"):
        presenter = ProgressPresenter(task_meta=task_meta, num_slots=2)

    _progress_task_id, input_tokens, slot_idx = presenter.acquire(7)
    # Bar relabeled with the task's model + input-token estimate.
    presenter.progress.update.assert_called_once()
    _, kwargs = presenter.progress.update.call_args
    assert "prov/model" in kwargs["description"]
    assert "123 tok in" in kwargs["description"]
    assert kwargs["total"] == 456
    assert kwargs["completed"] == 0
    assert input_tokens == 123

    # Releasing returns the slot to the pool.
    presenter.release(slot_idx)
    assert presenter.free_slots.qsize() == 2  # both slots free again


def test_null_presenter_is_noop():
    presenter = NullProgressPresenter()
    assert presenter.active is False
    assert presenter.progress is None
    # acquire/release/start/stop are safe no-ops.
    assert presenter.acquire(1) == (None, None, None)
    presenter.release(None)
    presenter.start()
    presenter.stop()
