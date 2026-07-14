"""Per-worker progress UI for batch execution.

Owns the ``rich.Progress`` instance, the per-task display metadata, and the
worker-slot bar pool (B6: bars scale with the worker count, not the task
count). Extracted from ``GlobalBatchProcessor`` (ARCHITECTURE_REVIEW.md §7.2 /
P1.3, ``ProgressPresenter``) so the UI concern is separable from execution.
"""

from __future__ import annotations

import queue as _queue

from rich.console import Console as RichConsole
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Per-task display metadata: (input_token_estimate, estimated_output, model_key).
TaskMeta = dict[int, tuple[int, int, str]]


class ProgressPresenter:
    """Real progress presenter: a pool of per-worker bars over one Progress."""

    def __init__(self, task_meta: TaskMeta, num_slots: int) -> None:
        self.task_meta = task_meta
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            console=RichConsole(),
            transient=False,
        )
        self.slot_bars: list[TaskID] = [
            self.progress.add_task(f"[dim]worker {i} idle[/dim]", total=1)
            for i in range(num_slots)
        ]
        self.free_slots: _queue.Queue[int] = _queue.Queue()
        for i in range(num_slots):
            self.free_slots.put(i)
        self.active = True

    def start(self) -> None:
        self.progress.start()

    def stop(self) -> None:
        self.progress.stop()

    def acquire(self, task_id: int) -> tuple[TaskID, int, int]:
        """Claim a worker slot and relabel its bar for *task_id*.

        Returns ``(progress_task_id, input_tokens, slot_idx)``. The slot pool
        size equals the worker count, so ``get()`` never blocks long.
        """
        slot_idx = self.free_slots.get()
        progress_task_id = self.slot_bars[slot_idx]
        in_tok, est_out, model_key = self.task_meta.get(
            task_id, (0, 1, "unknown/model")
        )
        self.progress.update(
            progress_task_id,
            description=f"[cyan]{model_key}[/cyan] Task {task_id} ({in_tok} tok in)",
            total=est_out,
            completed=0,
        )
        return progress_task_id, in_tok, slot_idx

    def release(self, slot_idx: int) -> None:
        """Return a worker slot to the pool."""
        self.free_slots.put(slot_idx)


class NullProgressPresenter:
    """No-op presenter used when progress display is disabled."""

    active = False
    progress: Progress | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def acquire(self, task_id: int) -> tuple[TaskID | None, int | None, int | None]:
        return None, None, None

    def release(self, slot_idx: int | None) -> None:
        pass
