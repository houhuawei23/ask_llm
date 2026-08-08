"""Shared batch checkpoint lifecycle (P4.1).

One canonical implementation of the create → optional resume load → filter
completed → run → merge → mark failed → save → unlink-on-full-success flow
previously copied (with drift) into ``batch_service`` and
``translation_service``. Both services now delegate here.

Canonical decisions where the copies drifted:
- Early return with everything already completed: the checkpoint is **kept**
  (batch behavior) — the caller still holds prior results and may want the
  file for auditing; translation previously unlinked it.
- Unlink happens only on a non-interrupted run with zero failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import BatchResult, BatchTask, GlobalBatchProcessor
from ask_llm.core.batch_checkpoint import BatchCheckpoint
from ask_llm.core.batch_models import TaskStatus
from ask_llm.core.global_batch_runner import run_global_batch_tasks


@dataclass
class CheckpointRunOutcome:
    """Result of a checkpointed batch run."""

    results: list[BatchResult]  # prior + new successful results plus new failures
    checkpoint_path: str
    new_results: list[BatchResult] = field(default_factory=list)  # this run only
    new_failed: list[BatchResult] = field(default_factory=list)
    interrupted: bool = False
    all_previously_completed: bool = False  # nothing left to run
    checkpoint_deleted: bool = False
    processor: GlobalBatchProcessor | None = field(default=None, repr=False)


def run_with_checkpoint(
    *,
    command: str,
    config_digest: str,
    checkpoint_path: str,
    tasks: list[BatchTask],
    config_manager: ConfigManager,
    resume: bool,
    max_retries: int,
    retry_delay: float = 1.0,
    retry_delay_max: float = 10.0,
    max_workers: int,
    verbose: bool = False,
    show_progress: bool = True,
    clamp_workers_to_task_count: bool = False,
    stream_api: bool = True,
) -> CheckpointRunOutcome:
    """Run *tasks* under the shared checkpoint lifecycle.

    Args:
        command: Command name recorded in the checkpoint ("batch" / "trans").
        config_digest: Digest of the source config/input recorded in the checkpoint.
        checkpoint_path: Where the checkpoint file lives.
        tasks: All tasks for the run (pre-resume-filtering).
        config_manager: Active config manager.
        resume: When True and *checkpoint_path* exists, completed tasks are
            filtered out and prior successful results are carried forward.
        max_retries / retry_delay / retry_delay_max: Retry policy for the runner.
        max_workers: Concurrency for the runner.
        verbose / show_progress / clamp_workers_to_task_count / stream_api:
            Forwarded to ``run_global_batch_tasks``.

    Returns:
        :class:`CheckpointRunOutcome` with merged results and lifecycle flags.
    """
    checkpoint = BatchCheckpoint.create(command=command, config_digest=config_digest)

    # 1. Optional resume: load prior progress, filter completed tasks.
    if resume and Path(checkpoint_path).exists():
        checkpoint = BatchCheckpoint.load(checkpoint_path)
        remaining = [t for t in tasks if not checkpoint.is_completed(t.task_id)]
        if len(remaining) < len(tasks):
            logger.info(
                f"[{command}] Resuming: {len(tasks) - len(remaining)} tasks already completed, "
                f"{len(remaining)} remaining"
            )
        tasks = remaining

    # 2. Early return: everything already completed (checkpoint kept).
    if not tasks:
        return CheckpointRunOutcome(
            results=list(checkpoint.successful_results),
            new_results=[],
            new_failed=[],
            checkpoint_path=checkpoint_path,
            all_previously_completed=True,
        )

    # 3. Run remaining tasks.
    new_results, processor = run_global_batch_tasks(
        tasks,
        config_manager,
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_delay_max=retry_delay_max,
        verbose=verbose,
        show_progress=show_progress,
        clamp_workers_to_task_count=clamp_workers_to_task_count,
        stream_api=stream_api,
    )

    # 4. Merge into checkpoint and persist.
    new_successful = [r for r in new_results if r.status == TaskStatus.SUCCESS]
    new_failed = [r for r in new_results if r.status == TaskStatus.FAILED]
    checkpoint.merge(new_successful)
    checkpoint.mark_all_failed_for_retry(new_failed)
    checkpoint.save(checkpoint_path)

    results = list(checkpoint.successful_results) + new_failed

    # 5. Unlink only on a clean, non-interrupted full success.
    metrics = getattr(processor, "last_metrics", None)
    interrupted = bool(getattr(metrics, "interrupted", False))
    checkpoint_deleted = False
    if interrupted:
        logger.warning(
            f"[{command}] Interrupted: {len(checkpoint.successful_results)} successful results "
            f"saved to checkpoint {checkpoint_path}; resume to continue"
        )
    elif not new_failed:
        Path(checkpoint_path).unlink(missing_ok=True)
        checkpoint_deleted = True

    return CheckpointRunOutcome(
        results=results,
        new_results=new_results,
        new_failed=new_failed,
        checkpoint_path=checkpoint_path,
        interrupted=interrupted,
        checkpoint_deleted=checkpoint_deleted,
        processor=processor,
    )
