"""Thin wrapper around GlobalBatchProcessor + process_global_tasks (trans / paper / batch)."""

from __future__ import annotations

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import BatchResult, BatchTask, GlobalBatchProcessor


def run_global_batch_tasks(
    tasks: list[BatchTask],
    config_manager: ConfigManager,
    *,
    max_workers: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_delay_max: float = 10.0,
    verbose: bool = False,
    show_progress: bool = True,
    clamp_workers_to_task_count: bool = False,
) -> tuple[list[BatchResult], GlobalBatchProcessor]:
    """
    Create a GlobalBatchProcessor and run process_global_tasks.

    When ``clamp_workers_to_task_count`` is True (e.g. paper explain), worker count is
    ``max(1, min(max_workers, len(tasks)))`` so we do not over-allocate threads for few jobs.

    For translation-style workloads with a fixed thread pool size, pass
    ``clamp_workers_to_task_count=False`` (default); the executor will not use extra threads anyway.
    """
    n = len(tasks)
    if clamp_workers_to_task_count and n > 0:
        effective_workers = max(1, min(max_workers, n))
    else:
        effective_workers = max(1, max_workers)

    processor = GlobalBatchProcessor(
        max_workers=effective_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_delay_max=retry_delay_max,
        verbose=verbose,
    )
    results = processor.process_global_tasks(tasks, config_manager, show_progress=show_progress)
    return results, processor
