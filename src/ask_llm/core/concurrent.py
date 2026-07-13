"""Bounded single-queue concurrent runner with retries.

Replaces the previous ``llm_engine.concurrent.run_thread_pool_with_retries`` usage.
The external helper submits every task to a ``ThreadPoolExecutor`` up front, which
for large batches can allocate a huge number of in-flight futures.  This runner
keeps at most ``max_workers`` tasks in flight at any time, using a bounded work
queue and a scheduler thread for retry delays, so the same thread pool can be
reused across file/chunk layers without nesting executors.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from loguru import logger

TTask = TypeVar("TTask")
TResult = TypeVar("TResult")


@dataclass
class RunMetrics:
    """Runtime metrics produced by :class:`BoundedRetryRunner`."""

    total_tasks: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    total_latency: float = 0.0
    average_latency: float = 0.0


def exponential_backoff_seconds(
    attempt_1_based: int,
    *,
    initial: float,
    maximum: float,
) -> float:
    """Delay before retry *attempt_1_based* (1 = first retry after initial failure).

    ``min(initial * 2 ** (n - 1), maximum)`` — matches the previous runner.
    """
    if attempt_1_based < 1:
        return 0.0
    raw: float = initial * (2 ** (attempt_1_based - 1))
    return float(min(raw, maximum))


class BoundedRetryRunner(Generic[TTask, TResult]):
    """Run tasks with bounded concurrency, retries and exponential backoff.

    The runner submits at most ``max_workers`` tasks to the executor at once.
    Retries are scheduled on a private heap and re-enqueued when their backoff
    expires, so worker threads are never blocked sleeping for retries.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        max_retries: int,
        retry_delay: float,
        retry_delay_max: float,
    ) -> None:
        self.max_workers = max(1, max_workers)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max

    def run_with_metrics(
        self,
        tasks: Sequence[TTask],
        worker: Callable[[TTask, int], TResult],
        *,
        is_failed: Callable[[TResult], bool],
        error_message: Callable[[TResult], str],
        retry_count_from_result: Callable[[TResult], int],
        is_retryable_error: Callable[[str], bool] | None = None,
        on_worker_exception: Callable[[TTask, BaseException], TResult] | None = None,
        on_retry_scheduled: Callable[[TTask, TResult], None] | None = None,
        order_key: Callable[[TResult], Any] = lambda r: getattr(r, "task_id", 0),
    ) -> tuple[list[TResult], RunMetrics]:
        """Run all tasks and return (sorted results, run metrics)."""
        if is_retryable_error is None:
            is_retryable_error = _is_transient_error

        results: list[TResult] = []
        pending: deque[tuple[TTask, int]] = deque((t, 0) for t in tasks)
        retry_heap: list[tuple[float, TTask, int]] = []
        inflight: dict[Any, TTask] = {}
        lock = threading.Lock()
        exception_during_run: BaseException | None = None
        retried_count = 0
        start_time = time.perf_counter()

        def _submit(task: TTask, retry_count: int) -> Any:
            future = executor.submit(worker, task, retry_count)
            with lock:
                inflight[future] = task
            return future

        def _process_future(future: Any) -> None:
            nonlocal exception_during_run, retried_count
            task = inflight.pop(future, None)
            if task is None:
                return
            try:
                result = future.result()
            except BaseException as exc:
                if on_worker_exception is None:
                    exception_during_run = exc
                    return
                try:
                    result = on_worker_exception(task, exc)
                except BaseException as handler_exc:
                    exception_during_run = handler_exc
                    return

            if is_failed(result):
                current_retry = retry_count_from_result(result)
                if current_retry < self.max_retries and is_retryable_error(
                    error_message(result) or ""
                ):
                    next_retry = current_retry + 1
                    retried_count += 1
                    delay = exponential_backoff_seconds(
                        next_retry,
                        initial=self.retry_delay,
                        maximum=self.retry_delay_max,
                    )
                    if on_retry_scheduled is not None:
                        try:
                            on_retry_scheduled(task, result)
                        except BaseException as exc:
                            logger.warning(f"on_retry_scheduled raised: {exc}")
                    heapq.heappush(retry_heap, (time.monotonic() + delay, task, next_retry))
                    return

            results.append(result)

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ask-llm-bounded",
        ) as executor:
            while True:
                # Move retries whose time has come back to the pending queue.
                now = time.monotonic()
                while retry_heap and retry_heap[0][0] <= now:
                    _, task, retry_count = heapq.heappop(retry_heap)
                    pending.append((task, retry_count))

                # Submit as many pending tasks as the pool allows.
                while pending and len(inflight) < self.max_workers:
                    task, retry_count = pending.popleft()
                    _submit(task, retry_count)

                if exception_during_run is not None:
                    raise exception_during_run

                if not inflight:
                    if not pending and not retry_heap:
                        break
                    # Nothing in flight; wait for the next retry to become ready.
                    if retry_heap:
                        sleep_for = max(0.0, retry_heap[0][0] - time.monotonic())
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                    continue

                # Wait until either a task finishes or the next retry is due.
                timeout: float | None = None
                if retry_heap:
                    timeout = max(0.0, retry_heap[0][0] - time.monotonic())

                done, _ = wait(
                    list(inflight.keys()),
                    return_when=FIRST_COMPLETED,
                    timeout=timeout,
                )
                for future in done:
                    _process_future(future)

                if exception_during_run is not None:
                    raise exception_during_run

        results.sort(key=order_key)
        total_time = time.perf_counter() - start_time
        metrics = RunMetrics(
            total_tasks=len(tasks),
            successful=sum(1 for r in results if not is_failed(r)),
            failed=sum(1 for r in results if is_failed(r)),
            retried=retried_count,
            total_latency=total_time,
            average_latency=total_time / len(tasks) if tasks else 0.0,
        )
        return results, metrics

    def run(
        self,
        tasks: Sequence[TTask],
        worker: Callable[[TTask, int], TResult],
        *,
        is_failed: Callable[[TResult], bool],
        error_message: Callable[[TResult], str],
        retry_count_from_result: Callable[[TResult], int],
        is_retryable_error: Callable[[str], bool] | None = None,
        on_worker_exception: Callable[[TTask, BaseException], TResult] | None = None,
        on_retry_scheduled: Callable[[TTask, TResult], None] | None = None,
        order_key: Callable[[TResult], Any] = lambda r: getattr(r, "task_id", 0),
    ) -> list[TResult]:
        """Run all tasks and return results sorted by ``order_key``."""
        results, _ = self.run_with_metrics(
            tasks,
            worker,
            is_failed=is_failed,
            error_message=error_message,
            retry_count_from_result=retry_count_from_result,
            is_retryable_error=is_retryable_error,
            on_worker_exception=on_worker_exception,
            on_retry_scheduled=on_retry_scheduled,
            order_key=order_key,
        )
        return results


def run_bounded_with_retries(
    tasks: Sequence[TTask],
    worker: Callable[[TTask, int], TResult],
    *,
    max_workers: int,
    max_retries: int,
    retry_delay: float,
    retry_delay_max: float,
    is_failed: Callable[[TResult], bool],
    error_message: Callable[[TResult], str],
    retry_count_from_result: Callable[[TResult], int],
    is_retryable_error: Callable[[str], bool] | None = None,
    on_worker_exception: Callable[[TTask, BaseException], TResult] | None = None,
    on_retry_scheduled: Callable[[TTask, TResult], None] | None = None,
    order_key: Callable[[TResult], Any] = lambda r: getattr(r, "task_id", 0),
) -> list[TResult]:
    """Convenience wrapper around :class:`BoundedRetryRunner`."""
    runner: BoundedRetryRunner[TTask, TResult] = BoundedRetryRunner(
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_delay_max=retry_delay_max,
    )
    return runner.run(
        tasks,
        worker,
        is_failed=is_failed,
        error_message=error_message,
        retry_count_from_result=retry_count_from_result,
        is_retryable_error=is_retryable_error,
        on_worker_exception=on_worker_exception,
        on_retry_scheduled=on_retry_scheduled,
        order_key=order_key,
    )


def _is_transient_error(error_message: str) -> bool:
    """Return True if *error_message* looks transient/retryable.

    Thin backward-compatible wrapper around :data:`RetryPolicy`. New code should
    construct a :class:`~ask_llm.core.retry_policy.RetryPolicy` directly to allow
    per-provider customization.
    """
    from ask_llm.core.retry_policy import DEFAULT_RETRY_POLICY

    return DEFAULT_RETRY_POLICY.is_retryable(error_message)
