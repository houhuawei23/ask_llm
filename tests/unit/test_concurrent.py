"""Tests for the bounded single-queue retry runner."""

from __future__ import annotations

import os
import signal
import threading
from dataclasses import dataclass

import pytest

from ask_llm.core.concurrent import BoundedRetryRunner, RunMetrics, run_bounded_with_retries


@dataclass
class _SimpleResult:
    task_id: int
    value: int
    retry_count: int
    error: str = ""


def _make_worker(failures: dict[int, int]) -> callable:
    """Return a worker that fails a fixed number of times per task_id."""
    attempts: dict[int, int] = {}

    def worker(task: int, retry_count: int) -> _SimpleResult:
        attempts[task] = attempts.get(task, 0) + 1
        if attempts[task] <= failures.get(task, 0):
            return _SimpleResult(
                task_id=task, value=-1, retry_count=retry_count, error="rate limit"
            )
        return _SimpleResult(task_id=task, value=task * 10, retry_count=retry_count)

    return worker


def test_runs_all_tasks_in_order():
    def worker(task: int, retry_count: int) -> _SimpleResult:
        return _SimpleResult(task_id=task, value=task * 10, retry_count=retry_count)

    results = run_bounded_with_retries(
        list(range(10)),
        worker,
        max_workers=3,
        max_retries=0,
        retry_delay=0.01,
        retry_delay_max=0.1,
        is_failed=lambda r: False,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    assert [r.value for r in results] == [i * 10 for i in range(10)]


def test_retries_transient_failures():
    failures = {2: 2, 5: 1}
    results = run_bounded_with_retries(
        list(range(7)),
        _make_worker(failures),
        max_workers=2,
        max_retries=3,
        retry_delay=0.01,
        retry_delay_max=0.1,
        is_failed=lambda r: r.value == -1,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    assert all(r.value != -1 for r in results)
    assert [r.value for r in results] == [i * 10 for i in range(7)]


def test_exhausted_retries_return_failed_result():
    failures = {1: 10}
    results = run_bounded_with_retries(
        list(range(3)),
        _make_worker(failures),
        max_workers=2,
        max_retries=2,
        retry_delay=0.01,
        retry_delay_max=0.1,
        is_failed=lambda r: r.value == -1,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    failed = [r for r in results if r.value == -1]
    successful = [r for r in results if r.value != -1]
    assert len(failed) == 1
    assert failed[0].retry_count == 2
    assert len(successful) == 2


def test_on_worker_exception_caught():
    def worker(task: int, retry_count: int) -> _SimpleResult:
        if task == 3:
            raise RuntimeError("boom")
        return _SimpleResult(task_id=task, value=task, retry_count=retry_count)

    exceptions: list[tuple[int, BaseException]] = []

    def on_exception(task: int, exc: BaseException) -> _SimpleResult:
        exceptions.append((task, exc))
        return _SimpleResult(task_id=task, value=-1, retry_count=0, error="handled")

    results = run_bounded_with_retries(
        list(range(5)),
        worker,
        max_workers=2,
        max_retries=0,
        retry_delay=0.01,
        retry_delay_max=0.1,
        is_failed=lambda r: r.value == -1,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        on_worker_exception=on_exception,
        order_key=lambda r: r.task_id,
    )
    assert len(exceptions) == 1
    assert exceptions[0][0] == 3
    assert len(results) == 5


def test_propagates_exception_without_handler():
    def worker(task: int, retry_count: int) -> _SimpleResult:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_bounded_with_retries(
            [1],
            worker,
            max_workers=1,
            max_retries=0,
            retry_delay=0.01,
            retry_delay_max=0.1,
            is_failed=lambda r: False,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
        )


def test_runner_class_reusable():
    runner = BoundedRetryRunner(
        max_workers=2,
        max_retries=2,
        retry_delay=0.01,
        retry_delay_max=0.1,
    )

    def worker(task: int, retry_count: int) -> _SimpleResult:
        return _SimpleResult(task_id=task, value=task, retry_count=retry_count)

    results1 = runner.run(
        [1, 2],
        worker,
        is_failed=lambda r: False,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    results2 = runner.run(
        [3, 4],
        worker,
        is_failed=lambda r: False,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    assert [r.value for r in results1] == [1, 2]
    assert [r.value for r in results2] == [3, 4]


def test_run_with_metrics_counts_retries_and_failures():
    failures = {1: 2, 3: 5}
    runner = BoundedRetryRunner(
        max_workers=2,
        max_retries=2,
        retry_delay=0.01,
        retry_delay_max=0.1,
    )

    _results, metrics = runner.run_with_metrics(
        list(range(4)),
        _make_worker(failures),
        is_failed=lambda r: r.value == -1,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )

    assert isinstance(metrics, RunMetrics)
    assert metrics.total_tasks == 4
    assert metrics.successful == 3
    assert metrics.failed == 1
    assert metrics.retried == 4
    assert metrics.total_latency >= 0
    assert metrics.average_latency >= 0


def test_sigint_returns_partial_results_and_drains_inflight():
    """B5: Ctrl-C stops scheduling new work, drains in-flight, returns partials.

    The runner must NOT re-raise KeyboardInterrupt; it returns whatever
    completed so the caller can persist a checkpoint and resume. A second
    Ctrl-C (handler restored) would hard-interrupt.
    """
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("SIGINT graceful-drain requires the main thread")

    completed: list[int] = []
    triggered = threading.Event()
    stop_after = 4

    def worker(task: int, retry_count: int) -> _SimpleResult:
        completed.append(task)
        # Once a few tasks finish, simulate Ctrl-C delivered from a worker.
        if len(completed) >= stop_after and not triggered.is_set():
            triggered.set()
            os.kill(os.getpid(), signal.SIGINT)
        return _SimpleResult(task_id=task, value=task, retry_count=retry_count)

    runner = BoundedRetryRunner(
        max_workers=2,
        max_retries=0,
        retry_delay=0.01,
        retry_delay_max=0.1,
    )
    results, metrics = runner.run_with_metrics(
        list(range(20)),
        worker,
        is_failed=lambda r: False,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )

    assert metrics.interrupted is True
    # Some tasks completed (before interrupt + drained in-flight), not all 20.
    assert 1 <= len(results) < 20
    # Every returned result is intact (drained, not lost).
    assert all(r.value == r.task_id for r in results)


def test_normal_run_not_marked_interrupted():
    """A run that completes without Ctrl-C must report interrupted=False."""
    runner = BoundedRetryRunner(
        max_workers=2,
        max_retries=0,
        retry_delay=0.01,
        retry_delay_max=0.1,
    )
    _results, metrics = runner.run_with_metrics(
        list(range(5)),
        lambda t, rc: _SimpleResult(task_id=t, value=t, retry_count=rc),
        is_failed=lambda r: False,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    assert metrics.interrupted is False
