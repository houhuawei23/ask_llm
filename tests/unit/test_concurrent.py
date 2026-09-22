"""Tests for the bounded single-queue retry runner."""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass

import pytest

from ask_llm.core.concurrent import (
    BoundedRetryRunner,
    RunMetrics,
    exponential_backoff_seconds,
    run_bounded_with_retries,
)


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


def test_on_result_fires_per_result_in_order():
    """D6: on_result fires once per appended result, on the main thread, in
    completion order — so callers can persist incremental checkpoint progress."""
    seen: list[int] = []
    runner = BoundedRetryRunner(
        max_workers=2,
        max_retries=0,
        retry_delay=0.01,
        retry_delay_max=0.1,
    )
    results, _ = runner.run_with_metrics(
        list(range(6)),
        lambda t, rc: _SimpleResult(task_id=t, value=t, retry_count=rc),
        is_failed=lambda r: False,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        on_result=lambda r: seen.append(r.task_id),
        order_key=lambda r: r.task_id,
    )
    assert len(results) == 6
    assert len(seen) == 6
    # Callback saw every task exactly once.
    assert sorted(seen) == list(range(6))


def test_on_result_exception_does_not_break_run():
    """A buggy on_result callback must not abort the run (logged + skipped)."""
    calls: list[int] = []
    runner = BoundedRetryRunner(
        max_workers=1,
        max_retries=0,
        retry_delay=0.01,
        retry_delay_max=0.1,
    )
    results, _ = runner.run_with_metrics(
        list(range(3)),
        lambda t, rc: _SimpleResult(task_id=t, value=t, retry_count=rc),
        is_failed=lambda r: False,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        on_result=lambda r: (
            calls.append(r.task_id) if r.task_id != 1 else (_ for _ in ()).throw(ValueError("boom"))
        ),
        order_key=lambda r: r.task_id,
    )
    assert len(results) == 3  # run completed despite the callback raising


@dataclass
class _UnorderableTask:
    """Plain dataclass: supports == but not <, like pydantic BatchTask."""

    task_id: int


def test_retry_heap_tolerates_identical_due_times(monkeypatch):
    """H8 regression: retry heap entries used to be ``(due, task, retry)``;
    with identical due timestamps (coarse clocks — Windows monotonic has
    ~15ms granularity — or identical backoff) tuple comparison fell through
    to the task object, and unorderable tasks (pydantic models, dataclasses)
    raised ``TypeError``, killing the whole run.

    A frozen-then-jump fake clock pins every due timestamp to the exact same
    float (real monotonic() + (base - real) drifts by an ulp and silently
    de-ties), while keeping the retries due only after the collision window.
    Tasks 0/1 fail once together (barrier); task 3 sleeps long enough to keep
    a future in flight across the failure window, so the runner cannot drain
    the heap via its idle sleep path between the two pushes.
    """
    real_monotonic = time.monotonic  # captured before patching the module attr
    clock_start = real_monotonic()

    def frozen_then_jump() -> float:
        # Frozen while the collision must happen, then jumps past the due
        # time so the frozen retries get picked up instead of sleeping ~10s.
        if real_monotonic() - clock_start < 0.2:
            return 1000.0
        return 1200.0

    monkeypatch.setattr("ask_llm.core.concurrent.time.monotonic", frozen_then_jump)
    barrier = threading.Barrier(2)
    attempts: dict[int, int] = {}

    def worker(task: _UnorderableTask, retry_count: int) -> _SimpleResult:
        tid = task.task_id
        attempts[tid] = attempts.get(tid, 0) + 1
        if tid in (0, 1) and attempts[tid] == 1:
            barrier.wait(timeout=5)
            time.sleep(0.02)
            return _SimpleResult(task_id=tid, value=-1, retry_count=retry_count, error="rate limit")
        if tid == 3 and attempts[tid] == 1:
            time.sleep(0.1)
        return _SimpleResult(task_id=tid, value=tid * 10, retry_count=retry_count)

    results = run_bounded_with_retries(
        [_UnorderableTask(i) for i in range(4)],
        worker,
        max_workers=4,
        max_retries=3,
        retry_delay=0.01,
        retry_delay_max=0.01,
        is_failed=lambda r: r.value == -1,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    assert sorted(r.value for r in results) == [0, 10, 20, 30]
    assert attempts == {0: 2, 1: 2, 2: 1, 3: 1}


class TestAudit31Jitter:
    """Audit 3.1: full-jitter backoff, bounded and seedable."""

    def test_jitter_bounded_and_seeded(self):
        import random as _random

        rng = _random.Random(42)
        delays = [
            exponential_backoff_seconds(n, initial=1.0, maximum=8.0, rng=rng) for n in range(1, 8)
        ]
        for d in delays:
            assert 0.0 <= d <= 8.0
        # Deterministic with the same seed.
        rng2 = _random.Random(42)
        again = [
            exponential_backoff_seconds(n, initial=1.0, maximum=8.0, rng=rng2) for n in range(1, 8)
        ]
        assert delays == again
        # Attempt 1 raw delay is 1.0; jitter lives in [0, 1].
        rng3 = _random.Random(7)
        first = [
            exponential_backoff_seconds(1, initial=1.0, maximum=8.0, rng=rng3) for _ in range(20)
        ]
        assert all(0.0 <= d <= 1.0 for d in first)

    def test_retry_delays_differ_between_workers(self):
        """Two simultaneous failures must not reschedule at the same instant."""
        import random as _random

        rng = _random.Random(123)
        d1 = exponential_backoff_seconds(1, initial=2.0, maximum=10.0, rng=rng)
        d2 = exponential_backoff_seconds(1, initial=2.0, maximum=10.0, rng=rng)
        assert d1 != d2


class TestAudit32InterruptedDrain:
    """Audit 3.2: abandoned tasks appear as explicit results."""

    def test_interrupt_reports_queued_retries(self):
        """Tasks left pending/queued at interrupt get fabricated results via
        make_interrupted_result and count in metrics."""
        if threading.current_thread() is not threading.main_thread():
            pytest.skip("SIGINT graceful-drain requires the main thread")

        triggered = threading.Event()

        def worker(task: int, retry_count: int) -> _SimpleResult:
            if task >= 2 and not triggered.is_set():
                triggered.set()
                os.kill(os.getpid(), signal.SIGINT)
            return _SimpleResult(task_id=task, value=task, retry_count=retry_count)

        def make_interrupted(task: int) -> _SimpleResult:
            return _SimpleResult(task_id=task, value=-1, retry_count=0, error="interrupted")

        runner = BoundedRetryRunner(
            max_workers=1,
            max_retries=0,
            retry_delay=0.01,
            retry_delay_max=0.05,
        )
        results, metrics = runner.run_with_metrics(
            list(range(10)),
            worker,
            is_failed=lambda r: bool(r.error),
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.task_id,
            make_interrupted_result=make_interrupted,
        )

        assert metrics.interrupted is True
        seen_ids = {r.task_id for r in results}
        # Every task accounted for: no silent drops (audit 3.2 core claim).
        assert seen_ids == set(range(10))
        assert metrics.successful + metrics.failed == metrics.total_tasks
        assert metrics.abandoned > 0  # abandoned tasks are always reported


class TestAudit33ThrottleAndStopEvent:
    """Audit 3.3: tail-requeue for throttled results; cooperative stop_event."""

    @staticmethod
    def _runner(stop_event=None, max_retries=2, max_workers=2):
        return BoundedRetryRunner(
            max_workers=max_workers,
            max_retries=max_retries,
            retry_delay=0.01,
            retry_delay_max=0.05,
            stop_event=stop_event,
        )

    def test_throttled_result_tail_requeued_without_retry_cost(self):
        """A throttled failure requeues at the tail; the retry budget is untouched."""
        calls: dict[int, int] = {}

        def worker(task: int, retry_count: int) -> _SimpleResult:
            calls[task] = calls.get(task, 0) + 1
            if calls[task] <= 2 and task == 1:
                # Non-retryable message: only the throttle path may requeue it.
                return _SimpleResult(
                    task_id=task, value=-1, retry_count=retry_count, error="auth failed"
                )
            return _SimpleResult(task_id=task, value=task * 10, retry_count=retry_count)

        results, metrics = self._runner().run_with_metrics(
            [1],
            worker,
            is_failed=lambda r: r.value == -1,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            is_throttled=lambda r: r.error == "auth failed",
            order_key=lambda r: r.task_id,
        )

        assert calls[1] == 3  # two throttle deferrals, then success
        assert metrics.retried == 0  # retry budget untouched
        assert len(results) == 1
        assert results[0].value == 10

    def test_throttled_deferrals_bounded_then_terminal(self):
        """A permanently throttled task terminates after max_retries+1 deferrals."""
        calls: dict[int, int] = []

        def worker(task: int, retry_count: int) -> _SimpleResult:
            calls.append(1)
            return _SimpleResult(task_id=task, value=-1, retry_count=retry_count, error="throttled")

        results, metrics = self._runner(max_retries=1).run_with_metrics(
            [1],
            worker,
            is_failed=lambda r: r.value == -1,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            is_throttled=lambda r: r.error == "throttled",
            order_key=lambda r: r.task_id,
        )

        # Deferrals: used=0,1 requeued (<= max_retries); then the normal
        # escalation path runs once (retry_count 1), and after that the task
        # terminates — total visits bounded by 2 * (max_retries + 1).
        assert len(calls) == 4
        assert len(results) == 1
        assert results[0].value == -1
        assert metrics.failed == 1

    def test_throttled_task_keeps_retry_budget_for_real_failures(self):
        """After a throttle deferral, a genuine transient error still retries."""
        calls: list[int] = []

        def worker(task: int, retry_count: int) -> _SimpleResult:
            calls.append(retry_count)
            if len(calls) == 1:
                return _SimpleResult(
                    task_id=task, value=-1, retry_count=retry_count, error="throttled"
                )
            if len(calls) == 2:
                return _SimpleResult(
                    task_id=task, value=-1, retry_count=retry_count, error="connection reset"
                )
            return _SimpleResult(task_id=task, value=task * 10, retry_count=retry_count)

        results, metrics = self._runner().run_with_metrics(
            [1],
            worker,
            is_failed=lambda r: r.value == -1,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            is_throttled=lambda r: r.error == "throttled",
            order_key=lambda r: r.task_id,
        )

        assert results[0].value == 10
        assert metrics.retried == 1  # only the genuine failure consumed budget

    def test_stop_event_interrupts_runner(self):
        """A shared stop_event (lane threads) drains like a SIGINT interrupt."""
        if threading.current_thread() is not threading.main_thread():
            pytest.skip("timing-sensitive drain check kept on the main thread")

        stop_event = threading.Event()

        def worker(task: int, retry_count: int) -> _SimpleResult:
            if task == 0:
                stop_event.set()  # cooperative stop while task 0 runs
            return _SimpleResult(task_id=task, value=task, retry_count=retry_count)

        def make_interrupted(task: int) -> _SimpleResult:
            return _SimpleResult(task_id=task, value=-1, retry_count=0, error="interrupted")

        # max_workers=1 keeps task 1 pending until task 0 observes the stop.
        runner = self._runner(stop_event=stop_event, max_workers=1)
        results, metrics = runner.run_with_metrics(
            [0, 1],
            worker,
            is_failed=lambda r: bool(r.error),
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.task_id,
            make_interrupted_result=make_interrupted,
        )

        assert metrics.interrupted is True
        seen_ids = {r.task_id for r in results}
        assert seen_ids == {0, 1}  # abandoned task 1 explicitly reported
        assert metrics.abandoned == 1
