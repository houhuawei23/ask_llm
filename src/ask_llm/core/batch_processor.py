"""GlobalBatchProcessor implementation (multi-provider batch execution)."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from rich.progress import Progress, TaskID

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager
    from ask_llm.config.unified_config import RateLimitConfig

from ask_llm.core.batch_models import (
    AttemptRecord,
    BatchResult,
    BatchTask,
    ModelConfig,
    TaskStatus,
    estimate_batch_task_tokens,
)
from ask_llm.core.concurrent import BoundedRetryRunner, RunMetrics
from ask_llm.core.constants import (
    DEFAULT_BATCH_FALLBACK_MODEL,
    DEFAULT_MIN_OUTPUT_TOKENS,
    OUTPUT_TOKEN_MULTIPLIERS,
    TaskKind,
)
from ask_llm.core.progress_presenter import NullProgressPresenter, ProgressPresenter
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.core.provider_manager import ProviderManager
from ask_llm.core.task_executor import TaskExecutor
from ask_llm.core.telemetry import (
    LogContext,
    bind_context,
    classify_error,
    should_fallback_for_error,
)
from ask_llm.utils.rate_limiter import get_global_rate_limiter


def estimate_output_tokens(task_kind: str, input_tokens: int) -> int:
    """
    Estimate expected output tokens based on task type and input tokens.

    Args:
        task_kind: Type of task (e.g., 'paper_explain', 'translation')
        input_tokens: Estimated input token count

    Returns:
        Estimated output token count
    """
    if input_tokens <= 0:
        return DEFAULT_MIN_OUTPUT_TOKENS

    # Try to match task_kind to TaskKind enum
    try:
        kind = TaskKind(task_kind)
        multiplier = OUTPUT_TOKEN_MULTIPLIERS.get(kind, OUTPUT_TOKEN_MULTIPLIERS[TaskKind.BATCH])
    except ValueError:
        # Unknown task kind, use default batch multiplier
        multiplier = OUTPUT_TOKEN_MULTIPLIERS[TaskKind.BATCH]

    return int(input_tokens * multiplier)


def rate_limit_config_from(config_manager: ConfigManager) -> RateLimitConfig | None:
    """Extract the rate-limit section from the config manager's unified config."""
    unified = config_manager.unified_config
    return unified.rate_limits if unified else None


class GlobalBatchProcessor:
    """Process batch tasks across multiple models concurrently."""

    def __init__(
        self,
        max_workers: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_delay_max: float = 10.0,
        verbose: bool = False,
        stream_api: bool = True,
        rate_limit_config: RateLimitConfig | None = None,
    ):
        """
        Initialize global batch processor.

        Args:
            max_workers: Maximum number of concurrent workers across all models
            max_retries: Maximum number of retries for failed tasks
            retry_delay: Initial delay between retries (exponential backoff)
            retry_delay_max: Maximum retry delay cap in seconds
            verbose: Enable verbose output with detailed API call information
            stream_api: Use streaming API calls; disable for higher batch throughput.
            rate_limit_config: Optional rate-limit configuration. When None the
                limiter falls back to ``GlobalRateLimiter.DEFAULT_LIMITS`` —
                limiting is never fully off (M7: the docstring used to claim
                ``None`` disabled rate limiting, which it does not).
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max
        self.rate_limit_config = rate_limit_config
        self._task_executor = TaskExecutor(verbose=verbose, stream_api=stream_api)
        self.last_metrics: RunMetrics | None = None

    @property
    def auth_error_logged(self) -> bool:
        # Delegates to the executor; the translators inspect this to detect
        # a batch-wide authentication failure across parallel workers.
        return self._task_executor.auth_error_logged

    def _effective_max_workers(self, tasks: list[BatchTask]) -> int:
        """Total worker slots across all (provider, model) lanes (audit 3.3).

        Sum of the per-lane caps — each lane is bounded by its own burst limit
        instead of every lane inheriting the tightest burst in the batch.
        Retained as a summary for callers/tests; scheduling uses
        :meth:`_build_lanes`.
        """
        return sum(workers for workers, _ in self._build_lanes(tasks).values()) or self.max_workers

    def _build_lanes(self, tasks: list[BatchTask]) -> dict[str, tuple[int, list[BatchTask]]]:
        """Partition tasks into per-(provider, model) lanes (audit 3.3 / M6).

        Each lane gets its own worker pool sized ``min(max_workers, burst)``
        for THAT provider/model, so a tight provider only throttles its own
        concurrency. The previous global min-burst cap let a single
        low-burst provider in a mixed batch drag every other provider down to
        its limit (and a single-provider batch to the same effective cap).
        """
        limiter = get_global_rate_limiter(self.rate_limit_config)
        grouped: dict[str, list[BatchTask]] = {}
        lane_bounds: dict[str, tuple[str, str]] = {}
        for task in tasks:
            if task.model_settings is None:
                key = "unknown:unknown"
                provider: str | None = None
                model: str | None = None
            else:
                provider = task.model_settings.provider
                model = task.model_settings.model
                key = f"{provider.lower()}:{model.lower()}"
            grouped.setdefault(key, []).append(task)
            if provider is not None and model is not None:
                lane_bounds[key] = (provider, model)

        lanes: dict[str, tuple[int, list[BatchTask]]] = {}
        for key, lane_tasks in grouped.items():
            bound = lane_bounds.get(key)
            burst = limiter.burst_for(*bound) if bound else self.max_workers
            workers = max(1, min(self.max_workers, burst))
            lanes[key] = (workers, lane_tasks)
        return lanes

    def _process_single_global_task(
        self,
        task: BatchTask,
        provider_cache: Mapping[str, LLMProviderProtocol],
        retry_count: int = 0,
        progress: Progress | None = None,
        progress_task_id: TaskID | None = None,
        input_tokens: int | None = None,
        attempt_history_by_task: dict[int, list[AttemptRecord]] | None = None,
    ) -> BatchResult:
        """Process one escalation step of a task: attempt exactly ONE config.

        Shared-budget escalation (ARCHITECTURE_REVIEW.md B1 / P1.1): the
        fallback chain and the retry budget are a *single* escalation of at most
        ``max_retries + 1`` attempts. Attempt ``retry_count`` uses
        ``configs[min(retry_count, len(configs) - 1)]`` — a transient failure
        advances to the next config (or re-tries the last one when the chain is
        shorter than the budget); a terminal failure (auth, content policy, …)
        stops at once. The :class:`BoundedRetryRunner` drives ``retry_count`` and
        the backoff heap; the worker itself never re-walks the chain, so the
        number of API calls per task is bounded by the retry budget instead of
        multiplying by the chain length.

        ``attempt_history_by_task`` threads the flat attempt records across runner
        retries (the worker is stateless between calls). May be ``None`` for an
        isolated single-step call.

        Returns:
            Batch result for this step (success, transient failure, or terminal
            failure with ``retry_count`` saturated to ``max_retries``).
        """
        if not task.model_settings:
            raise ValueError("Task must have model_settings for global batch processing")

        configs = [task.model_settings, *task.fallback_model_configs]
        # Flat attempt records (not BatchResults) — keeps the object graph acyclic
        # by construction, so no manual cycle-guard slicing is needed. See B7.
        history: list[AttemptRecord] = (
            attempt_history_by_task.setdefault(task.task_id, [])
            if attempt_history_by_task is not None
            else []
        )

        model_config = configs[min(retry_count, len(configs) - 1)]
        result = self._task_executor.try_run_with_config(
            task,
            model_config,
            provider_cache,
            retry_count,
            progress,
            progress_task_id,
            input_tokens,
        )

        history.append(AttemptRecord.from_result(result))
        # History holds the *preceding* attempts; the current step's own record
        # (always the last element) is excluded from the result's attempt_history.
        result.attempt_history = history[:-1]

        if result.status == TaskStatus.SUCCESS:
            return result

        # Terminal error: a different provider/model won't help. Stop escalating
        # by saturating retry_count so the runner declines to schedule again.
        if result.error_category and not should_fallback_for_error(result.error_category):
            bind_context(LogContext(task_id=task.task_id, phase="global_batch")).warning(
                f"Stopping escalation at attempt {retry_count + 1}: "
                f"terminal error category '{result.error_category.value}'"
            )
            result.retry_count = self.max_retries
        return result

    def process_global_tasks(
        self,
        tasks: list[BatchTask],
        config_manager: ConfigManager,
        show_progress: bool = True,
        on_result: Callable[[BatchResult], None] | None = None,
    ) -> list[BatchResult]:
        """
        Process multiple tasks across different models concurrently.

        Args:
            tasks: List of batch tasks, each with model_settings
            config_manager: Configuration manager instance
            show_progress: Whether to show progress bars
            on_result: Optional main-thread callback fired after each result,
                so callers can persist incremental checkpoint progress (D6).

        Returns:
            List of batch results
        """
        default_model = (
            tasks[0].model_settings.model
            if tasks and tasks[0].model_settings
            else DEFAULT_BATCH_FALLBACK_MODEL
        )
        # M11/M5: tokenize each task's full prompt exactly once via the shared
        # estimator. Sorting and the progress-meta build used to expand+encode
        # the whole batch twice.
        task_estimates: list[tuple[BatchTask, int]] = estimate_batch_task_tokens(
            tasks, default_model
        )
        task_estimates.sort(key=lambda pair: pair[1], reverse=True)
        pending_tasks = [task for task, _ in task_estimates]

        # Pre-build provider cache to avoid per-task adapter creation and ConfigManager mutation
        provider_cache = ProviderManager(config_manager).build_provider_cache(pending_tasks)

        # Audit 3.3 (M6): per-(provider, model) lanes. Each lane runs its own
        # bounded pool sized by its own burst limit, so a tight provider no
        # longer pins the whole batch to its concurrency ceiling. Lanes execute
        # concurrently; the longest-first ordering is preserved within a lane.
        lanes = self._build_lanes(pending_tasks)
        total_slots = sum(min(workers, len(lane_tasks)) for workers, lane_tasks in lanes.values())

        # B6 / P1.3: progress UI is a pool of per-worker bars, owned by a
        # presenter. Bars scale with the worker count, not the task count -- N=1000
        # tasks no longer render 1000 live bars. The presenter acquires a slot,
        # relabels its bar for the task, and releases it when done.
        if show_progress:
            # Pre-calculate per-task display metadata (input tokens, est. output,
            # model key) so a worker can relabel its bar instantly when it picks
            # the task up.
            task_meta: dict[int, tuple[int, int, str]] = {}
            for task, input_token_estimate in task_estimates:
                estimated_output = estimate_output_tokens(
                    task.task_kind,
                    input_token_estimate,
                )
                model_key = (
                    f"{task.model_settings.provider}/{task.model_settings.model}"
                    if task.model_settings
                    else "unknown/model"
                )
                task_meta[task.task_id] = (input_token_estimate, estimated_output, model_key)
            num_slots = max(1, min(total_slots, len(pending_tasks)))
            presenter: ProgressPresenter | NullProgressPresenter = ProgressPresenter(
                task_meta, num_slots
            )
        else:
            presenter = NullProgressPresenter()

        presenter.start()

        # Flat attempt records accumulated across runner retries (B1 / P1.1).
        # Shared across lanes; keys are disjoint per task so concurrent
        # setdefault/append from lane threads never interleave on one entry.
        attempt_history_by_task: dict[int, list[AttemptRecord]] = {}

        # Audit 3.3: cooperative interrupt. The SIGINT handler lives on the
        # main thread and sets a shared stop event; every lane runner observes
        # it, stops scheduling, drains in-flight work, and reports abandoned
        # tasks. The prior handler is restored immediately so a second Ctrl-C
        # hard-kills.
        stop_event = threading.Event()
        install_handler = threading.current_thread() is threading.main_thread()
        prev_handler = signal.getsignal(signal.SIGINT) if install_handler else None

        def _request_stop(_signum: int, _frame: Any) -> None:
            stop_event.set()
            if prev_handler is not None:
                signal.signal(signal.SIGINT, prev_handler)

        if install_handler:
            signal.signal(signal.SIGINT, _request_stop)

        def _on_retry_scheduled(task: BatchTask, failed_result: BatchResult) -> None:
            bind_context(LogContext(task_id=task.task_id, phase="global_batch")).debug(
                f"Task will be retried (attempt {failed_result.retry_count + 1}/{self.max_retries})"
            )

        def _on_worker_exception(task: BatchTask, exc: BaseException) -> BatchResult:
            error_msg = f"Unexpected error: {exc!s}"
            category = classify_error(error_msg)
            bind_context(LogContext(task_id=task.task_id, phase="global_batch")).bind(
                error_category=category.value
            ).error(f"Unexpected error processing task: {exc}")
            # Note: progress update is handled by _process_single_task/_process_single_global_task exception handler
            # This callback is for exceptions that escape before any progress is set
            return BatchResult(
                task_id=task.task_id,
                prompt=task.prompt,
                content=task.content,
                output_filename=task.output_filename,
                model_settings=task.model_settings
                or ModelConfig(provider="unknown", model="unknown"),
                status=TaskStatus.FAILED,
                error=error_msg,
                error_category=category,
            )

        def _make_interrupted_result(task: BatchTask) -> BatchResult:
            # Audit 3.2: an explicit failure record for every task the
            # interrupt abandoned, so summaries/reports/checkpoint all see
            # it (resume re-runs it via mark_all_failed_for_retry).
            return BatchResult(
                task_id=task.task_id,
                prompt=task.prompt,
                content=task.content,
                output_filename=task.output_filename,
                model_settings=task.model_settings
                or ModelConfig(provider="unknown", model="unknown"),
                status=TaskStatus.FAILED,
                error="Interrupted (Ctrl-C) before completion",
                error_category=classify_error("interrupted"),
            )

        lane_results: list[BatchResult] = []
        lane_metrics: list[RunMetrics] = []
        lane_errors: list[BaseException] = []
        lane_lock = threading.Lock()

        def _run_lane(lane_tasks: list[BatchTask], lane_workers: int) -> None:
            runner: BoundedRetryRunner[BatchTask, BatchResult] = BoundedRetryRunner(
                max_workers=lane_workers,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                retry_delay_max=self.retry_delay_max,
                stop_event=stop_event,
            )

            def _worker(task: BatchTask, retry_count: int) -> BatchResult:
                progress_task_id, input_tokens, slot_idx = presenter.acquire(task.task_id)
                try:
                    return self._process_single_global_task(
                        task,
                        provider_cache,
                        retry_count,
                        presenter.progress,
                        progress_task_id,
                        input_tokens,
                        attempt_history_by_task,
                    )
                finally:
                    presenter.release(slot_idx)

            try:
                results, metrics = runner.run_with_metrics(
                    lane_tasks,
                    _worker,
                    is_failed=lambda r: r.status == TaskStatus.FAILED,
                    error_message=lambda r: r.error or "",
                    retry_count_from_result=lambda r: r.retry_count,
                    is_throttled=lambda r: r.throttled,
                    on_worker_exception=_on_worker_exception,
                    on_retry_scheduled=_on_retry_scheduled,
                    on_result=on_result,
                    order_key=lambda r: r.task_id,
                    make_interrupted_result=_make_interrupted_result,
                )
            except BaseException as exc:
                with lane_lock:
                    lane_errors.append(exc)
                return
            with lane_lock:
                lane_results.extend(results)
                lane_metrics.append(metrics)

        try:
            if len(lanes) == 1:
                # Single lane: run inline (no extra thread, SIGINT via the
                # runner's own main-thread handler is irrelevant — the shared
                # stop_event path above already covers it).
                _workers, only_tasks = next(iter(lanes.values()))
                _run_lane(only_tasks, _workers)
            else:
                lane_threads = [
                    threading.Thread(
                        target=_run_lane,
                        args=(lane_tasks, workers),
                        name=f"ask-llm-lane-{key}",
                        daemon=False,
                    )
                    for key, (workers, lane_tasks) in lanes.items()
                ]
                for thread in lane_threads:
                    thread.start()
                for thread in lane_threads:
                    thread.join()
        finally:
            if install_handler and prev_handler is not None:
                signal.signal(signal.SIGINT, prev_handler)
            presenter.stop()

        if lane_errors:
            raise lane_errors[0]

        self.last_metrics = RunMetrics(
            total_tasks=len(pending_tasks),
            successful=sum(1 for r in lane_results if r.status != TaskStatus.FAILED),
            failed=sum(1 for r in lane_results if r.status == TaskStatus.FAILED),
            retried=sum(m.retried for m in lane_metrics),
            total_latency=max((m.total_latency for m in lane_metrics), default=0.0),
            interrupted=any(m.interrupted for m in lane_metrics),
            abandoned=sum(m.abandoned for m in lane_metrics),
        )

        lane_results.sort(key=lambda r: r.task_id)
        return lane_results
