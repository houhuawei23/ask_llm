"""GlobalBatchProcessor implementation (multi-provider batch execution)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from rich.progress import Progress, TaskID

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager
    from ask_llm.config.unified_config import RateLimitConfig

from ask_llm.core.batch_models import (
    AttemptRecord,
    BatchResult,
    BatchStatistics,
    BatchTask,
    ModelConfig,
    TaskStatus,
    sort_batch_tasks_by_estimated_input,
)
from ask_llm.core.concurrent import BoundedRetryRunner, RunMetrics
from ask_llm.core.constants import (
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
from ask_llm.utils.token_counter import TokenCounter


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


def calculate_statistics_by_model(results: list[BatchResult]) -> dict[str, BatchStatistics]:
    """Calculate per-model statistics from batch results.

    Thin delegate to :meth:`BatchStatistics.from_results` (single source of
    truth). Kept as a module-level function for import compatibility.
    """
    return BatchStatistics.from_results(results)



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
            rate_limit_config: Optional rate-limit configuration. If None, rate limiting is disabled.
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max
        self.rate_limit_config = rate_limit_config
        self._task_executor = TaskExecutor(verbose=verbose, stream_api=stream_api)
        self.last_metrics: RunMetrics | None = None

    @property
    def _auth_error_logged(self) -> bool:
        # Delegates to the executor; translation_service inspects this to detect
        # a batch-wide authentication failure across parallel workers.
        return self._task_executor._auth_error_logged

    def _effective_max_workers(self, tasks: list[BatchTask]) -> int:
        """Return ``max_workers`` capped by the tightest burst limit among tasks."""
        limiter = get_global_rate_limiter(self.rate_limit_config)
        min_burst: int | None = None
        for task in tasks:
            if task.model_settings is None:
                continue
            burst = limiter.burst_for(task.model_settings.provider, task.model_settings.model)
            if min_burst is None or burst < min_burst:
                min_burst = burst
        if min_burst is None:
            return self.max_workers
        return max(1, min(self.max_workers, min_burst))

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
    ) -> list[BatchResult]:
        """
        Process multiple tasks across different models concurrently.

        Args:
            tasks: List of batch tasks, each with model_settings
            config_manager: Configuration manager instance
            show_progress: Whether to show progress bars

        Returns:
            List of batch results
        """
        default_model = (
            tasks[0].model_settings.model
            if tasks and tasks[0].model_settings
            else "gpt-3.5-turbo"
        )
        pending_tasks = sort_batch_tasks_by_estimated_input(tasks.copy(), default_model)

        # Pre-build provider cache to avoid per-task adapter creation and ConfigManager mutation
        provider_cache = ProviderManager(config_manager).build_provider_cache(pending_tasks)

        # Cap the thread pool size to the smallest configured burst limit so that
        # workers do not sit blocked on the rate limiter waiting for tokens.
        effective_max_workers = self._effective_max_workers(pending_tasks)

        # B6 / P1.3: progress UI is a pool of per-worker bars, owned by a
        # presenter. Bars scale with the worker count, not the task count -- N=1000
        # tasks no longer render 1000 live bars. The presenter acquires a slot,
        # relabels its bar for the task, and releases it when done.
        if show_progress:
            # Pre-calculate per-task display metadata (input tokens, est. output,
            # model key) so a worker can relabel its bar instantly when it picks
            # the task up.
            task_meta: dict[int, tuple[int, int, str]] = {}
            for task in pending_tasks:
                estimated_prompt = (
                    task.prompt.replace("{content}", task.content)
                    if "{content}" in task.prompt
                    else f"{task.prompt}\n\n{task.content}"
                )
                input_token_estimate = TokenCounter.estimate_tokens(
                    estimated_prompt,
                    task.model_settings.model if task.model_settings else "gpt-3.5-turbo",
                )["token_count"]
                estimated_output = estimate_output_tokens(
                    task.task_kind if hasattr(task, "task_kind") else "translation",
                    input_token_estimate,
                )
                model_key = (
                    f"{task.model_settings.provider}/{task.model_settings.model}"
                    if task.model_settings
                    else "unknown/model"
                )
                task_meta[task.task_id] = (input_token_estimate, estimated_output, model_key)
            num_slots = max(1, min(effective_max_workers, len(pending_tasks)))
            presenter: ProgressPresenter | NullProgressPresenter = ProgressPresenter(
                task_meta, num_slots
            )
        else:
            presenter = NullProgressPresenter()

        presenter.start()

        # Flat attempt records accumulated across runner retries (B1 / P1.1). The
        # worker is stateless between calls; this side channel threads history so
        # the final result's attempt_history captures every preceding attempt.
        attempt_history_by_task: dict[int, list[AttemptRecord]] = {}

        try:

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

            def _on_retry_scheduled(task: BatchTask, failed_result: BatchResult) -> None:
                bind_context(LogContext(task_id=task.task_id, phase="global_batch")).debug(
                    f"Task will be retried "
                    f"(attempt {failed_result.retry_count + 1}/{self.max_retries})"
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

            runner: BoundedRetryRunner[BatchTask, BatchResult] = BoundedRetryRunner(
                max_workers=effective_max_workers,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                retry_delay_max=self.retry_delay_max,
            )
            results, self.last_metrics = runner.run_with_metrics(
                pending_tasks,
                _worker,
                is_failed=lambda r: r.status == TaskStatus.FAILED,
                error_message=lambda r: r.error or "",
                retry_count_from_result=lambda r: r.retry_count,
                on_worker_exception=_on_worker_exception,
                on_retry_scheduled=_on_retry_scheduled,
                order_key=lambda r: r.task_id,
            )

        finally:
            presenter.stop()

        return results

    def calculate_statistics(self, results: list[BatchResult]) -> dict[str, BatchStatistics]:
        """Calculate statistics from batch results, grouped by model.

        Thin delegate to the module-level :func:`calculate_statistics_by_model`.
        """
        return calculate_statistics_by_model(results)
