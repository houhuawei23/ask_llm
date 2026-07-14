"""GlobalBatchProcessor implementation (multi-provider batch execution)."""

from __future__ import annotations

import queue as _queue
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING

from rich.progress import Progress, TaskID

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager
    from ask_llm.config.unified_config import RateLimitConfig

from ask_llm.config.context import get_config_or_none
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
from ask_llm.core.models import RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.core.provider_manager import ProviderManager
from ask_llm.core.stream_collector import stream_and_collect
from ask_llm.core.telemetry import (
    ErrorCategory,
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


def paper_request_timeout_seconds() -> float:
    """Paper explain HTTP timeout from unified config, default 600s.

    Module-level so both provider-cache construction and task scheduling can share
    it without coupling to the processor instance.
    """
    lr = get_config_or_none()
    if lr is None:
        return 600.0
    return float(lr.unified_config.paper.request_timeout_seconds)


def update_global_task_progress_failed(
    progress: Progress | None,
    progress_task_id: TaskID | None,
    model_key: str,
    task_id: int,
    progress_tokens: str,
) -> None:
    """Mark a global task as failed in the progress bar (no-op if no progress)."""
    if progress and progress_task_id is not None:
        progress.update(
            progress_task_id,
            description=(f"{model_key} Task {task_id} ({progress_tokens} tokens): ✗ Failed"),
            completed=100,
        )


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
        self.verbose = verbose
        self.stream_api = stream_api
        self.rate_limit_config = rate_limit_config
        self._auth_error_lock = threading.Lock()
        self._auth_error_logged = False
        self.last_metrics: RunMetrics | None = None

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

    def _log_global_task_failure(
        self,
        task_id: int,
        model_key: str,
        error_msg: str,
        category: ErrorCategory | None = None,
    ) -> None:
        """Log task failure; collapse duplicate authentication errors from parallel workers."""
        if category is None:
            category = classify_error(error_msg)
        ctx = LogContext(task_id=task_id, phase="global_batch")
        bound = bind_context(ctx).bind(model_key=model_key, error_category=category.value)
        if category == ErrorCategory.AUTHENTICATION:
            with self._auth_error_lock:
                if not self._auth_error_logged:
                    self._auth_error_logged = True
                    bound.error(
                        f"API authentication failed ({model_key}): {error_msg}\n"
                        "(Further parallel tasks with the same auth error are logged at DEBUG only.)"
                    )
                else:
                    bound.debug(f"Task failed (auth): {error_msg}")
        else:
            bound.error(f"Task failed ({model_key}): {error_msg}")

    def _run_paper_explain_global_task(
        self,
        task: BatchTask,
        model_config: ModelConfig,
        model_key: str,
        processor: RequestProcessor,
        provider: LLMProviderProtocol,
        paper_timeout: float | None,
        progress: Progress | None,
        progress_task_id: TaskID | None,
        input_tokens: int | None,
        result: BatchResult,
    ) -> tuple[BatchResult, str]:
        """Paper explain: streaming completion + live output token estimate (same UX as trans)."""
        input_stats = TokenCounter.estimate_tokens(task.prompt.strip(), model_config.model)
        input_token_count = input_stats["token_count"]
        display_input_tokens = input_tokens if input_tokens is not None else input_token_count
        progress_tokens = f"paper≈{display_input_tokens} tok"

        if progress and progress_task_id is not None:
            progress.update(
                progress_task_id,
                description=(
                    f"{model_key} paper task {task.task_id} ({progress_tokens} tokens): 0 out"
                ),
            )

        ctx = LogContext(
            task_id=task.task_id,
            provider=model_config.provider,
            model=model_config.model,
            attempt=result.retry_count + 1,
            phase="global_batch",
        )

        if self.verbose:
            api_params = [
                f"provider={model_config.provider}",
                f"model={model_config.model}",
            ]
            if model_config.temperature is not None:
                api_params.append(f"temperature={model_config.temperature}")
            if model_config.max_tokens is not None:
                api_params.append(f"max_tokens={model_config.max_tokens}")
            api_params.append(f"input_tokens={display_input_tokens}")
            api_params.append(f"timeout={paper_timeout}s")
            bind_context(ctx).info(f"Paper API: {', '.join(api_params)}")

        stream_iter = processor.iter_process_raw_stream(
            task.prompt,
            temperature=model_config.temperature,
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            return_reasoning=task.return_reasoning,
        )
        description_prefix = f"{model_key} paper task {task.task_id} ({progress_tokens} tokens)"
        response, reasoning_out, _output_token_count, latency = stream_and_collect(
            stream_iter,
            model_config,
            progress,
            progress_task_id,
            description_prefix,
            return_reasoning=task.return_reasoning,
        )

        out_for_count = response
        if reasoning_out:
            out_for_count = f"{reasoning_out}\n{response}"
        output_stats = TokenCounter.estimate_tokens(out_for_count, model_config.model)

        metadata = RequestMetadata.from_execution(
            provider_name=provider.name,
            model=model_config.model,
            temperature=model_config.temperature,
            default_temperature=provider.config.api_temperature,
            input_stats=input_stats,
            output_words=output_stats["word_count"],
            output_tokens=output_stats["token_count"],
            latency=latency,
        )

        result.response = response
        result.reasoning = reasoning_out
        result.metadata = metadata
        result.status = TaskStatus.SUCCESS

        if progress and progress_task_id is not None:
            ot = metadata.output_tokens
            progress.update(
                progress_task_id,
                description=(f"{model_key} paper task {task.task_id}: ✓ Complete ({ot} out)"),
                completed=100,
            )

        bind_context(ctx).debug("Paper job completed")
        return result, progress_tokens

    def _run_translation_chunk_global_task(
        self,
        task: BatchTask,
        model_config: ModelConfig,
        model_key: str,
        processor: RequestProcessor,
        provider: LLMProviderProtocol,
        progress: Progress | None,
        progress_task_id: TaskID | None,
        input_tokens: int | None,
        result: BatchResult,
        body_tokens: int,
    ) -> tuple[BatchResult, str]:
        """Translation chunk: format prompt + stream via RequestProcessor.process."""
        full_prompt = processor._format_prompt(task.content, task.prompt)

        input_stats = TokenCounter.estimate_tokens(full_prompt, model_config.model)
        input_token_count = input_stats["token_count"]
        display_input_tokens = input_tokens if input_tokens is not None else input_token_count
        progress_tokens = f"body≈{body_tokens} input≈{display_input_tokens}"

        ctx = LogContext(
            task_id=task.task_id,
            provider=model_config.provider,
            model=model_config.model,
            attempt=result.retry_count + 1,
            phase="global_batch",
        )

        if self.verbose:
            api_params = [
                f"provider={model_config.provider}",
                f"model={model_config.model}",
            ]
            if model_config.temperature is not None:
                api_params.append(f"temperature={model_config.temperature}")
            if model_config.max_tokens is not None:
                api_params.append(f"max_tokens={model_config.max_tokens}")
            if model_config.top_p is not None:
                api_params.append(f"top_p={model_config.top_p}")
            api_params.append(f"input_tokens={display_input_tokens}")
            bind_context(ctx).info(f"API Call: {', '.join(api_params)}")

        if progress and progress_task_id is not None:
            progress.update(
                progress_task_id,
                description=f"{model_key} Task {task.task_id} ({progress_tokens} tokens): Processing...",
            )
        elif progress is None:
            bind_context(ctx).info(f"Translation chunk start ({model_key}, {progress_tokens})")

        stream_iter = processor.process(
            content=task.content,
            prompt_template=task.prompt,
            temperature=model_config.temperature,
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            stream=self.stream_api,
        )
        description_prefix = f"{model_key} Task {task.task_id} ({progress_tokens} tokens)"
        response, _, output_token_count, latency = stream_and_collect(
            stream_iter,
            model_config,
            progress,
            progress_task_id,
            description_prefix,
            return_reasoning=False,
        )

        metadata = RequestMetadata.from_execution(
            provider_name=provider.name,
            model=model_config.model,
            temperature=model_config.temperature,
            default_temperature=provider.config.api_temperature,
            input_stats=input_stats,
            output_words=TokenCounter.count_words(response),
            output_tokens=output_token_count,
            latency=latency,
        )

        result.response = response
        result.metadata = metadata
        result.status = TaskStatus.SUCCESS

        if self.verbose:
            bind_context(ctx).info(
                f"API Call Completed: "
                f"output_tokens={output_token_count}, "
                f"latency={latency:.2f}s, "
                f"status=success"
            )

        if progress and progress_task_id is not None:
            progress.update(
                progress_task_id,
                description=f"{model_key} Task {task.task_id} ({progress_tokens} tokens): ✓ Complete ({output_token_count} out)",
                completed=100,
            )
        elif progress is None:
            bind_context(ctx).info(
                f"Translation chunk complete ({output_token_count} output tokens)"
            )

        bind_context(ctx).debug("Task completed successfully")
        return result, progress_tokens

    def _try_run_with_config(
        self,
        task: BatchTask,
        model_config: ModelConfig,
        provider_cache: Mapping[str, LLMProviderProtocol],
        retry_count: int,
        progress: Progress | None,
        progress_task_id: TaskID | None,
        input_tokens: int | None,
    ) -> BatchResult:
        """Attempt to process a task with a single provider/model config.

        Returns a ``BatchResult`` (success or failure) without raising.
        """
        model_key = f"{model_config.provider}/{model_config.model}"
        display_input_tokens = input_tokens if input_tokens is not None else 0

        result = BatchResult(
            task_id=task.task_id,
            prompt=task.prompt,
            content=task.content,
            output_filename=task.output_filename,
            model_settings=model_config,
            status=TaskStatus.PROCESSING,
            retry_count=retry_count,
        )
        ctx = LogContext(
            task_id=task.task_id,
            provider=model_config.provider,
            model=model_config.model,
            attempt=retry_count + 1,
            phase="global_batch",
        )
        body_tokens = TokenCounter.count_tokens(task.content, model_config.model)
        progress_tokens = f"body≈{body_tokens} input≈{display_input_tokens}"

        try:
            limiter = get_global_rate_limiter()
            acquire_timeout = limiter.acquire_timeout(
                model_config.provider, model_config.model
            )
            acquired = limiter.acquire(
                model_config.provider,
                model_config.model,
                timeout=acquire_timeout,
            )
            if not acquired:
                raise RuntimeError(
                    f"Rate limit timeout for {model_config.provider}/{model_config.model} "
                    f"after {acquire_timeout:.0f}s (configure "
                    f"rate_limits.<provider>.acquire_timeout_seconds to raise it)"
                )

            paper_timeout: float | None = None
            if task.task_kind == "paper_explain":
                paper_timeout = paper_request_timeout_seconds()

            cache_key = model_key
            if task.task_kind == "paper_explain":
                cache_key += f" / timeout={paper_timeout}"
            provider = provider_cache.get(cache_key)
            if provider is None:
                raise ValueError(f"Provider not found in cache for {cache_key}")
            processor = RequestProcessor(provider)

            if task.task_kind == "paper_explain":
                result, _ = self._run_paper_explain_global_task(
                    task,
                    model_config,
                    model_key,
                    processor,
                    provider,
                    paper_timeout,
                    progress,
                    progress_task_id,
                    input_tokens,
                    result,
                )
            else:
                result, _ = self._run_translation_chunk_global_task(
                    task,
                    model_config,
                    model_key,
                    processor,
                    provider,
                    progress,
                    progress_task_id,
                    input_tokens,
                    result,
                    body_tokens,
                )
            return result

        except Exception as e:
            error_msg = str(e)
            category = classify_error(error_msg)
            self._log_global_task_failure(task.task_id, model_key, error_msg, category)

            if self.verbose:
                bind_context(ctx).bind(error_category=category.value).error(
                    f"API Call Failed: "
                    f"error={error_msg}, "
                    f"provider={model_config.provider}, "
                    f"model={model_config.model}"
                )

            result.status = TaskStatus.FAILED
            result.error = error_msg
            result.error_category = category

            update_global_task_progress_failed(
                progress, progress_task_id, model_key, task.task_id, progress_tokens
            )
            return result

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
        result = self._try_run_with_config(
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

        # Create a pool of progress bars sized to the worker count, not the task
        # count. A bar is acquired per worker, relabeled for the task it picks up,
        # and released when done. This keeps the UI at O(workers) bars instead of
        # O(tasks) -- N=1000 tasks no longer render 1000 live bars. See B6.
        if show_progress:
            from rich.console import Console as RichConsole
            from rich.progress import (
                BarColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            rich_console = RichConsole()

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("<"),
                TimeRemainingColumn(),
                console=rich_console,
                transient=False,
            )
            progress.start()

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
            slot_bars: list[TaskID] = [
                progress.add_task(f"[dim]worker {i} idle[/dim]", total=1) for i in range(num_slots)
            ]
            free_slots: _queue.Queue[int] = _queue.Queue()
            for i in range(num_slots):
                free_slots.put(i)
        else:
            progress = None
            task_meta = {}
            slot_bars = []
            free_slots = None  # type: ignore[assignment]

        # Flat attempt records accumulated across runner retries (B1 / P1.1). The
        # worker is stateless between calls; this side channel threads history so
        # the final result's attempt_history captures every preceding attempt.
        attempt_history_by_task: dict[int, list[AttemptRecord]] = {}

        try:

            def _worker(task: BatchTask, retry_count: int) -> BatchResult:
                progress_task_id: TaskID | None = None
                input_tokens: int | None = None
                slot_idx: int | None = None
                if progress is not None and free_slots is not None:
                    slot_idx = free_slots.get()  # pool == worker count, never blocks long
                    progress_task_id = slot_bars[slot_idx]
                    in_tok, est_out, model_key = task_meta.get(
                        task.task_id, (0, 1, "unknown/model")
                    )
                    input_tokens = in_tok
                    # Relabel + reset the reused bar for this task.
                    progress.update(
                        progress_task_id,
                        description=f"[cyan]{model_key}[/cyan] Task {task.task_id} ({in_tok} tok in)",
                        total=est_out,
                        completed=0,
                    )
                try:
                    return self._process_single_global_task(
                        task,
                        provider_cache,
                        retry_count,
                        progress,
                        progress_task_id,
                        input_tokens,
                        attempt_history_by_task,
                    )
                finally:
                    if (
                        progress is not None
                        and free_slots is not None
                        and slot_idx is not None
                    ):
                        free_slots.put(slot_idx)

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
            if show_progress and progress:
                progress.stop()

        return results

    def calculate_statistics(self, results: list[BatchResult]) -> dict[str, BatchStatistics]:
        """Calculate statistics from batch results, grouped by model.

        Thin delegate to the module-level :func:`calculate_statistics_by_model`.
        """
        return calculate_statistics_by_model(results)
