"""BatchProcessor and GlobalBatchProcessor implementations."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from rich.progress import Progress, TaskID

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager
    from ask_llm.config.unified_config import RateLimitConfig

from ask_llm.core.batch_models import (
    BatchResult,
    BatchStatistics,
    BatchTask,
    ModelConfig,
    TaskStatus,
    sort_batch_tasks_by_estimated_input,
)
from ask_llm.core.concurrent import BoundedRetryRunner, RunMetrics, run_bounded_with_retries
from ask_llm.core.models import RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol, ReasoningChunk
from ask_llm.core.telemetry import (
    ErrorCategory,
    LogContext,
    bind_context,
    classify_error,
    should_fallback_for_error,
)
from ask_llm.utils.provider_cache import ProviderAdapterCache
from ask_llm.utils.rate_limiter import get_global_rate_limiter
from ask_llm.utils.token_counter import TokenCounter

if TYPE_CHECKING:
    from ask_llm.config.unified_config import RateLimitConfig


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
        return 100  # Default minimum estimate

    if task_kind == "paper_explain":
        # Explanations tend to be longer than input
        return int(input_tokens * 2.0)
    else:
        # Translations typically similar length or slightly shorter
        return int(input_tokens * 1.1)


class BatchProcessor:
    """Process batch tasks with multi-threading and retry support."""

    def __init__(
        self,
        provider: LLMProviderProtocol,
        model_config: ModelConfig,
        max_workers: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_delay_max: float = 10.0,
    ):
        """
        Initialize batch processor.

        Args:
            provider: LLM provider instance
            model_config: Model configuration
            max_workers: Maximum number of concurrent workers
            max_retries: Maximum number of retries for failed tasks
            retry_delay: Initial delay between retries (exponential backoff)
            retry_delay_max: Maximum retry delay cap in seconds
        """
        self.provider = provider
        self.model_config = model_config
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max
        self.processor = RequestProcessor(provider)

    def _process_single_task(
        self,
        task: BatchTask,
        retry_count: int = 0,
        progress: Progress | None = None,
        progress_task_id: TaskID | None = None,
    ) -> BatchResult:
        """
        Process a single task with streaming support.

        Args:
            task: Batch task to process
            retry_count: Current retry count
            progress: Rich Progress object for updating progress
            progress_task_id: Task ID in progress bar

        Returns:
            Batch result
        """
        result = BatchResult(
            task_id=task.task_id,
            prompt=task.prompt,
            content=task.content,
            output_filename=task.output_filename,
            model_settings=self.model_config,
            status=TaskStatus.PROCESSING,
            retry_count=retry_count,
        )
        ctx = LogContext(
            task_id=task.task_id,
            provider=self.model_config.provider,
            model=self.model_config.model,
            attempt=retry_count + 1,
            phase="batch_processor",
        )

        try:
            # Format prompt with content
            full_prompt = self.processor._format_prompt(task.content, task.prompt)

            # Count input tokens
            input_stats = TokenCounter.estimate_tokens(full_prompt, self.model_config.model)

            # Process with streaming
            start_time = time.time()
            response_parts = []
            output_token_count = 0

            # Update progress description
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"Task {task.task_id}: Processing...",
                )

            # Stream response
            encoding = TokenCounter.get_encoding(self.model_config.model)
            for chunk in self.processor.process(
                content=task.content,
                prompt_template=task.prompt,
                temperature=self.model_config.temperature,
                model=self.model_config.model,
                max_tokens=self.model_config.max_tokens,
                stream=True,
            ):
                text_chunk = chunk.content if isinstance(chunk, ReasoningChunk) else chunk
                response_parts.append(text_chunk)
                # Incremental token counting (avoid O(N^2) re-encode)
                if encoding is not None:
                    output_token_count += len(encoding.encode(text_chunk))
                else:
                    output_token_count += TokenCounter.count_words(text_chunk)

                # Update progress with token count
                if progress and progress_task_id is not None:
                    progress.update(
                        progress_task_id,
                        completed=output_token_count,
                        description=f"Task {task.task_id}: {output_token_count} tok",
                    )

            response = "".join(response_parts)
            latency = time.time() - start_time

            # Create metadata
            metadata = RequestMetadata(
                provider=self.processor.provider.name,
                model=self.model_config.model,
                temperature=self.model_config.temperature
                if self.model_config.temperature is not None
                else self.processor.provider.config.api_temperature,
                input_words=input_stats["word_count"],
                input_tokens=input_stats["token_count"],
                output_words=TokenCounter.count_words(response),
                output_tokens=output_token_count,
                latency=latency,
            )

            result.response = response
            result.metadata = metadata
            result.status = TaskStatus.SUCCESS

            # Update progress to completed
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    completed=output_token_count,
                    description=f"Task {task.task_id}: ✓ Complete ({output_token_count} tok)",
                )

            bind_context(ctx).debug("Task completed successfully")
            return result

        except Exception as e:
            error_msg = str(e)
            category = classify_error(error_msg)
            bind_context(ctx).bind(error_category=category.value).error(f"Task failed: {error_msg}")

            result.status = TaskStatus.FAILED
            result.error = error_msg
            result.error_category = category

            # Update progress to failed
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"Task {task.task_id}: ✗ Failed",
                    completed=100,
                )

            return result

    def process_tasks(
        self, tasks: list[BatchTask], show_progress: bool = True
    ) -> list[BatchResult]:
        """
        Process multiple tasks concurrently with streaming and a single overall progress bar.

        Args:
            tasks: List of batch tasks
            show_progress: Whether to show progress bars

        Returns:
            List of batch results
        """
        pending_tasks = sort_batch_tasks_by_estimated_input(tasks.copy(), self.model_config.model)

        # Create per-task progress bars (one per concurrent worker)
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

            # Create per-task progress tracking with estimated output tokens as total
            task_to_progress_id: dict[int, TaskID] = {}
            for task in pending_tasks:
                # Estimate input tokens for this task
                estimated_prompt = (
                    task.prompt.replace("{content}", task.content)
                    if "{content}" in task.prompt
                    else f"{task.prompt}\n\n{task.content}"
                )
                input_token_estimate = TokenCounter.estimate_tokens(
                    estimated_prompt,
                    self.model_config.model,
                )["token_count"]

                # Estimate output tokens for progress calculation
                estimated_output = estimate_output_tokens(
                    task.task_kind if hasattr(task, "task_kind") else "translation",
                    input_token_estimate,
                )

                task_id = progress.add_task(
                    f"[cyan]{self.model_config.model}[/cyan] Task {task.task_id} ({input_token_estimate} tok in)",
                    total=estimated_output,
                )
                task_to_progress_id[task.task_id] = task_id
        else:
            progress = None
            task_to_progress_id = {}

        try:

            def _worker(task: BatchTask, retry_count: int) -> BatchResult:
                return self._process_single_task(
                    task,
                    retry_count,
                    progress,
                    task_to_progress_id.get(task.task_id),
                )

            def _on_retry_scheduled(task: BatchTask, failed_result: BatchResult) -> None:
                bind_context(LogContext(task_id=task.task_id, phase="batch_processor")).debug(
                    f"Task will be retried "
                    f"(attempt {failed_result.retry_count + 1}/{self.max_retries})"
                )

            def _on_worker_exception(task: BatchTask, exc: BaseException) -> BatchResult:
                error_msg = f"Unexpected error: {exc!s}"
                category = classify_error(error_msg)
                bind_context(LogContext(task_id=task.task_id, phase="batch_processor")).bind(
                    error_category=category.value
                ).error(f"Unexpected error processing task: {exc}")
                # Note: progress update is handled by _process_single_task/_process_single_global_task exception handler
                # This callback is for exceptions that escape before any progress is set
                return BatchResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    content=task.content,
                    output_filename=task.output_filename,
                    model_settings=self.model_config,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                    error_category=category,
                )

            results = run_bounded_with_retries(
                pending_tasks,
                _worker,
                max_workers=self.max_workers,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                retry_delay_max=self.retry_delay_max,
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

    def calculate_statistics(self, results: list[BatchResult]) -> BatchStatistics:
        """
        Calculate statistics from batch results.

        Args:
            results: List of batch results

        Returns:
            Batch statistics
        """
        stats = BatchStatistics(total_tasks=len(results))

        successful_results = [r for r in results if r.status == TaskStatus.SUCCESS]
        stats.successful_tasks = len(successful_results)
        stats.failed_tasks = len(results) - stats.successful_tasks

        if successful_results:
            latencies = [r.metadata.latency for r in successful_results if r.metadata]
            if latencies:
                stats.total_latency = sum(latencies)
                stats.average_latency = stats.total_latency / len(latencies)

            input_tokens = [r.metadata.input_tokens for r in successful_results if r.metadata]
            output_tokens = [r.metadata.output_tokens for r in successful_results if r.metadata]

            stats.total_input_tokens = sum(input_tokens)
            stats.total_output_tokens = sum(output_tokens)

        return stats


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

    @staticmethod
    def _is_authentication_error(error_msg: str) -> bool:
        e = (error_msg or "").lower()
        return any(
            k in e
            for k in (
                "401",
                "authentication",
                "invalid api key",
                "api key",
                "invalid_request_error",
                "authentication_error",
                "unauthorized",
            )
        )

    def _paper_request_timeout_seconds(self) -> float:
        """Paper explain HTTP timeout from unified config, default 600s."""
        lr = get_config_or_none()
        if lr is None:
            return 600.0
        return float(lr.unified_config.paper.request_timeout_seconds)

    def _effective_max_workers(self, tasks: list[BatchTask]) -> int:
        """Return ``max_workers`` capped by the tightest burst limit among tasks."""
        limiter = get_global_rate_limiter(self.rate_limit_config)
        min_burst: int | None = None
        for task in tasks:
            if task.task_model_config is None:
                continue
            burst = limiter.burst_for(task.task_model_config.provider, task.task_model_config.model)
            if min_burst is None or burst < min_burst:
                min_burst = burst
        if min_burst is None:
            return self.max_workers
        return max(1, min(self.max_workers, min_burst))

    def _update_global_task_progress_failed(
        self,
        progress: Progress | None,
        progress_task_id: TaskID | None,
        model_key: str,
        task_id: int,
        progress_tokens: str,
    ) -> None:
        if progress and progress_task_id is not None:
            progress.update(
                progress_task_id,
                description=(f"{model_key} Task {task_id} ({progress_tokens} tokens): ✗ Failed"),
                completed=100,
            )

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

    def _build_provider_cache(
        self,
        tasks: list[BatchTask],
        config_manager: ConfigManager,
    ) -> dict[str, LLMProviderProtocol]:
        """
        Pre-build provider adapter cache for all unique (provider, model, timeout) combos.

        This eliminates per-task adapter creation overhead and avoids mutating the
        shared ConfigManager inside worker threads.
        """
        cache: dict[str, LLMProviderProtocol] = {}

        def _add_config(mc: ModelConfig) -> None:
            key = f"{mc.provider}/{mc.model}"
            if task.task_kind == "paper_explain":
                timeout = self._paper_request_timeout_seconds()
                key += f" / timeout={timeout}"
            if key in seen:
                return
            seen.add(key)

            base_cfg = config_manager.config.get_provider_config(mc.provider)
            overrides: dict[str, Any] = {}
            if mc.temperature is not None:
                overrides["api_temperature"] = mc.temperature
            if mc.max_tokens is not None:
                overrides["max_tokens"] = mc.max_tokens
            if mc.top_p is not None:
                overrides["api_top_p"] = mc.top_p
            if task.task_kind == "paper_explain":
                overrides["timeout"] = float(self._paper_request_timeout_seconds())
            provider_cfg = base_cfg.model_copy(update=overrides)
            default_model = mc.model

            provider = ProviderAdapterCache.get(provider_cfg, default_model=default_model)
            cache[key] = provider

        seen: set[str] = set()
        for task in tasks:
            if not task.task_model_config:
                continue
            _add_config(task.task_model_config)
            for fallback_config in task.fallback_model_configs:
                _add_config(fallback_config)
        return cache

    def _stream_and_collect(
        self,
        stream_iter: Iterator[str | ReasoningChunk],
        task: BatchTask,
        model_config: ModelConfig,
        progress: Progress | None,
        progress_task_id: TaskID | None,
        description_prefix: str,
        return_reasoning: bool = False,
    ) -> tuple[str, str | None, int, float]:
        """Run a streaming iterator, counting tokens and updating progress."""
        start_time = time.time()
        response_parts: list[str] = []
        reasoning_parts: list[str] = []
        output_token_count = 0

        encoding = TokenCounter.get_encoding(model_config.model)
        for chunk in stream_iter:
            if return_reasoning:
                assert isinstance(chunk, ReasoningChunk)
                if chunk.content:
                    response_parts.append(chunk.content)
                    if encoding is not None:
                        output_token_count += len(encoding.encode(chunk.content))
                    else:
                        output_token_count += TokenCounter.count_words(chunk.content)
                if chunk.reasoning:
                    reasoning_parts.append(chunk.reasoning)
                    if encoding is not None:
                        output_token_count += len(encoding.encode(chunk.reasoning))
                    else:
                        output_token_count += TokenCounter.count_words(chunk.reasoning)
            else:
                text_chunk = chunk if isinstance(chunk, str) else str(chunk)
                response_parts.append(text_chunk)
                if encoding is not None:
                    output_token_count += len(encoding.encode(text_chunk))
                else:
                    output_token_count += TokenCounter.count_words(text_chunk)

            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    completed=output_token_count,
                    description=f"{description_prefix}: {output_token_count} tok",
                )

        response = "".join(response_parts).strip()
        reasoning = "".join(reasoning_parts).strip() if reasoning_parts else None
        latency = time.time() - start_time
        return response, reasoning, output_token_count, latency

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
        response, reasoning_out, _output_token_count, latency = self._stream_and_collect(
            stream_iter,
            task,
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

        metadata = RequestMetadata(
            provider=provider.name,
            model=model_config.model,
            temperature=model_config.temperature
            if model_config.temperature is not None
            else provider.config.api_temperature,
            input_words=input_stats["word_count"],
            input_tokens=input_stats["token_count"],
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
        response, _, output_token_count, latency = self._stream_and_collect(
            stream_iter,
            task,
            model_config,
            progress,
            progress_task_id,
            description_prefix,
            return_reasoning=False,
        )

        metadata = RequestMetadata(
            provider=provider.name,
            model=model_config.model,
            temperature=model_config.temperature
            if model_config.temperature is not None
            else provider.config.api_temperature,
            input_words=input_stats["word_count"],
            input_tokens=input_stats["token_count"],
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
            acquired = limiter.acquire(
                model_config.provider,
                model_config.model,
                timeout=60.0,
            )
            if not acquired:
                raise RuntimeError(
                    f"Rate limit timeout for {model_config.provider}/{model_config.model}"
                )

            paper_timeout: float | None = None
            if task.task_kind == "paper_explain":
                paper_timeout = self._paper_request_timeout_seconds()

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

            self._update_global_task_progress_failed(
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
    ) -> BatchResult:
        """
        Process a single task, trying the primary config and any fallbacks.

        Args:
            task: Batch task with task_model_config
            provider_cache: Pre-built provider adapter cache
            retry_count: Current retry count
            progress: Rich Progress object for updating progress
            progress_task_id: Task ID in progress bar

        Returns:
            Batch result
        """
        if not task.task_model_config:
            raise ValueError("Task must have task_model_config for global batch processing")

        configs = [task.task_model_config, *task.fallback_model_configs]
        last_result: BatchResult | None = None
        attempt_history: list[BatchResult] = []

        for idx, model_config in enumerate(configs):
            result = self._try_run_with_config(
                task,
                model_config,
                provider_cache,
                retry_count,
                progress,
                progress_task_id,
                input_tokens,
            )
            attempt_history.append(result)
            if result.status == TaskStatus.SUCCESS:
                # Record only the preceding failed attempts, not the successful
                # result itself, to avoid a circular reference during serialization.
                result.attempt_history = list(attempt_history[:-1])
                return result
            last_result = result

            # Stop fallback chain early for errors that won't be fixed by a
            # different provider/model (e.g. bad API key or content policy).
            if result.error_category and not should_fallback_for_error(result.error_category):
                terminal = result.error_category.value
                bind_context(LogContext(task_id=task.task_id, phase="global_batch")).warning(
                    f"Stopping fallback chain at attempt {idx + 1}: terminal error category '{terminal}'"
                )
                break

        final = last_result or BatchResult(
            task_id=task.task_id,
            prompt=task.prompt,
            content=task.content,
            output_filename=task.output_filename,
            model_settings=task.task_model_config,
            status=TaskStatus.FAILED,
            error="All provider/model configurations failed",
            error_category=ErrorCategory.UNKNOWN,
        )
        # If ``final`` is one of the recorded attempts, exclude itself from its
        # own history to keep the model graph acyclic.
        final.attempt_history = (
            list(attempt_history[:-1]) if final in attempt_history else list(attempt_history)
        )
        return final

    def process_global_tasks(
        self,
        tasks: list[BatchTask],
        config_manager: ConfigManager,
        show_progress: bool = True,
    ) -> list[BatchResult]:
        """
        Process multiple tasks across different models concurrently.

        Args:
            tasks: List of batch tasks, each with task_model_config
            config_manager: Configuration manager instance
            show_progress: Whether to show progress bars

        Returns:
            List of batch results
        """
        default_model = (
            tasks[0].task_model_config.model
            if tasks and tasks[0].task_model_config
            else "gpt-3.5-turbo"
        )
        pending_tasks = sort_batch_tasks_by_estimated_input(tasks.copy(), default_model)

        # Pre-build provider cache to avoid per-task adapter creation and ConfigManager mutation
        provider_cache = self._build_provider_cache(pending_tasks, config_manager)

        # Cap the thread pool size to the smallest configured burst limit so that
        # workers do not sit blocked on the rate limiter waiting for tokens.
        effective_max_workers = self._effective_max_workers(pending_tasks)

        # Create per-task progress bars (one per concurrent worker)
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

            # Pre-calculate input tokens and create per-task progress tracking
            task_to_input_tokens: dict[int, int] = {}
            task_to_progress_id: dict[int, TaskID] = {}
            for task in pending_tasks:
                # Pre-calculate input tokens for display
                estimated_prompt = (
                    task.prompt.replace("{content}", task.content)
                    if "{content}" in task.prompt
                    else f"{task.prompt}\n\n{task.content}"
                )
                input_token_estimate = TokenCounter.estimate_tokens(
                    estimated_prompt,
                    task.task_model_config.model if task.task_model_config else "gpt-3.5-turbo",
                )["token_count"]
                task_to_input_tokens[task.task_id] = input_token_estimate

                # Estimate output tokens for progress calculation
                estimated_output = estimate_output_tokens(
                    task.task_kind if hasattr(task, "task_kind") else "translation",
                    input_token_estimate,
                )

                # Create a per-task progress bar with estimated output tokens as total
                model_key = (
                    f"{task.task_model_config.provider}/{task.task_model_config.model}"
                    if task.task_model_config
                    else "unknown/model"
                )
                task_id = progress.add_task(
                    f"[cyan]{model_key}[/cyan] Task {task.task_id} ({input_token_estimate} tok in)",
                    total=estimated_output,
                )
                task_to_progress_id[task.task_id] = task_id
        else:
            progress = None
            task_to_input_tokens = {}
            task_to_progress_id = {}

        try:

            def _worker(task: BatchTask, retry_count: int) -> BatchResult:
                return self._process_single_global_task(
                    task,
                    provider_cache,
                    retry_count,
                    progress,
                    task_to_progress_id.get(task.task_id),
                    task_to_input_tokens.get(task.task_id),
                )

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
                    model_settings=task.task_model_config
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
        """
        Calculate statistics from batch results, grouped by model.

        Args:
            results: List of batch results

        Returns:
            Dictionary mapping model_key to statistics
        """
        # Group results by model
        results_by_model: dict[str, list[BatchResult]] = {}
        for result in results:
            model_key = f"{result.model_settings.provider}/{result.model_settings.model}"
            if model_key not in results_by_model:
                results_by_model[model_key] = []
            results_by_model[model_key].append(result)

        # Calculate statistics for each model
        statistics: dict[str, BatchStatistics] = {}
        for model_key, model_results in results_by_model.items():
            stats = BatchStatistics(total_tasks=len(model_results))

            successful_results = [r for r in model_results if r.status == TaskStatus.SUCCESS]
            stats.successful_tasks = len(successful_results)
            stats.failed_tasks = len(model_results) - stats.successful_tasks

            if successful_results:
                latencies = [r.metadata.latency for r in successful_results if r.metadata]
                if latencies:
                    stats.total_latency = sum(latencies)
                    stats.average_latency = stats.total_latency / len(latencies)

                input_tokens = [r.metadata.input_tokens for r in successful_results if r.metadata]
                output_tokens = [r.metadata.output_tokens for r in successful_results if r.metadata]

                stats.total_input_tokens = sum(input_tokens)
                stats.total_output_tokens = sum(output_tokens)

            statistics[model_key] = stats

        return statistics
