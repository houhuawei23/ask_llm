"""Single-config task execution for the batch pipeline (P1.3 / TaskExecutor).

Owns, for one provider/model attempt: rate-limit acquire, adapter lookup,
streaming collection (via :func:`stream_and_collect`), request metadata
construction, progress updates, and the auth-error log de-duplication.

Extracted from ``GlobalBatchProcessor`` (ARCHITECTURE_REVIEW.md §7.2 / P1.3) so
execution is separable from scheduling and fallback orchestration. The two
module-level helpers moved with it because they are consumed only here
(``update_global_task_progress_failed``) or by both here and the provider cache
(``paper_request_timeout_seconds``); keeping them in this module avoids a
circular import with ``batch_processor``.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from rich.progress import Progress, TaskID

from ask_llm.config.context import get_config_or_none
from ask_llm.core.batch_models import BatchResult, BatchTask, ModelConfig, TaskStatus
from ask_llm.core.models import RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.core.stream_collector import stream_and_collect
from ask_llm.core.telemetry import (
    ErrorCategory,
    LogContext,
    bind_context,
    classify_error,
)
from ask_llm.utils.rate_limiter import get_global_rate_limiter
from ask_llm.utils.token_counter import TokenCounter


def paper_request_timeout_seconds() -> float:
    """Paper explain HTTP timeout from unified config, default 600s.

    Shared by provider-cache construction and task execution.
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


class TaskExecutor:
    """Execute a single provider/model attempt for a batch task."""

    def __init__(self, *, verbose: bool = False, stream_api: bool = True) -> None:
        self.verbose = verbose
        self.stream_api = stream_api
        self._auth_error_lock = threading.Lock()
        self._auth_error_logged = False

    def log_task_failure(
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

    def _run_paper_explain(
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

    def _run_translation_chunk(
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

    def try_run_with_config(
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
                result, _ = self._run_paper_explain(
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
                result, _ = self._run_translation_chunk(
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
            self.log_task_failure(task.task_id, model_key, error_msg, category)

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
