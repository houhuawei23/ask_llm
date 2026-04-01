"""Batch processing core logic and data models."""

import threading
import time
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from rich.progress import Progress, TaskID

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager

from ask_llm.config.context import get_config_or_none
from ask_llm.core.models import RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.utils.token_counter import TokenCounter
from llm_engine.concurrent import run_thread_pool_with_retries


class TaskStatus(str, Enum):
    """Task processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class ModelConfig(BaseModel):
    """Model configuration for batch processing."""

    provider: str
    model: str
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)


class BatchTask(BaseModel):
    """A single batch processing task."""

    task_id: int
    prompt: str
    content: str
    output_filename: Optional[str] = None  # Optional output filename for split mode
    task_model_config: Optional[
        ModelConfig
    ] = None  # Optional for backward compatibility (renamed from model_config to avoid Pydantic reserved keyword)
    # Paper explain (ask-llm paper): non-streaming completion; one provider per task like trans
    paper_mode: bool = False
    return_reasoning: bool = False


def sort_batch_tasks_by_estimated_input(
    tasks: List[BatchTask],
    default_model: str,
) -> List[BatchTask]:
    """
    Sort tasks by descending estimated full prompt tokens.

    When concurrent workers are fewer than tasks, heavy requests start earlier and reduce
    wall-clock tail latency.
    """

    def _estimate(t: BatchTask) -> int:
        model = t.task_model_config.model if t.task_model_config else default_model
        if "{content}" in t.prompt:
            full = t.prompt.replace("{content}", t.content)
        else:
            full = f"{t.prompt}\n\n{t.content}"
        return int(TokenCounter.estimate_tokens(full, model)["token_count"])

    return sorted(tasks, key=_estimate, reverse=True)


class BatchResult(BaseModel):
    """Result of a batch processing task."""

    model_config = ConfigDict(protected_namespaces=())

    task_id: int
    prompt: str
    content: str
    output_filename: Optional[str] = None  # Optional output filename for split mode
    model_settings: (
        ModelConfig  # Renamed from model_config to avoid conflict with Pydantic's reserved field
    )
    response: Optional[str] = None
    metadata: Optional[RequestMetadata] = None
    reasoning: Optional[str] = None  # e.g. DeepSeek reasoner when paper_mode + return_reasoning
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = 0


class BatchStatistics(BaseModel):
    """Statistics for batch processing."""

    total_tasks: int = Field(default=0, description="Total number of tasks")
    successful_tasks: int = Field(default=0, description="Number of successful tasks")
    failed_tasks: int = Field(default=0, description="Number of failed tasks")
    total_latency: float = Field(default=0.0, description="Total latency in seconds")
    average_latency: float = Field(default=0.0, description="Average latency per task")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")


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
        progress: Optional[Progress] = None,
        progress_task_id: Optional[TaskID] = None,
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
            for chunk in self.processor.process(
                content=task.content,
                prompt_template=task.prompt,
                temperature=self.model_config.temperature,
                model=self.model_config.model,
                max_tokens=self.model_config.max_tokens,
                stream=True,
            ):
                response_parts.append(chunk)
                # Estimate output tokens incrementally
                output_token_count = TokenCounter.count_tokens(
                    "".join(response_parts), self.model_config.model
                )

                # Update progress with token count
                if progress and progress_task_id is not None:
                    progress.update(
                        progress_task_id,
                        description=f"Task {task.task_id}: {output_token_count} tokens",
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
                    description=f"Task {task.task_id}: ✓ Complete ({output_token_count} tokens)",
                    completed=100,
                )

            logger.debug(f"Task {task.task_id} completed successfully")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.task_id} failed: {error_msg}")

            result.status = TaskStatus.FAILED
            result.error = error_msg

            # Update progress to failed
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"Task {task.task_id}: ✗ Failed",
                    completed=100,
                )

        return result

    def process_tasks(
        self, tasks: List[BatchTask], show_progress: bool = True
    ) -> List[BatchResult]:
        """
        Process multiple tasks concurrently with streaming and individual progress bars.

        Args:
            tasks: List of batch tasks
            show_progress: Whether to show progress bars

        Returns:
            List of batch results
        """
        pending_tasks = sort_batch_tasks_by_estimated_input(tasks.copy(), self.model_config.model)

        # Create progress display with multiple tasks
        if show_progress:
            # Get Rich console instance
            from rich.console import Console as RichConsole
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            rich_console = RichConsole()

            # Note: Rich Progress doesn't support max_completed parameter
            # Completed tasks will be shown, but terminal scrolling may be needed for many tasks
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=rich_console,
                transient=False,
            )
            progress.start()

            # Create a progress task for each batch task
            task_to_progress_id: Dict[int, TaskID] = {}
            for task in pending_tasks:
                progress_id = progress.add_task(
                    f"Task {task.task_id}: Waiting...",
                    total=100,  # Use 100 as total for percentage display
                )
                task_to_progress_id[task.task_id] = progress_id
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
                logger.debug(
                    f"Task {task.task_id} will be retried "
                    f"(attempt {failed_result.retry_count + 1}/{self.max_retries})"
                )

            def _on_worker_exception(task: BatchTask, exc: BaseException) -> BatchResult:
                logger.error(f"Unexpected error processing task {task.task_id}: {exc}")
                if progress and task.task_id in task_to_progress_id:
                    progress.update(
                        task_to_progress_id[task.task_id],
                        completed=100,
                    )
                return BatchResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    content=task.content,
                    output_filename=task.output_filename,
                    model_settings=self.model_config,
                    status=TaskStatus.FAILED,
                    error=f"Unexpected error: {exc!s}",
                )

            results = run_thread_pool_with_retries(
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

    def calculate_statistics(self, results: List[BatchResult]) -> BatchStatistics:
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
    ):
        """
        Initialize global batch processor.

        Args:
            max_workers: Maximum number of concurrent workers across all models
            max_retries: Maximum number of retries for failed tasks
            retry_delay: Initial delay between retries (exponential backoff)
            retry_delay_max: Maximum retry delay cap in seconds
            verbose: Enable verbose output with detailed API call information
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max
        self.verbose = verbose
        self._auth_error_lock = threading.Lock()
        self._auth_error_logged = False

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

    def _log_global_task_failure(self, task_id: int, model_key: str, error_msg: str) -> None:
        """Log task failure; collapse duplicate authentication errors from parallel workers."""
        if self._is_authentication_error(error_msg):
            with self._auth_error_lock:
                if not self._auth_error_logged:
                    self._auth_error_logged = True
                    logger.error(
                        f"API authentication failed ({model_key}): {error_msg}\n"
                        "(Further parallel tasks with the same auth error are logged at DEBUG only.)"
                    )
                else:
                    logger.debug(f"Task {task_id} ({model_key}) failed (auth): {error_msg}")
        else:
            logger.error(f"Task {task_id} ({model_key}) failed: {error_msg}")

    def _create_provider_for_task(
        self,
        model_config: ModelConfig,
        config_manager: "ConfigManager",
        timeout_override: Optional[float] = None,
    ) -> Tuple[LLMProviderProtocol, ModelConfig]:
        """
        Create provider instance for a specific model configuration.

        Args:
            model_config: Model configuration
            config_manager: Configuration manager instance

        Returns:
            Tuple of (provider instance, model_config)
        """
        # Set provider in config manager
        config_manager.set_provider(model_config.provider)

        # Apply model-specific overrides
        config_manager.apply_overrides(
            model=model_config.model,
            temperature=model_config.temperature,
        )

        provider_config_with_overrides = config_manager.get_provider_config()
        if timeout_override is not None:
            provider_config_with_overrides = provider_config_with_overrides.model_copy(
                update={"timeout": float(timeout_override)}
            )
        default_model = config_manager.get_model_override() or config_manager.get_default_model()

        # Create provider adapter
        from llm_engine import create_provider_adapter

        llm_provider = create_provider_adapter(
            provider_config_with_overrides, default_model=default_model
        )

        return llm_provider, model_config

    def _process_single_global_task(
        self,
        task: BatchTask,
        config_manager: "ConfigManager",
        retry_count: int = 0,
        progress: Optional[Progress] = None,
        progress_task_id: Optional[TaskID] = None,
        input_tokens: Optional[int] = None,
    ) -> BatchResult:
        """
        Process a single task with its associated model configuration.

        Args:
            task: Batch task with task_model_config
            config_manager: Configuration manager instance
            retry_count: Current retry count
            progress: Rich Progress object for updating progress
            progress_task_id: Task ID in progress bar

        Returns:
            Batch result
        """
        if not task.task_model_config:
            raise ValueError("Task must have task_model_config for global batch processing")

        model_config = task.task_model_config
        model_key = f"{model_config.provider}/{model_config.model}"

        # Initialize display_input_tokens early for use in error handling
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
        body_tokens = TokenCounter.count_tokens(task.content, model_config.model)
        progress_tokens = f"body≈{body_tokens} input≈{display_input_tokens}"

        try:
            paper_timeout: Optional[float] = None
            if task.paper_mode:
                lr = get_config_or_none()
                paper_timeout = 600.0
                if lr is not None:
                    paper_timeout = float(lr.unified_config.paper.request_timeout_seconds)

            # Create provider for this task (paper: longer HTTP timeout for reasoner / large completions)
            provider, _ = self._create_provider_for_task(
                model_config,
                config_manager,
                timeout_override=paper_timeout if task.paper_mode else None,
            )
            processor = RequestProcessor(provider)

            # Paper explain: streaming completion + live output token estimate (same UX as trans)
            if task.paper_mode:
                input_stats = TokenCounter.estimate_tokens(
                    task.prompt.strip(), model_config.model
                )
                input_token_count = input_stats["token_count"]
                display_input_tokens = (
                    input_tokens if input_tokens is not None else input_token_count
                )
                progress_tokens = f"paper≈{display_input_tokens} tok"

                if progress and progress_task_id is not None:
                    progress.update(
                        progress_task_id,
                        description=(
                            f"{model_key} paper task {task.task_id} ({progress_tokens} tokens): "
                            "0 out"
                        ),
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
                    logger.info(f"[Task {task.task_id}] Paper API: {', '.join(api_params)}")

                start_time = time.time()
                response_parts: List[str] = []
                reasoning_parts: List[str] = []

                for chunk in processor.iter_process_raw_stream(
                    task.prompt,
                    temperature=model_config.temperature,
                    model=model_config.model,
                    max_tokens=model_config.max_tokens,
                    return_reasoning=task.return_reasoning,
                ):
                    if task.return_reasoning:
                        c, r = chunk  # type: ignore[misc]
                        if c:
                            response_parts.append(c)
                        if r:
                            reasoning_parts.append(r)
                        out_so_far = "".join(reasoning_parts) + "".join(response_parts)
                    else:
                        response_parts.append(str(chunk))
                        out_so_far = "".join(response_parts)

                    output_token_count = TokenCounter.count_tokens(
                        out_so_far, model_config.model
                    )
                    if progress and progress_task_id is not None:
                        progress.update(
                            progress_task_id,
                            description=(
                                f"{model_key} paper task {task.task_id} ({progress_tokens} tokens): "
                                f"{output_token_count} out"
                            ),
                        )

                response = "".join(response_parts).strip()
                reasoning_joined = "".join(reasoning_parts).strip()
                reasoning_out = reasoning_joined if reasoning_joined else None
                latency = time.time() - start_time

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
                        description=(
                            f"{model_key} paper task {task.task_id}: ✓ Complete ({ot} out)"
                        ),
                        completed=100,
                    )

                logger.debug(f"Task {task.task_id} ({model_key}) paper job completed")
                return result

            # Format prompt with content
            full_prompt = processor._format_prompt(task.content, task.prompt)

            # Count input tokens
            input_stats = TokenCounter.estimate_tokens(full_prompt, model_config.model)
            input_token_count = input_stats["token_count"]
            # Use provided input_tokens if available, otherwise use calculated value
            display_input_tokens = input_tokens if input_tokens is not None else input_token_count
            progress_tokens = f"body≈{body_tokens} input≈{display_input_tokens}"

            # Log detailed API call information in verbose mode
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
                logger.info(f"[Task {task.task_id}] API Call: {', '.join(api_params)}")

            # Process with streaming
            start_time = time.time()
            response_parts = []
            output_token_count = 0

            # Update progress description
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"{model_key} Task {task.task_id} ({progress_tokens} tokens): Processing...",
                )
            elif progress is None:
                logger.info(
                    f"Translation chunk {task.task_id} start ({model_key}, {progress_tokens})"
                )

            # Stream response
            for chunk in processor.process(
                content=task.content,
                prompt_template=task.prompt,
                temperature=model_config.temperature,
                model=model_config.model,
                max_tokens=model_config.max_tokens,
                stream=True,
            ):
                response_parts.append(chunk)
                # Estimate output tokens incrementally
                output_token_count = TokenCounter.count_tokens(
                    "".join(response_parts), model_config.model
                )

                # Update progress with token count
                if progress and progress_task_id is not None:
                    progress.update(
                        progress_task_id,
                        description=f"{model_key} Task {task.task_id} ({progress_tokens} tokens): {output_token_count} out",
                    )

            response = "".join(response_parts)
            latency = time.time() - start_time

            # Create metadata
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

            # Log detailed API call result in verbose mode
            if self.verbose:
                logger.info(
                    f"[Task {task.task_id}] API Call Completed: "
                    f"output_tokens={output_token_count}, "
                    f"latency={latency:.2f}s, "
                    f"status=success"
                )

            # Update progress to completed
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"{model_key} Task {task.task_id} ({progress_tokens} tokens): ✓ Complete ({output_token_count} out)",
                    completed=100,
                )
            elif progress is None:
                logger.info(
                    f"Translation chunk {task.task_id} complete ({output_token_count} output tokens)"
                )

            logger.debug(f"Task {task.task_id} ({model_key}) completed successfully")

        except Exception as e:
            error_msg = str(e)
            self._log_global_task_failure(task.task_id, model_key, error_msg)

            # Log detailed error information in verbose mode
            if self.verbose:
                logger.error(
                    f"[Task {task.task_id}] API Call Failed: "
                    f"error={error_msg}, "
                    f"provider={model_config.provider}, "
                    f"model={model_config.model}"
                )

            result.status = TaskStatus.FAILED
            result.error = error_msg

            # Update progress to failed
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"{model_key} Task {task.task_id} ({progress_tokens} tokens): ✗ Failed",
                    completed=100,
                )

        return result

    def process_global_tasks(
        self,
        tasks: List[BatchTask],
        config_manager: "ConfigManager",
        show_progress: bool = True,
    ) -> List[BatchResult]:
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

        # Create progress display with multiple tasks
        if show_progress:
            from rich.console import Console as RichConsole
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            rich_console = RichConsole()

            # Note: Rich Progress doesn't support max_completed parameter
            # Completed tasks will be shown, but terminal scrolling may be needed for many tasks
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=rich_console,
                transient=False,
            )
            progress.start()

            # Create a progress task for each batch task
            task_to_progress_id: Dict[int, TaskID] = {}
            task_to_input_tokens: Dict[int, int] = {}
            for task in pending_tasks:
                model_key = (
                    f"{task.task_model_config.provider}/{task.task_model_config.model}"
                    if task.task_model_config
                    else "unknown"
                )
                # Pre-calculate input tokens for display
                # Simple estimation: combine prompt template and content
                # Replace {content} placeholder if present, otherwise append content
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

                progress_id = progress.add_task(
                    f"{model_key} Task {task.task_id} ({input_token_estimate} tokens): Waiting...",
                    total=100,  # Use 100 as total for percentage display
                )
                task_to_progress_id[task.task_id] = progress_id
        else:
            progress = None
            task_to_progress_id = {}
            task_to_input_tokens = {}

        try:

            def _worker(task: BatchTask, retry_count: int) -> BatchResult:
                return self._process_single_global_task(
                    task,
                    config_manager,
                    retry_count,
                    progress,
                    task_to_progress_id.get(task.task_id),
                    task_to_input_tokens.get(task.task_id),
                )

            def _on_retry_scheduled(task: BatchTask, failed_result: BatchResult) -> None:
                logger.debug(
                    f"Task {task.task_id} will be retried "
                    f"(attempt {failed_result.retry_count + 1}/{self.max_retries})"
                )

            def _on_worker_exception(task: BatchTask, exc: BaseException) -> BatchResult:
                logger.error(f"Unexpected error processing task {task.task_id}: {exc}")
                if progress and task.task_id in task_to_progress_id:
                    progress.update(
                        task_to_progress_id[task.task_id],
                        completed=100,
                    )
                return BatchResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    content=task.content,
                    output_filename=task.output_filename,
                    model_settings=task.task_model_config
                    or ModelConfig(provider="unknown", model="unknown"),
                    status=TaskStatus.FAILED,
                    error=f"Unexpected error: {exc!s}",
                )

            results = run_thread_pool_with_retries(
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

    def calculate_statistics(self, results: List[BatchResult]) -> Dict[str, BatchStatistics]:
        """
        Calculate statistics from batch results, grouped by model.

        Args:
            results: List of batch results

        Returns:
            Dictionary mapping model_key to statistics
        """
        # Group results by model
        results_by_model: Dict[str, List[BatchResult]] = {}
        for result in results:
            model_key = f"{result.model_settings.provider}/{result.model_settings.model}"
            if model_key not in results_by_model:
                results_by_model[model_key] = []
            results_by_model[model_key].append(result)

        # Calculate statistics for each model
        statistics: Dict[str, BatchStatistics] = {}
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
