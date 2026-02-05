"""Batch processing core logic and data models."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field
from rich.progress import Progress, TaskID

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager

from ask_llm.core.models import RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.utils.token_counter import TokenCounter


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
    task_model_config: Optional[
        ModelConfig
    ] = None  # Optional for backward compatibility (renamed from model_config to avoid Pydantic reserved keyword)


class BatchResult(BaseModel):
    """Result of a batch processing task."""

    task_id: int
    prompt: str
    content: str
    model_settings: (
        ModelConfig  # Renamed from model_config to avoid conflict with Pydantic's reserved field
    )
    response: Optional[str] = None
    metadata: Optional[RequestMetadata] = None
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
    ):
        """
        Initialize batch processor.

        Args:
            provider: LLM provider instance
            model_config: Model configuration
            max_workers: Maximum number of concurrent workers
            max_retries: Maximum number of retries for failed tasks
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.provider = provider
        self.model_config = model_config
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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

    def _is_retryable_error(self, error: str) -> bool:
        """
        Check if an error is retryable.

        Args:
            error: Error message

        Returns:
            True if error is retryable
        """
        retryable_keywords = [
            "timeout",
            "connection",
            "network",
            "rate limit",
            "429",
            "503",
            "502",
            "500",
        ]
        error_lower = error.lower()
        return any(keyword in error_lower for keyword in retryable_keywords)

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
        results: List[BatchResult] = []
        pending_tasks = tasks.copy()

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
            for task in tasks:
                progress_id = progress.add_task(
                    f"Task {task.task_id}: Waiting...",
                    total=100,  # Use 100 as total for percentage display
                )
                task_to_progress_id[task.task_id] = progress_id
        else:
            progress = None
            task_to_progress_id = {}

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit initial tasks with progress tracking
                future_to_task = {}
                for task in pending_tasks:
                    progress_task_id = task_to_progress_id.get(task.task_id)
                    future = executor.submit(
                        self._process_single_task,
                        task,
                        0,  # retry_count
                        progress,
                        progress_task_id,
                    )
                    future_to_task[future] = task

                # Process completed tasks and retry failed ones
                retry_queue: List[Tuple[BatchTask, int]] = []  # (task, retry_count)

                while future_to_task or retry_queue:
                    # Process completed futures
                    if future_to_task:
                        for future in as_completed(future_to_task):
                            task = future_to_task.pop(future)
                            try:
                                result = future.result()

                                if result.status == TaskStatus.FAILED:
                                    # Check if we should retry
                                    if (
                                        result.retry_count < self.max_retries
                                        and self._is_retryable_error(result.error or "")
                                    ):
                                        # Add to retry queue with incremented retry count
                                        retry_queue.append((task, result.retry_count + 1))
                                        logger.debug(
                                            f"Task {task.task_id} will be retried "
                                            f"(attempt {result.retry_count + 1}/{self.max_retries})"
                                        )
                                    else:
                                        results.append(result)
                                        # Mark progress as complete
                                        if progress and task.task_id in task_to_progress_id:
                                            progress.update(
                                                task_to_progress_id[task.task_id],
                                                completed=100,
                                            )
                                else:
                                    results.append(result)
                                    # Mark progress as complete
                                    if progress and task.task_id in task_to_progress_id:
                                        progress.update(
                                            task_to_progress_id[task.task_id],
                                            completed=100,
                                        )

                            except Exception as e:
                                logger.error(
                                    f"Unexpected error processing task {task.task_id}: {e}"
                                )
                                results.append(
                                    BatchResult(
                                        task_id=task.task_id,
                                        prompt=task.prompt,
                                        content=task.content,
                                        model_settings=self.model_config,
                                        status=TaskStatus.FAILED,
                                        error=f"Unexpected error: {e!s}",
                                    )
                                )
                                # Mark progress as complete
                                if progress and task.task_id in task_to_progress_id:
                                    progress.update(
                                        task_to_progress_id[task.task_id],
                                        completed=100,
                                    )

                    # Retry failed tasks with exponential backoff
                    if retry_queue:
                        # Calculate delay with exponential backoff
                        delay = self.retry_delay * (2 ** (retry_queue[0][1] - 1))
                        time.sleep(min(delay, 10.0))  # Cap at 10 seconds

                        # Submit retry tasks
                        for task, retry_count in retry_queue:
                            progress_task_id = task_to_progress_id.get(task.task_id)
                            future = executor.submit(
                                self._process_single_task,
                                task,
                                retry_count,
                                progress,
                                progress_task_id,
                            )
                            future_to_task[future] = task

                        retry_queue.clear()

        finally:
            if show_progress and progress:
                progress.stop()

        # Sort results by task_id to maintain order
        results.sort(key=lambda r: r.task_id)

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
    ):
        """
        Initialize global batch processor.

        Args:
            max_workers: Maximum number of concurrent workers across all models
            max_retries: Maximum number of retries for failed tasks
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _create_provider_for_task(
        self,
        model_config: ModelConfig,
        config_manager: "ConfigManager",
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

        result = BatchResult(
            task_id=task.task_id,
            prompt=task.prompt,
            content=task.content,
            model_settings=model_config,
            status=TaskStatus.PROCESSING,
            retry_count=retry_count,
        )

        try:
            # Create provider for this task
            provider, _ = self._create_provider_for_task(model_config, config_manager)
            processor = RequestProcessor(provider)

            # Format prompt with content
            full_prompt = processor._format_prompt(task.content, task.prompt)

            # Count input tokens
            input_stats = TokenCounter.estimate_tokens(full_prompt, model_config.model)

            # Process with streaming
            start_time = time.time()
            response_parts = []
            output_token_count = 0

            # Update progress description
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"{model_key} Task {task.task_id}: Processing...",
                )

            # Stream response
            for chunk in processor.process(
                content=task.content,
                prompt_template=task.prompt,
                temperature=model_config.temperature,
                model=model_config.model,
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
                        description=f"{model_key} Task {task.task_id}: {output_token_count} tokens",
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

            # Update progress to completed
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"{model_key} Task {task.task_id}: ✓ Complete ({output_token_count} tokens)",
                    completed=100,
                )

            logger.debug(f"Task {task.task_id} ({model_key}) completed successfully")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.task_id} ({model_key}) failed: {error_msg}")

            result.status = TaskStatus.FAILED
            result.error = error_msg

            # Update progress to failed
            if progress and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=f"{model_key} Task {task.task_id}: ✗ Failed",
                    completed=100,
                )

        return result

    def _is_retryable_error(self, error: str) -> bool:
        """
        Check if an error is retryable.

        Args:
            error: Error message

        Returns:
            True if error is retryable
        """
        retryable_keywords = [
            "timeout",
            "connection",
            "network",
            "rate limit",
            "429",
            "503",
            "502",
            "500",
        ]
        error_lower = error.lower()
        return any(keyword in error_lower for keyword in retryable_keywords)

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
        results: List[BatchResult] = []
        pending_tasks = tasks.copy()

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
            for task in tasks:
                model_key = (
                    f"{task.task_model_config.provider}/{task.task_model_config.model}"
                    if task.task_model_config
                    else "unknown"
                )
                progress_id = progress.add_task(
                    f"{model_key} Task {task.task_id}: Waiting...",
                    total=100,  # Use 100 as total for percentage display
                )
                task_to_progress_id[task.task_id] = progress_id
        else:
            progress = None
            task_to_progress_id = {}

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit initial tasks with progress tracking
                future_to_task = {}
                for task in pending_tasks:
                    progress_task_id = task_to_progress_id.get(task.task_id)
                    future = executor.submit(
                        self._process_single_global_task,
                        task,
                        config_manager,
                        0,  # retry_count
                        progress,
                        progress_task_id,
                    )
                    future_to_task[future] = task

                # Process completed tasks and retry failed ones
                retry_queue: List[Tuple[BatchTask, int]] = []  # (task, retry_count)

                while future_to_task or retry_queue:
                    # Process completed futures
                    if future_to_task:
                        for future in as_completed(future_to_task):
                            task = future_to_task.pop(future)
                            try:
                                result = future.result()

                                if result.status == TaskStatus.FAILED:
                                    # Check if we should retry
                                    if (
                                        result.retry_count < self.max_retries
                                        and self._is_retryable_error(result.error or "")
                                    ):
                                        # Add to retry queue with incremented retry count
                                        retry_queue.append((task, result.retry_count + 1))
                                        logger.debug(
                                            f"Task {task.task_id} will be retried "
                                            f"(attempt {result.retry_count + 1}/{self.max_retries})"
                                        )
                                    else:
                                        results.append(result)
                                        # Mark progress as complete
                                        if progress and task.task_id in task_to_progress_id:
                                            progress.update(
                                                task_to_progress_id[task.task_id],
                                                completed=100,
                                            )
                                else:
                                    results.append(result)
                                    # Mark progress as complete
                                    if progress and task.task_id in task_to_progress_id:
                                        progress.update(
                                            task_to_progress_id[task.task_id],
                                            completed=100,
                                        )

                            except Exception as e:
                                logger.error(
                                    f"Unexpected error processing task {task.task_id}: {e}"
                                )
                                if task.task_model_config:
                                    model_key = f"{task.task_model_config.provider}/{task.task_model_config.model}"
                                else:
                                    model_key = "unknown"
                                results.append(
                                    BatchResult(
                                        task_id=task.task_id,
                                        prompt=task.prompt,
                                        content=task.content,
                                        model_settings=task.task_model_config
                                        or ModelConfig(provider="unknown", model="unknown"),
                                        status=TaskStatus.FAILED,
                                        error=f"Unexpected error: {e!s}",
                                    )
                                )
                                # Mark progress as complete
                                if progress and task.task_id in task_to_progress_id:
                                    progress.update(
                                        task_to_progress_id[task.task_id],
                                        completed=100,
                                    )

                    # Retry failed tasks with exponential backoff
                    if retry_queue:
                        # Calculate delay with exponential backoff
                        delay = self.retry_delay * (2 ** (retry_queue[0][1] - 1))
                        time.sleep(min(delay, 10.0))  # Cap at 10 seconds

                        # Submit retry tasks
                        for task, retry_count in retry_queue:
                            progress_task_id = task_to_progress_id.get(task.task_id)
                            future = executor.submit(
                                self._process_single_global_task,
                                task,
                                config_manager,
                                retry_count,
                                progress,
                                progress_task_id,
                            )
                            future_to_task[future] = task

                        retry_queue.clear()

        finally:
            if show_progress and progress:
                progress.stop()

        # Sort results by task_id to maintain order
        results.sort(key=lambda r: r.task_id)

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
