"""Batch processing orchestration service.

Moves the core batch workflow (load config, validate models, build tasks,
run them through GlobalBatchProcessor) out of the CLI command so the command
module stays focused on argument parsing, output formatting and user-facing
messages.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.config.unified_config import BatchConfig as UnifiedBatchConfig
from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig
from ask_llm.core.batch_checkpoint import BatchCheckpoint
from ask_llm.core.batch_models import BatchStatistics, TaskStatus
from ask_llm.core.execution_report import ExecutionReport, build_report_from_batch_results
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.models import AppConfig
from ask_llm.utils.api_key_gate import (
    api_key_is_missing_or_unresolved,
    ensure_api_key_for_provider,
    require_resolved_api_key,
)
from ask_llm.utils.batch_exporter import BatchResultExporter
from ask_llm.utils.batch_loader import BatchConfigLoader
from ask_llm.utils.console import console
from ask_llm.utils.interactive_config import InteractiveConfigHelper
from ask_llm.utils.pricing import format_cost_estimate
from ask_llm.utils.provider_cache import ProviderAdapterCache
from ask_llm.utils.provider_router import build_fallback_chain

PricingMap = dict[tuple[str, str], dict[str, float]]


@dataclass
class BatchRunResult:
    """Result of a batch processing run, ready for CLI export/summary."""

    all_results: list[BatchResult]
    model_statistics: dict[str, BatchStatistics]
    validated_models: list[ModelConfig]
    skipped_models: list[str]
    original_tasks: list[BatchTask]
    batch_mode: str
    batch_config: dict[str, Any]
    config_file: str
    report: ExecutionReport | None = None


@dataclass
class _ValidationResult:
    """Internal result from the model validation step."""

    validated: list[ModelConfig] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _validate_models(
    provider_models: list[ModelConfig],
    app_config: AppConfig,
    config_manager: ConfigManager,
    skip_api_key_check: bool,
) -> _ValidationResult:
    """Validate provider/model list and test connections."""
    result = _ValidationResult()

    for model_config in provider_models:
        model_key = f"{model_config.provider}/{model_config.model}"
        console.print(f"  Checking {model_key}...", end=" ")

        try:
            if model_config.provider not in app_config.providers:
                console.print("[red]✗[/red] Provider not found")
                result.skipped.append(model_key)
                continue

            provider_config = app_config.providers[model_config.provider]

            if api_key_is_missing_or_unresolved(provider_config.api_key):
                console.print("[red]✗[/red] API key not configured")
                result.skipped.append(model_key)
                continue

            if provider_config.models and model_config.model not in provider_config.models:
                console.print(
                    f"[red]✗[/red] Model not available. Available: {', '.join(provider_config.models)}"
                )
                result.skipped.append(model_key)
                continue

            config_manager.set_provider(model_config.provider)
            config_manager.apply_overrides(
                model=model_config.model,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                top_p=model_config.top_p,
            )

            provider_config_with_overrides = config_manager.get_provider_config()
            default_model = (
                config_manager.get_model_override() or config_manager.get_default_model()
            )

            try:
                test_provider = ProviderAdapterCache.get(
                    provider_config_with_overrides, default_model=default_model
                )
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to create provider adapter: {e}")
                result.skipped.append(model_key)
                continue

            success, message, latency = test_provider.test_connection()

            if not success:
                console.print(f"[red]✗[/red] Connection test failed: {message}")
                result.skipped.append(model_key)
                continue

            console.print(f"[green]✓ ({latency:.2f}s)[/green]")
            result.validated.append(model_config)

        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
            result.skipped.append(model_key)

    return result


def _default_batch_checkpoint_path(config_file: str) -> str:
    """Return a default checkpoint path next to the batch config file."""
    p = Path(config_file)
    return str(p.parent / f"{p.name}.checkpoint.json")


def run_batch_from_config(
    config_file: str,
    app_config: AppConfig,
    config_manager: ConfigManager,
    batch_config_unified: Any,
    *,
    output_format: str,
    threads: int,
    retries: int,
    retry_delay: float,
    retry_delay_max: float,
    skip_api_key_check: bool = False,
    verbose: bool = False,
    resume_checkpoint_path: str | None = None,
    use_fallback: bool = True,
) -> BatchRunResult:
    """Load a batch YAML config, validate models, and execute all tasks.

    Args:
        config_file: Path to the batch YAML configuration file.
        app_config: Loaded application config (providers, etc.).
        config_manager: Active config manager for provider/model overrides.
        batch_config_unified: ``batch`` section from the unified config.
        output_format: Desired output format (used only for validation logging here).
        threads: Max concurrent workers.
        retries: Max retries per failed task.
        retry_delay: Initial retry delay.
        retry_delay_max: Max retry delay cap.
        skip_api_key_check: Skip API key validation.
        verbose: Enable verbose provider output.
        use_fallback: Whether to enable fallback to alternate providers/models.

    Returns:
        BatchRunResult with all results, statistics and metadata for export.

    Raises:
        ValueError: If no providers validate.
    """
    logger.debug(f"Batch output format: {output_format}")

    # Load batch configuration
    console.print_info(f"Loading batch configuration from: {config_file}")
    batch_config = BatchConfigLoader.load(config_file)

    tasks = batch_config["tasks"]
    provider_models: list[ModelConfig] = batch_config.get("provider_models", [])

    console.print_success(f"Loaded {len(tasks)} tasks from configuration")

    # If no models specified, use interactive selection
    if not provider_models:
        console.print_info("No models specified in configuration. Using interactive selection.")
        helper = InteractiveConfigHelper(config_manager)
        provider_models = helper.select_provider_and_models(allow_multiple=True)

    batch_mode = batch_config.get("mode", batch_config_unified.mode)

    unique_providers = sorted({m.provider for m in provider_models})
    for pname in unique_providers:
        strict_gate = ensure_api_key_for_provider(
            config_manager,
            pname,
            skip_api_key_check=skip_api_key_check,
        )
        if strict_gate:
            require_resolved_api_key(config_manager, pname)

    # Validate all models and test connections
    console.print()
    console.print("[bold]Validating models and testing connections...[/bold]")
    validation = _validate_models(
        provider_models,
        app_config,
        config_manager,
        skip_api_key_check,
    )

    if validation.skipped:
        console.print()
        console.print_warning(f"Skipped {len(validation.skipped)} provider(s):")
        for skipped in validation.skipped:
            console.print(f"  - {skipped}")

    if not validation.validated:
        raise ValueError("No providers were successfully validated. Cannot process tasks.")

    # Create global task list (each task with model_config)
    console.print()
    console.print(
        f"[bold]Processing {len(validation.validated)} model(s) with "
        f"{len(tasks)} task(s) each...[/bold]"
    )
    global_tasks: list[BatchTask] = []
    task_id_counter = 0

    for model_config in validation.validated:
        fallback_configs = build_fallback_chain(app_config, model_config) if use_fallback else []
        for original_task in tasks:
            global_task = BatchTask(
                task_id=task_id_counter,
                prompt=original_task.prompt,
                content=original_task.content,
                output_filename=original_task.output_filename,
                model_settings=model_config,
                fallback_model_configs=fallback_configs,
            )
            global_tasks.append(global_task)
            task_id_counter += 1

    # Checkpoint handling for resume
    checkpoint_path = resume_checkpoint_path or _default_batch_checkpoint_path(config_file)
    checkpoint = BatchCheckpoint.create(command="batch", config_digest=config_file)
    prior_successful_results: list[BatchResult] = []

    if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
        checkpoint = BatchCheckpoint.load(resume_checkpoint_path)
        prior_successful_results = list(checkpoint.successful_results)
        remaining_tasks = [t for t in global_tasks if not checkpoint.is_completed(t.task_id)]
        console.print_info(
            f"Resuming from checkpoint: {len(remaining_tasks)}/{len(global_tasks)} tasks remaining"
        )
        global_tasks = remaining_tasks

    if not global_tasks:
        console.print_info("All tasks already completed according to checkpoint.")
        all_results_list = list(prior_successful_results)
        model_statistics = BatchStatistics.from_results(all_results_list)
        report = build_report_from_batch_results(
            "batch",
            all_results_list,
            metadata={"config_file": config_file, "checkpoint": checkpoint_path},
        )
        return BatchRunResult(
            all_results=all_results_list,
            model_statistics=model_statistics,
            validated_models=validation.validated,
            skipped_models=validation.skipped,
            original_tasks=tasks,
            batch_mode=batch_mode,
            batch_config=batch_config,
            config_file=config_file,
            report=report,
        )

    all_results_list, global_processor = run_global_batch_tasks(
        global_tasks,
        config_manager,
        max_workers=threads,
        max_retries=retries,
        retry_delay=retry_delay,
        retry_delay_max=retry_delay_max,
        verbose=verbose,
        show_progress=True,
        clamp_workers_to_task_count=False,
    )

    # Merge new successful results into checkpoint and persist
    new_successful = [r for r in all_results_list if r.status == TaskStatus.SUCCESS]
    new_failed = [r for r in all_results_list if r.status == TaskStatus.FAILED]
    checkpoint.merge(new_successful)
    checkpoint.mark_all_failed_for_retry(new_failed)
    checkpoint.save(checkpoint_path)

    # Rebuild full result list including prior successful results from resume
    all_results_list = list(checkpoint.successful_results) + new_failed

    if not new_failed:
        Path(checkpoint_path).unlink(missing_ok=True)
        console.print_info(f"All tasks succeeded. Removed checkpoint: {checkpoint_path}")

    model_statistics = global_processor.calculate_statistics(all_results_list)
    report = build_report_from_batch_results(
        "batch",
        all_results_list,
        metadata={"config_file": config_file, "checkpoint": checkpoint_path},
    )

    return BatchRunResult(
        all_results=all_results_list,
        model_statistics=model_statistics,
        validated_models=validation.validated,
        skipped_models=validation.skipped,
        original_tasks=tasks,
        batch_mode=batch_mode,
        batch_config=batch_config,
        config_file=config_file,
        report=report,
    )


@dataclass
class BatchExportResult:
    """Result of exporting batch results to disk."""

    exported_paths: list[str]
    export_mode: str  # "single" | "separate" | "split"


class BatchService:
    """High-level service for batch result statistics and export."""

    def __init__(
        self,
        run_result: BatchRunResult,
        batch_cfg: UnifiedBatchConfig,
        *,
        pricing_map: PricingMap | None = None,
    ) -> None:
        """Initialize the batch service.

        Args:
            run_result: Result of a batch run.
            batch_cfg: ``batch`` section from the unified config.
            pricing_map: Optional pricing data for cost estimates.
        """
        self.run_result = run_result
        self.batch_cfg = batch_cfg
        self.pricing_map = pricing_map or {}

    def _group_results_by_model(self) -> dict[str, list[BatchResult]]:
        """Group results by provider/model key."""
        grouped: dict[str, list[BatchResult]] = {}
        for result in self.run_result.all_results:
            model_key = f"{result.model_settings.provider}/{result.model_settings.model}"
            grouped.setdefault(model_key, []).append(result)
        return grouped

    def _combined_results(self) -> list[BatchResult]:
        """Return all results flattened into a single list."""
        combined: list[BatchResult] = []
        for results in self._group_results_by_model().values():
            combined.extend(results)
        return combined

    def print_statistics(self) -> None:
        """Print per-model statistics and cost estimates to the console."""
        console.print()
        for model_key, statistics in self.run_result.model_statistics.items():
            console.print(f"[bold]Statistics for {model_key}:[/bold]")
            console.print(f"  Total Tasks: {statistics.total_tasks}")
            console.print(f"  Successful: {statistics.successful_tasks}")
            console.print(f"  Failed: {statistics.failed_tasks}")
            if statistics.successful_tasks > 0:
                success_rate = statistics.successful_tasks / statistics.total_tasks * 100
                console.print(f"  Success Rate: {success_rate:.1f}%")
                console.print(f"  Average Latency: {statistics.average_latency:.2f}s")
                console.print(
                    f"  Total Tokens: {statistics.total_input_tokens + statistics.total_output_tokens:,}"
                )
                parts = model_key.split("/", 1)
                prov, mod = parts[0], parts[1] if len(parts) > 1 else ""
                console.print(
                    format_cost_estimate(
                        prov,
                        mod,
                        statistics.total_input_tokens,
                        statistics.total_output_tokens,
                        self.pricing_map,
                    )
                )

    def print_skipped_providers(self) -> None:
        """Print a warning listing any skipped providers."""
        if not self.run_result.skipped_models:
            return
        console.print()
        console.print_warning(f"Skipped {len(self.run_result.skipped_models)} provider(s):")
        for skipped in self.run_result.skipped_models:
            console.print(f"  - {skipped}")

    def export_results(
        self,
        output: str | None,
        output_format: str,
        *,
        split: bool = False,
        separate_files: bool = False,
    ) -> BatchExportResult:
        """Export batch results according to the selected mode.

        Args:
            output: Explicit output path or directory.
            output_format: Output format (json, yaml, csv, markdown).
            split: Export one file per original task.
            separate_files: Export one file per model when multiple models were used.

        Returns:
            BatchExportResult with exported paths and mode.

        Raises:
            ValueError: If no results are available to export.
        """
        grouped = self._group_results_by_model()

        if not grouped:
            raise ValueError("No providers were successfully processed. Cannot generate results.")

        if split:
            return self._export_split(output, output_format, grouped)

        if separate_files and len(self.run_result.validated_models) > 1:
            return self._export_separate(output, output_format, grouped)

        return self._export_single(output, output_format)

    def _export_split(
        self,
        output: str | None,
        output_format: str,
        grouped: dict[str, list[BatchResult]],
    ) -> BatchExportResult:
        """Export split files: one file per original task."""
        combined_results: list[BatchResult] = []
        for results in grouped.values():
            combined_results.extend(results)

        task_groups: dict[tuple[str, str, str | None], list[BatchResult]] = defaultdict(list)
        for result in combined_results:
            task_key = (result.prompt, result.content, result.output_filename)
            task_groups[task_key].append(result)

        num_original_tasks = len(self.run_result.original_tasks)
        deduped_results: list[BatchResult] = []
        for task_key in sorted(
            task_groups.keys(),
            key=lambda k: min(r.task_id % num_original_tasks for r in task_groups[k]),
        ):
            task_results = sorted(task_groups[task_key], key=lambda r: r.task_id)
            deduped_results.append(task_results[0])

        if output:
            output_path_obj = Path(output)
            if output_path_obj.exists() and output_path_obj.is_file():
                raise ValueError(
                    f"Output path '{output}' is a file. "
                    "When using --split, output must be a directory."
                )
            output_dir = output
        else:
            config_file_path = Path(self.run_result.config_file)
            output_dir = str(config_file_path.parent / self.batch_cfg.batch_output_dir)

        exported_files = BatchResultExporter.export_split_files(
            deduped_results, output_dir, self.run_result.batch_mode
        )
        console.print()
        console.print_success(f"Results exported to {len(exported_files)} files in: {output_dir}")
        for file_path in exported_files:
            console.print(f"  - {file_path}")
        return BatchExportResult(exported_paths=exported_files, export_mode="split")

    def _export_separate(
        self,
        output: str | None,
        output_format: str,
        grouped: dict[str, list[BatchResult]],
    ) -> BatchExportResult:
        """Export separate files per model."""
        output_dir = output or self.batch_cfg.batch_results_dir
        exported_files = BatchResultExporter.export_multiple_models(
            grouped,
            self.run_result.model_statistics,
            output_dir,
            output_format,
            self.run_result.batch_mode,
        )
        console.print()
        console.print_success(f"Results exported to {len(exported_files)} files:")
        for file_path in exported_files:
            console.print(f"  - {file_path}")
        return BatchExportResult(exported_paths=exported_files, export_mode="separate")

    def _export_single(self, output: str | None, output_format: str) -> BatchExportResult:
        """Export all results to a single file."""
        combined_results = self._combined_results()
        combined_stats = BatchStatistics(total_tasks=len(combined_results))
        combined_stats.successful_tasks = sum(
            stats.successful_tasks for stats in self.run_result.model_statistics.values()
        )
        combined_stats.failed_tasks = sum(
            stats.failed_tasks for stats in self.run_result.model_statistics.values()
        )
        combined_stats.total_latency = sum(
            stats.total_latency for stats in self.run_result.model_statistics.values()
        )
        if combined_stats.successful_tasks > 0:
            combined_stats.average_latency = (
                combined_stats.total_latency / combined_stats.successful_tasks
            )
            combined_stats.total_input_tokens = sum(
                stats.total_input_tokens for stats in self.run_result.model_statistics.values()
            )
            combined_stats.total_output_tokens = sum(
                stats.total_output_tokens for stats in self.run_result.model_statistics.values()
            )

        if output:
            output_path = output
        else:
            config_file_path = Path(self.run_result.config_file)
            output_path = str(
                config_file_path.parent
                / f"{config_file_path.stem}{self.batch_cfg.output_suffix}.{output_format}"
            )

        exporter = BatchResultExporter(combined_results, combined_stats, self.run_result.batch_mode)
        exported_file = exporter.export(output_path, output_format)
        console.print()
        console.print_success(f"Results exported to: {exported_file}")
        return BatchExportResult(exported_paths=[exported_file], export_mode="single")

    def export_report(self, report_path: str | None) -> str | None:
        """Export the execution report to ``report_path`` if one is available.

        Args:
            report_path: Destination path for the JSON report.

        Returns:
            The exported path, or ``None`` if no report was generated or no path
            was requested.
        """
        if not report_path or self.run_result.report is None:
            return None
        self.run_result.report.to_json_file(report_path)
        console.print_info(f"Execution report saved to: {report_path}")
        return report_path
