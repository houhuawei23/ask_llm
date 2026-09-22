"""Batch processing orchestration service.

Moves the core batch workflow (load config, validate models, build tasks,
run them through GlobalBatchProcessor) out of the CLI command so the command
module stays focused on argument parsing, output formatting and user-facing
messages.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.config.unified_config import BatchConfig as UnifiedBatchConfig
from ask_llm.core.batch_models import BatchResult, BatchStatistics, BatchTask, ModelConfig
from ask_llm.core.command_runner import compute_checkpoint_digest, run_with_checkpoint
from ask_llm.core.execution_report import ExecutionReport, build_report_from_batch_results
from ask_llm.core.models import AppConfig
from ask_llm.utils.api_key_gate import (
    api_key_is_missing_or_unresolved,
    ensure_resolved_provider_keys,
)
from ask_llm.utils.batch_exporter import BatchResultExporter
from ask_llm.utils.batch_loader import BatchConfigLoader
from ask_llm.utils.console import console
from ask_llm.utils.fallback_chain import build_fallback_chain
from ask_llm.utils.interactive_config import InteractiveConfigHelper
from ask_llm.utils.pricing import format_cost_estimate
from ask_llm.utils.provider_cache import ProviderAdapterCache

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
    *,
    max_workers: int = 8,
) -> _ValidationResult:
    """Validate provider/model list and test connections.

    Plan 5.6: the connection probes fan out through a small thread pool — a
    dead endpoint no longer stalls the whole validation stage for its full
    timeout. Results (and console lines) preserve the input order.

    Builds each overridden provider view on a detached ``model_copy`` — the
    loop must never ``set_provider`` / ``apply_overrides`` on the *shared*
    manager (audit 2.5): it used to return with the manager left pointing at
    the last validated model, its sampling overrides installed and the
    run-level model override polluted.
    """
    result = _ValidationResult()

    def _check(index: int, model_config: ModelConfig) -> tuple[int, str, bool, str]:
        """Run the checks for one model; returns (index, key, ok, note)."""
        model_key = f"{model_config.provider}/{model_config.model}"

        if model_config.provider not in app_config.providers:
            return index, model_key, False, "Provider not found"

        provider_config = app_config.providers[model_config.provider]

        if api_key_is_missing_or_unresolved(provider_config.api_key):
            return index, model_key, False, "API key not configured"

        if provider_config.models and model_config.model not in provider_config.models:
            return (
                index,
                model_key,
                False,
                f"Model not available. Available: {', '.join(provider_config.models)}",
            )

        overrides: dict[str, Any] = {}
        if model_config.temperature is not None:
            overrides["api_temperature"] = model_config.temperature
        if model_config.top_p is not None:
            overrides["api_top_p"] = model_config.top_p
        if model_config.max_tokens is not None:
            overrides["max_tokens"] = model_config.max_tokens
        provider_config_with_overrides = (
            provider_config.model_copy(update=overrides) if overrides else provider_config
        )

        default_model = model_config.model or (
            provider_config.models[0] if provider_config.models else ""
        )
        if not default_model:
            return index, model_key, False, "No model available for this provider"

        try:
            test_provider = ProviderAdapterCache.get(
                provider_config_with_overrides, default_model=default_model
            )
        except Exception as e:
            return index, model_key, False, f"Failed to create provider adapter: {e}"

        success, message, latency = test_provider.test_connection()
        if not success:
            return index, model_key, False, f"Connection test failed: {message}"
        return index, model_key, True, f"{latency:.2f}s"

    workers = max(1, min(max_workers, len(provider_models)))
    ordered: list[tuple[int, str, bool, str] | None] = [None] * len(provider_models)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ask-llm-validate") as pool:
        futures = {pool.submit(_check, i, mc): i for i, mc in enumerate(provider_models)}
        for future in as_completed(futures):
            checked: tuple[int, str, bool, str] = future.result()
            ordered[checked[0]] = checked
            _index, model_key, ok, note = checked
            if ok:
                console.print(f"  [green]✓[/green] {model_key} ({note})")
            else:
                console.print(f"  [red]✗[/red] {model_key}: {note}")

    # Preserve input order regardless of probe completion order.
    for entry in ordered:
        if entry is None:
            continue
        index, model_key, ok, _note = entry
        if ok:
            result.validated.append(provider_models[index])
        else:
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
    skip_validation: bool = False,
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
        skip_validation: Skip the model/connection validation stage entirely
            (plan 5.6) — every configured provider/model runs unvalidated;
            failures surface per task at execution time.
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
    if not skip_api_key_check:
        # Service-layer fail-fast (pure error, no interactive prompt / typer.Exit):
        # the interactive gate lives in the CLI bootstrap for single-provider
        # commands; batch resolves keys before spawning concurrent calls.
        ensure_resolved_provider_keys(config_manager, unique_providers)

    # Validate all models and test connections (skippable, plan 5.6).
    if skip_validation:
        console.print()
        console.print_warning("Skipping model validation (--skip-validation).")
        validation = _ValidationResult(validated=list(provider_models))
    else:
        console.print()
        console.print("[bold]Validating models and testing connections...[/bold]")
        validation = _validate_models(
            provider_models,
            app_config,
            config_manager,
        )

    # Skipped providers are reported once by the CLI via BatchService.print_skipped_providers().
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

    # Shared checkpoint lifecycle (P4.1): resume-filter -> run -> merge ->
    # save -> unlink on clean full success.
    checkpoint_path = resume_checkpoint_path or _default_batch_checkpoint_path(config_file)
    outcome = run_with_checkpoint(
        command="batch",
        config_digest=compute_checkpoint_digest(config_file, global_tasks),
        checkpoint_path=checkpoint_path,
        tasks=global_tasks,
        config_manager=config_manager,
        resume=bool(resume_checkpoint_path and Path(resume_checkpoint_path).exists()),
        max_retries=retries,
        retry_delay=retry_delay,
        retry_delay_max=retry_delay_max,
        max_workers=threads,
        verbose=verbose,
        show_progress=True,
        clamp_workers_to_task_count=False,
    )

    if outcome.all_previously_completed:
        console.print_info("All tasks already completed according to checkpoint.")
    elif outcome.interrupted:
        console.print_warning(
            f"Interrupted: progress saved to checkpoint {checkpoint_path}. "
            f"Re-run with --resume to continue."
        )
    elif outcome.checkpoint_deleted:
        console.print_info(f"All tasks succeeded. Removed checkpoint: {checkpoint_path}")

    all_results_list = outcome.results
    return _build_run_result(
        all_results_list,
        validation,
        tasks=tasks,
        batch_mode=batch_mode,
        batch_config=batch_config,
        config_file=config_file,
        checkpoint_path=checkpoint_path,
    )


def _build_run_result(
    all_results: list[BatchResult],
    validation: _ValidationResult,
    *,
    tasks: list[BatchTask],
    batch_mode: str,
    batch_config: dict[str, Any],
    config_file: str,
    checkpoint_path: str,
) -> BatchRunResult:
    """Aggregate run results into the export-ready :class:`BatchRunResult`."""
    model_statistics = BatchStatistics.from_results(all_results)
    report = build_report_from_batch_results(
        "batch",
        all_results,
        metadata={"config_file": config_file, "checkpoint": checkpoint_path},
    )
    return BatchRunResult(
        all_results=all_results,
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
        force: bool = False,
    ) -> BatchExportResult:
        """Export batch results according to the selected mode.

        Args:
            output: Explicit output path or directory.
            output_format: Output format (json, yaml, csv, markdown).
            split: Export one file per original task.
            separate_files: Export one file per model when multiple models were used.
            force: Overwrite existing output files (audit 4.5); without it an
                existing target raises ``FileExistsError``.

        Returns:
            BatchExportResult with exported paths and mode.

        Raises:
            ValueError: If no results are available to export.
        """
        grouped = self._group_results_by_model()

        if not grouped:
            raise ValueError("No providers were successfully processed. Cannot generate results.")

        if split:
            return self._export_split(output, output_format, grouped, force=force)

        if separate_files and len(self.run_result.validated_models) > 1:
            return self._export_separate(output, output_format, grouped, force=force)

        return self._export_single(output, output_format, force=force)

    def _export_split(
        self,
        output: str | None,
        output_format: str,
        grouped: dict[str, list[BatchResult]],
        *,
        force: bool = False,
    ) -> BatchExportResult:
        """Export split files: one file per (task, model) answer.

        Audit 2.5: the old code kept only the lowest task_id per (prompt,
        content, output_filename) group — silently discarding every other
        validated model's paid answer and collapsing duplicate tasks. Filename
        conflicts are resolved by ``export_split_files`` (``_N`` suffixes), so
        keeping every distinct result is safe and lossless; a warning notes
        when one task produced several files.
        """
        combined_results = [r for results in grouped.values() for r in results]

        task_groups: dict[tuple[str, str, str | None], list[BatchResult]] = defaultdict(list)
        for result in combined_results:
            task_key = (result.prompt, result.content, result.output_filename)
            task_groups[task_key].append(result)

        num_original_tasks = len(self.run_result.original_tasks)
        deduped_results: list[BatchResult] = []
        for task_key in sorted(
            task_groups.keys(),
            key=lambda k: (
                min(r.task_id % num_original_tasks for r in task_groups[k])
                if num_original_tasks
                else 0
            ),
        ):
            # Collapse exact duplicates (same task_id re-merged on resume);
            # keep every distinct (task, model) answer.
            seen_ids: set[int] = set()
            for result in sorted(task_groups[task_key], key=lambda r: r.task_id):
                if result.task_id in seen_ids:
                    continue
                seen_ids.add(result.task_id)
                deduped_results.append(result)
            if len(seen_ids) > 1:
                label = task_key[2] or (task_key[0][:40] + "…")
                console.print_warning(
                    f"Multiple model answers for task '{label}'; exported as {len(seen_ids)} files."
                )

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
            deduped_results, output_dir, self.run_result.batch_mode, force=force
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
        *,
        force: bool = False,
    ) -> BatchExportResult:
        """Export separate files per model."""
        output_dir = output or self.batch_cfg.batch_results_dir
        exported_files = BatchResultExporter.export_multiple_models(
            grouped,
            self.run_result.model_statistics,
            output_dir,
            output_format,
            self.run_result.batch_mode,
            force=force,
        )
        console.print()
        console.print_success(f"Results exported to {len(exported_files)} files:")
        for file_path in exported_files:
            console.print(f"  - {file_path}")
        return BatchExportResult(exported_paths=exported_files, export_mode="separate")

    def _export_single(
        self, output: str | None, output_format: str, *, force: bool = False
    ) -> BatchExportResult:
        """Export all results to a single file."""
        combined_results = self._combined_results()
        combined_stats = BatchStatistics.combined_from_results(combined_results)

        if output:
            output_path = output
        else:
            config_file_path = Path(self.run_result.config_file)
            output_path = str(
                config_file_path.parent
                / f"{config_file_path.stem}{self.batch_cfg.output_suffix}.{output_format}"
            )

        exporter = BatchResultExporter(combined_results, combined_stats, self.run_result.batch_mode)
        exported_file = exporter.export(output_path, output_format, force=force)
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
