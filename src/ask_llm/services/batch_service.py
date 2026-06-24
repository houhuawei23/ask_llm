"""Batch processing orchestration service.

Moves the core batch workflow (load config, validate models, build tasks,
run them through GlobalBatchProcessor) out of the CLI command so the command
module stays focused on argument parsing, output formatting and user-facing
messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig
from ask_llm.core.batch_models import BatchStatistics
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.models import AppConfig
from ask_llm.utils.api_key_gate import (
    api_key_is_missing_or_unresolved,
    ensure_api_key_for_provider,
    require_resolved_api_key,
)
from ask_llm.utils.batch_loader import BatchConfigLoader
from ask_llm.utils.console import console
from ask_llm.utils.interactive_config import InteractiveConfigHelper

try:
    from llm_engine import create_provider_adapter
except ImportError:  # pragma: no cover - guarded by package deps
    create_provider_adapter = None


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

            if create_provider_adapter is None:
                console.print("[red]✗[/red] llm_engine not available")
                result.skipped.append(model_key)
                continue

            test_provider = create_provider_adapter(
                provider_config_with_overrides, default_model=default_model
            )
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

    Returns:
        BatchRunResult with all results, statistics and metadata for export.

    Raises:
        typer.Exit: If no providers validate or no results are produced.
    """
    import typer

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
        console.print_error("No providers were successfully validated. Cannot process tasks.")
        raise typer.Exit(1)

    # Create global task list (each task with model_config)
    console.print()
    console.print(
        f"[bold]Processing {len(validation.validated)} model(s) with "
        f"{len(tasks)} task(s) each...[/bold]"
    )
    global_tasks: list[BatchTask] = []
    task_id_counter = 0

    for model_config in validation.validated:
        for original_task in tasks:
            global_task = BatchTask(
                task_id=task_id_counter,
                prompt=original_task.prompt,
                content=original_task.content,
                output_filename=original_task.output_filename,
                task_model_config=model_config,
            )
            global_tasks.append(global_task)
            task_id_counter += 1

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

    model_statistics = global_processor.calculate_statistics(all_results_list)

    return BatchRunResult(
        all_results=all_results_list,
        model_statistics=model_statistics,
        validated_models=validation.validated,
        skipped_models=validation.skipped,
        original_tasks=tasks,
        batch_mode=batch_mode,
        batch_config=batch_config,
    )
