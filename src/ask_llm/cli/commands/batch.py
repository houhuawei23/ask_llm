"""Typer command `batch` (split from former cli.py)."""

from __future__ import annotations

import time
from pathlib import Path

import typer
from loguru import logger
from typing_extensions import Annotated

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.config.cli_session import (
    load_cli_session,
)
from ask_llm.core.batch import (
    BatchTask,
    ModelConfig,
)
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.utils.api_key_gate import (
    api_key_is_missing_or_unresolved,
    ensure_api_key_for_provider,
    require_resolved_api_key,
)
from ask_llm.utils.console import console

try:
    from llm_engine import create_provider_adapter
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise


def batch(
    config_file: Annotated[
        str,
        typer.Argument(help="Batch configuration file path (YAML format)"),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory path (default: auto-generated)",
        ),
    ] = None,
    output_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format: json, yaml, csv, or markdown (auto-detected from file extension if not specified)",
        ),
    ] = None,
    threads: Annotated[
        int | None,
        typer.Option(
            "--threads",
            "-t",
            help="Number of concurrent threads (from default_config.yml if not set)",
            min=1,
            max=50,
        ),
    ] = None,
    retries: Annotated[
        int | None,
        typer.Option(
            "--retries",
            "-r",
            help="Maximum number of retries (from default_config.yml if not set)",
            min=0,
            max=10,
        ),
    ] = None,
    config_path: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path",
        ),
    ] = None,
    separate_files: Annotated[
        bool,
        typer.Option(
            "--separate-files",
            help="Save results in separate files per model",
        ),
    ] = False,
    split: Annotated[
        bool,
        typer.Option(
            "--split",
            help="Split results into separate files (one file per task, content only)",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable verbose output with detailed API call information",
        ),
    ] = False,
    skip_api_key_check: Annotated[
        bool,
        typer.Option(
            "--skip-api-key-check",
            help="Skip API key presence check (not recommended)",
        ),
    ] = False,
) -> None:
    """
    Process batch tasks from YAML configuration file.

    Supports two configuration formats:
    1. prompt-contents.yml: Same prompt with multiple contents
    2. prompt-content-pairs.yml: Multiple (prompt, content) pairs

    Examples:
        ask-llm batch batch-examples/prompt-contents.yml
        ask-llm batch batch-examples/prompt-content-pairs.yml -o results.json -f json
        ask-llm batch config.yml --threads 10 --retries 5
    """
    try:
        _t0 = time.perf_counter()
        from ask_llm.core.batch import (
            BatchStatistics,
            BatchTask,
        )
        from ask_llm.utils.batch_exporter import BatchResultExporter
        from ask_llm.utils.batch_loader import BatchConfigLoader
        from ask_llm.utils.interactive_config import InteractiveConfigHelper
        from ask_llm.utils.pricing import format_cost_estimate, load_providers_pricing

        # Setup console with verbose mode
        if verbose:
            console.setup(quiet=False, debug=True)
        else:
            console.setup(quiet=False, debug=False)

        # Load configuration first (required for batch defaults)
        load_result, config_manager = load_cli_session(config_path)
        batch_cfg = load_result.unified_config.batch
        pricing_map, pricing_source = load_providers_pricing(None)
        effective_threads = threads if threads is not None else batch_cfg.threads
        effective_retries = retries if retries is not None else batch_cfg.retries

        # Load batch configuration
        console.print_info(f"Loading batch configuration from: {config_file}")
        batch_config = BatchConfigLoader.load(config_file)

        tasks = batch_config["tasks"]
        provider_models = batch_config.get("provider_models", [])

        # Auto-detect output format from file extension if not specified
        if output_format is None and output:
            output_path_obj = Path(output)
            suffix = output_path_obj.suffix.lower()
            # Map file extensions to formats
            extension_to_format = {
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".csv": "csv",
                ".md": "markdown",
                ".markdown": "markdown",
            }
            output_format = extension_to_format.get(suffix, "json")
            logger.debug(
                f"Auto-detected output format '{output_format}' from file extension '{suffix}'"
            )
        elif output_format is None:
            output_format = batch_cfg.default_output_format

        console.print_success(f"Loaded {len(tasks)} tasks from configuration")

        # If no models specified, use interactive selection
        if not provider_models:
            console.print_info("No models specified in configuration. Using interactive selection.")
            helper = InteractiveConfigHelper(config_manager)
            provider_models = helper.select_provider_and_models(allow_multiple=True)

        batch_mode = batch_config.get("mode", batch_cfg.mode)
        app_config = load_result.app_config

        unique_providers = sorted({m.provider for m in provider_models})
        for pname in unique_providers:
            strict_gate = ensure_api_key_for_provider(
                config_manager,
                pname,
                skip_api_key_check=skip_api_key_check,
            )
            if strict_gate:
                require_resolved_api_key(config_manager, pname)

        # Step 1: Validate all models and test connections
        console.print()
        console.print("[bold]Validating models and testing connections...[/bold]")
        validated_models: list[ModelConfig] = []
        skipped_providers: list[str] = []

        for model_config in provider_models:
            model_key = f"{model_config.provider}/{model_config.model}"
            console.print(f"  Checking {model_key}...", end=" ")

            try:
                # Check if provider exists
                if model_config.provider not in app_config.providers:
                    console.print("[red]✗[/red] Provider not found")
                    skipped_providers.append(model_key)
                    continue

                # Get provider config
                provider_config = app_config.providers[model_config.provider]

                # Check API key
                if api_key_is_missing_or_unresolved(provider_config.api_key):
                    console.print("[red]✗[/red] API key not configured")
                    skipped_providers.append(model_key)
                    continue

                # Check if model is available for this provider
                if provider_config.models and model_config.model not in provider_config.models:
                    console.print(
                        f"[red]✗[/red] Model not available. Available: {', '.join(provider_config.models)}"
                    )
                    skipped_providers.append(model_key)
                    continue

                # Set provider in config manager
                config_manager.set_provider(model_config.provider)

                # Apply model-specific overrides
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

                # Test connection
                try:
                    test_provider = create_provider_adapter(
                        provider_config_with_overrides, default_model=default_model
                    )
                    success, message, latency = test_provider.test_connection()

                    if not success:
                        console.print(f"[red]✗[/red] Connection test failed: {message}")
                        skipped_providers.append(model_key)
                        continue

                    console.print(f"[green]✓ ({latency:.2f}s)[/green]")
                    validated_models.append(model_config)

                except Exception as e:
                    console.print(f"[red]✗[/red] Connection test error: {e}")
                    skipped_providers.append(model_key)
                    continue

            except Exception as e:
                console.print(f"[red]✗[/red] Error: {e}")
                skipped_providers.append(model_key)
                continue

        # Display summary of skipped providers
        if skipped_providers:
            console.print()
            console.print_warning(f"Skipped {len(skipped_providers)} provider(s):")
            for skipped in skipped_providers:
                console.print(f"  - {skipped}")

        # Check if we have any validated models
        if not validated_models:
            console.print_error("No providers were successfully validated. Cannot process tasks.")
            raise typer.Exit(1)

        # Step 2: Create global task list (each task with model_config)
        console.print()
        console.print(
            f"[bold]Processing {len(validated_models)} model(s) with {len(tasks)} task(s) each...[/bold]"
        )
        global_tasks: list[BatchTask] = []
        task_id_counter = 0

        for model_config in validated_models:
            for original_task in tasks:
                # Create a new task with task_model_config attached
                # Preserve original task_id for split mode filename generation
                # Use a composite ID: (model_index * num_tasks + original_task_id) for unique identification
                # But for split mode, we'll use original_task_id directly
                global_task = BatchTask(
                    task_id=task_id_counter,
                    prompt=original_task.prompt,
                    content=original_task.content,
                    output_filename=original_task.output_filename,  # Preserve output_filename
                    task_model_config=model_config,
                )
                global_tasks.append(global_task)
                task_id_counter += 1

        all_results_list, global_processor = run_global_batch_tasks(
            global_tasks,
            config_manager,
            max_workers=effective_threads,
            max_retries=effective_retries,
            retry_delay=batch_cfg.retry_delay,
            retry_delay_max=batch_cfg.retry_delay_max,
            verbose=verbose,
            show_progress=True,
            clamp_workers_to_task_count=False,
        )

        # Step 4: Group results by model and calculate statistics
        all_results: dict[str, list] = {}
        all_statistics: dict[str, BatchStatistics] = {}

        # Group results by model
        for result in all_results_list:
            model_key = f"{result.model_settings.provider}/{result.model_settings.model}"
            if model_key not in all_results:
                all_results[model_key] = []
            all_results[model_key].append(result)

        # Calculate statistics for each model
        model_statistics = global_processor.calculate_statistics(all_results_list)
        all_statistics = model_statistics

        # Display statistics for each model
        console.print()
        for model_key, statistics in all_statistics.items():
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
                        pricing_map,
                        pricing_source=pricing_source,
                    )
                )

        # Display summary of skipped providers
        if skipped_providers:
            console.print()
            console.print_warning(f"Skipped {len(skipped_providers)} provider(s):")
            for skipped in skipped_providers:
                console.print(f"  - {skipped}")

        # Check if we have any results to export
        if not all_results:
            console.print_error(
                "No providers were successfully processed. Cannot generate results."
            )
            raise typer.Exit(1)

        # Handle split mode: export each task to a separate file
        if split:
            # Combine all results from all models
            combined_results = []
            for results in all_results.values():
                combined_results.extend(results)

            # In split mode with multiple models, group results by original task
            # and use the first result for each task
            # Group by (prompt, content, output_filename) to identify original tasks
            from collections import defaultdict

            task_groups: dict[tuple[str, str, str | None], list] = defaultdict(list)
            for result in combined_results:
                task_key = (result.prompt, result.content, result.output_filename)
                task_groups[task_key].append(result)

            # Use the first result for each original task
            # Sort by minimum task_id to maintain original order
            # Since task_id is assigned sequentially (0, 1, 2, ...) for each model,
            # we can recover original task index by: task_id % num_original_tasks
            num_original_tasks = len(tasks)
            combined_results = []
            for task_key in sorted(
                task_groups.keys(),
                key=lambda k: min(r.task_id % num_original_tasks for r in task_groups[k]),
            ):
                # Use the first result (lowest task_id) for each original task
                task_results = sorted(task_groups[task_key], key=lambda r: r.task_id)
                combined_results.append(task_results[0])

            # Determine output directory
            if output:
                output_dir = output
                # Validate that output is a directory (not a file)
                output_path_obj = Path(output_dir)
                if output_path_obj.exists() and output_path_obj.is_file():
                    console.print_error(
                        f"Output path '{output_dir}' is a file. "
                        "When using --split, output must be a directory."
                    )
                    raise typer.Exit(1)
            else:
                # Default output directory
                config_file_path = Path(config_file)
                output_dir = str(config_file_path.parent / batch_cfg.batch_output_dir)

            # Export split files
            exported_files = BatchResultExporter.export_split_files(
                combined_results, output_dir, batch_mode
            )
            console.print()
            console.print_success(
                f"Results exported to {len(exported_files)} files in: {output_dir}"
            )
            for file_path in exported_files:
                console.print(f"  - {file_path}")
            return

        # Export results
        if separate_files and len(validated_models) > 1:
            # Export to separate files per model
            output_dir = output or batch_cfg.batch_results_dir
            exported_files = BatchResultExporter.export_multiple_models(
                all_results, all_statistics, output_dir, output_format, batch_mode
            )
            console.print()
            console.print_success(f"Results exported to {len(exported_files)} files:")
            for file_path in exported_files:
                console.print(f"  - {file_path}")
        else:
            # Export all results to a single file
            # Combine all results
            combined_results = []
            for results in all_results.values():
                combined_results.extend(results)

            # Calculate combined statistics
            combined_stats = BatchStatistics(total_tasks=len(combined_results))
            combined_stats.successful_tasks = sum(
                stats.successful_tasks for stats in all_statistics.values()
            )
            combined_stats.failed_tasks = sum(
                stats.failed_tasks for stats in all_statistics.values()
            )
            combined_stats.total_latency = sum(
                stats.total_latency for stats in all_statistics.values()
            )
            if combined_stats.successful_tasks > 0:
                combined_stats.average_latency = (
                    combined_stats.total_latency / combined_stats.successful_tasks
                )
                combined_stats.total_input_tokens = sum(
                    stats.total_input_tokens for stats in all_statistics.values()
                )
                combined_stats.total_output_tokens = sum(
                    stats.total_output_tokens for stats in all_statistics.values()
                )

            # Generate output path
            if output:
                output_path = output
            else:
                config_file_path = Path(config_file)
                output_path = str(
                    config_file_path.parent
                    / f"{config_file_path.stem}{batch_cfg.output_suffix}.{output_format}"
                )

            exporter = BatchResultExporter(combined_results, combined_stats, batch_mode)
            exported_file = exporter.export(output_path, output_format)
            console.print()
            console.print_success(f"Results exported to: {exported_file}")

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\nBatch processing interrupted by user")
        raise typer.Exit(1) from None
    except Exception as e:
        raise_unexpected_cli_error("batch", e)
    finally:
        logger.debug("batch CLI wall time: {:.2f}s", time.perf_counter() - _t0)
