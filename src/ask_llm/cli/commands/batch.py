"""Typer command `batch` (split from former cli.py)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.config.cli_session import load_cli_session
from ask_llm.services.batch_service import BatchService, run_batch_from_config
from ask_llm.utils.console import console
from ask_llm.utils.pricing import load_providers_pricing


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
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            "-R",
            help="Resume from a checkpoint file (auto-generated if omitted)",
        ),
    ] = None,
    fallback: Annotated[
        bool,
        typer.Option(
            "--fallback/--no-fallback",
            help="Enable fallback to alternate providers/models on failure",
        ),
    ] = True,
    report: Annotated[
        str | None,
        typer.Option(
            "--report",
            help="Export a structured execution report (JSON) to the given path",
        ),
    ] = None,
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

        load_result, config_manager = load_cli_session(config_path)
        batch_cfg = load_result.unified_config.batch
        pricing_map, pricing_source = load_providers_pricing(None)
        if pricing_source:
            console.print_info(f"API pricing loaded from: {pricing_source}")
        else:
            console.print_info(
                "No providers.yml with pricing found; token counts will still be shown, "
                "cost estimate unavailable (add pricing_per_million_tokens or use --providers-pricing)"
            )
        effective_threads = threads if threads is not None else batch_cfg.threads
        effective_retries = retries if retries is not None else batch_cfg.retries

        if output_format is None and output:
            output_path_obj = Path(output)
            suffix = output_path_obj.suffix.lower()
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

        if verbose:
            console.setup(quiet=False, debug=True)

        run_result = run_batch_from_config(
            config_file,
            load_result.app_config,
            config_manager,
            batch_cfg,
            output_format=output_format,
            threads=effective_threads,
            retries=effective_retries,
            retry_delay=batch_cfg.retry_delay,
            retry_delay_max=batch_cfg.retry_delay_max,
            skip_api_key_check=skip_api_key_check,
            verbose=verbose,
            resume_checkpoint_path=resume,
            use_fallback=fallback,
        )

        service = BatchService(
            run_result,
            batch_cfg,
            pricing_map=pricing_map,
        )

        service.print_statistics()
        service.print_skipped_providers()
        service.export_results(
            output,
            output_format,
            split=split,
            separate_files=separate_files,
        )
        service.export_report(report)

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
