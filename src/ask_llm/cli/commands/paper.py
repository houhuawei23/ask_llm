"""Typer command `paper` (split from former cli.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.config.cli_session import (
    apply_cli_overrides_and_gate_api_key,
    bootstrap_command,
)
from ask_llm.services.paper_service import PaperExplainOptions, PaperService
from ask_llm.utils.console import console

try:
    from ask_llm.utils import (
        engine_facade as _engine_facade,  # noqa: F401 — fail fast if engine missing
    )
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise


def paper(
    input_path: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="Path to a paper .md file or an arxiv2md-beta output directory",
        ),
    ],
    run_mode: Annotated[
        str,
        typer.Option(
            "--run",
            "-r",
            help="sections (per-section + meta), full (whole paper), or all",
        ),
    ] = "all",
    sections: Annotated[
        str | None,
        typer.Option(
            "--sections",
            "-s",
            help="Comma-separated keys: meta,abstract,...,appendices,full, combo:<id>, "
            "abstract:tpl-stem, combo:abstract_intro:tpl-stem, ...",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-a", help="API provider"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name"),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", "-t", help="Sampling temperature", min=0.0, max=2.0),
    ] = None,
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Configuration file path"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing explain/*.md files"),
    ] = False,
    metadata: Annotated[
        bool,
        typer.Option("--metadata", help="Include token/latency metadata in each output file"),
    ] = False,
    skip_api_key_check: Annotated[
        bool,
        typer.Option("--skip-api-key-check", help="Skip API key check"),
    ] = False,
    providers_pricing: Annotated[
        str | None,
        typer.Option(
            "--providers-pricing",
            help="Path to providers.yml (pricing_per_million_tokens). "
            "Default search: ASK_LLM_PROVIDERS_YML, ./providers.yml, package root, ~/.config/ask_llm/providers.yml",
        ),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            "-j",
            help="Parallel LLM calls (default: paper.concurrency in config; thread pool for I/O)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Show detected sections and token estimates without making API calls",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Skip sections whose output already exists",
        ),
    ] = False,
    pipeline: Annotated[
        str | None,
        typer.Option(
            "--pipeline",
            help="Path to paper-explain-pipeline.yml (overrides paper.pipeline_config in config)",
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
    Explain a paper: split Markdown by headings (or load arxiv2md-beta dir), call LLM per section,
    write results to ./explain/ next to the input file or directory.

    Prompt templates and job-key mapping are loaded from paper-explain-pipeline.yml (see paper.pipeline_config).

    Multiple sections run in parallel via GlobalBatchProcessor (same pipeline as ``trans``) when concurrency > 1.

    Examples:
        ask-llm paper -i paper.md --run all
        ask-llm paper -i output2/20170612-Arxiv-Attention-Is-All-You-Need --run sections
        ask-llm paper -i paper.md -r full
        ask-llm paper -i paper.md -j 8
    """
    try:
        path = Path(input_path).expanduser().resolve()
        if not path.exists():
            console.print_error(f"Path not found: {path}")
            raise typer.Exit(1)

        (
            load_result,
            config_manager,
            pricing_map,
            pricing_source,
            final_provider,
            final_model,
        ) = bootstrap_command(
            config_path,
            pricing_path=providers_pricing,
            cli_provider=provider,
            cli_model=model,
        )

        apply_cli_overrides_and_gate_api_key(
            config_manager,
            provider=final_provider,
            model=final_model,
            temperature=temperature,
            skip_api_key_check=skip_api_key_check,
        )
        paper_cfg = load_result.unified_config.paper
        workers = concurrency if concurrency is not None else paper_cfg.concurrency
        if workers < 1 or workers > 64:
            console.print_error("--concurrency / paper.concurrency must be between 1 and 64")
            raise typer.Exit(1)

        section_filter: set[str] | None = None
        if sections:
            section_filter = {x.strip().lower() for x in sections.split(",") if x.strip()}

        options = PaperExplainOptions(
            run_mode=run_mode,
            section_filter=section_filter,
            temperature=temperature,
            force=force,
            include_metadata=metadata,
            concurrency=workers,
            dry_run=dry_run,
            resume=resume,
            pipeline_path=pipeline,
            use_fallback=fallback,
        )

        service = PaperService(
            config_manager=config_manager,
            unified_config=load_result.unified_config,
            provider=final_provider,
            model=final_model,
            pricing_map=pricing_map,
            pricing_source=pricing_source,
            app_config=load_result.app_config,
        )

        try:
            session_result = service.explain_paper(path, options)
        finally:
            service.export_report(report)

        # P4.2: service returns statuses; the CLI owns exit codes.
        if session_result.status == "failed":
            console.print_error(f"Paper explain failed: {session_result.error}")
            raise typer.Exit(1)
        if session_result.status in ("dry_run", "nothing_to_do"):
            raise typer.Exit(0)

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except typer.Exit:
        # typer.Exit subclasses RuntimeError (click) — must re-raise before the
        # RuntimeError handler below swallows it as an "API error".
        raise
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        raise_unexpected_cli_error("paper", e)
