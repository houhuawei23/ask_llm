"""Typer command `trans` (split from former cli.py)."""

from __future__ import annotations

import time
from typing import Annotated

import typer
from loguru import logger

try:
    from ask_llm.utils import (
        engine_facade as _engine_facade,  # noqa: F401 — fail fast if engine missing
    )
except ImportError:
    from ask_llm.utils.console import console

    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise

from ask_llm.cli.common import paid_command_prelude
from ask_llm.cli.errors import cli_errors
from ask_llm.core.constants import MAX_CONCURRENCY
from ask_llm.services.translation_service import (
    TranslationOptions,
    TranslationService,
)
from ask_llm.utils.console import console


def trans(
    files: Annotated[
        list[str],
        typer.Argument(help="Input file(s) to translate (supports glob patterns)"),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory path (default: auto-generated)",
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to default_config.yml",
        ),
    ] = None,
    target_lang: Annotated[
        str | None,
        typer.Option(
            "--target-lang",
            "-t",
            help="Target language code (from default_config.yml if not set). "
            "Note: in trans, -t/-T/-s mean target-lang/threads/source-lang — "
            "other commands use them for temperature/type/system.",
        ),
    ] = None,
    source_lang: Annotated[
        str | None,
        typer.Option(
            "--source-lang",
            "-s",
            help="Source language code (default: auto-detect)",
        ),
    ] = None,
    threads: Annotated[
        int | None,
        typer.Option(
            "--threads",
            "-T",
            help="Max concurrent API calls per file (from default_config.yml if not set)",
            min=1,
            max=MAX_CONCURRENCY,
        ),
    ] = None,
    max_parallel_files: Annotated[
        int | None,
        typer.Option(
            "--max-parallel-files",
            help="Max files to translate in parallel (default: 3)",
            min=1,
            max=MAX_CONCURRENCY,
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
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-a",
            help="API provider to use",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model name to use",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing output file",
        ),
    ] = False,
    preserve_format: Annotated[
        bool,
        typer.Option(
            "--preserve-format/--no-preserve-format",
            help="Preserve original formatting (default: True)",
        ),
    ] = True,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Stream translation progress to console (progress bars only)",
        ),
    ] = False,
    stream_api: Annotated[
        bool,
        typer.Option(
            "--stream-api/--no-stream-api",
            help="Use streaming API calls; disable for higher batch throughput",
        ),
    ] = True,
    prompt_file: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Path to prompt template file (supports @ prefix for project-relative paths, e.g., @prompts/tech-paper-trans.md)",
        ),
    ] = None,
    providers_pricing: Annotated[
        str | None,
        typer.Option(
            "--providers-pricing",
            help="Path to providers.yml (pricing_per_million_tokens). "
            "Default search: ASK_LLM_PROVIDERS_YML, package root, ~/.config/ask_llm/providers.yml",
        ),
    ] = None,
    no_balance_chunks: Annotated[
        bool,
        typer.Option(
            "--no-balance-chunks",
            help="Disable token-based chunk rebalancing (structure-only splitting)",
        ),
    ] = False,
    max_chunk_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-chunk-tokens",
            help="Max estimated body tokens per chunk after rebalance (default: config)",
            min=256,
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            help="Sampling temperature override (E4/2.25; default: config translation.temperature)",
        ),
    ] = None,
    include_original: Annotated[
        bool | None,
        typer.Option(
            "--include-original/--no-include-original",
            help="Keep the original text next to each translated chunk (E4/2.25; default: config)",
        ),
    ] = None,
    skip_api_key_check: Annotated[
        bool,
        typer.Option(
            "--skip-api-key-check",
            help="Skip API key presence check (not recommended)",
        ),
    ] = False,
    glossary: Annotated[
        str | None,
        typer.Option(
            "--glossary",
            "-g",
            help="Path to glossary file (YAML map or JSONL {src,tgt})",
        ),
    ] = None,
    translated_suffix: Annotated[
        str | None,
        typer.Option(
            "--translated-suffix",
            help="Suffix for translated output files (default: config file.translated_suffix)",
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            is_flag=False,
            flag_value="",
            help=(
                "Resume translation from per-file checkpoints. Takes no value "
                "here (checkpoints live next to each output file); only "
                "batch/format accept an explicit --resume PATH."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Estimate chunks, tokens and cost without any API call",
        ),
    ] = False,
    report: Annotated[
        str | None,
        typer.Option(
            "--report",
            help="Export a structured execution report (JSON) to the given path",
        ),
    ] = None,
) -> None:
    """
    Translate text files using LLM API.

    Supports plain text (.txt), Markdown (.md), and Jupyter notebooks (.ipynb).
    For .ipynb files: only markdown cells are translated, code cells are preserved.
    Uses intelligent text splitting to handle long documents.

    Examples:
        ask-llm trans document.txt
        ask-llm trans /path/to/dir/ -o translated/
        ask-llm trans *.md -o translated/
        ask-llm trans notebook.ipynb -o translated/
        ask-llm trans file.txt -t en -s zh --threads 10
        ask-llm trans doc.md -m gpt-4 --preserve-format
        ask-llm trans paper.md -p @prompts/tech-paper-trans.md
        ask-llm trans ./posts/ --max-parallel-files 5
    """
    _t0 = time.perf_counter()
    try:
        with cli_errors("trans"):
            prelude = paid_command_prelude(
                config,
                provider=provider,
                model=model,
                temperature=lambda lr: lr.unified_config.translation.temperature,
                pricing_path=providers_pricing,
                skip_api_key_check=skip_api_key_check or dry_run,
            )
            trans_cfg = prelude.load_result.unified_config.translation

            if resume:
                console.print_error(
                    "--resume for trans takes no value (per-file checkpoints "
                    "are auto-named next to each output file)."
                )
                raise typer.Exit(2)
            resume_enabled = resume is not None

            if dry_run:
                from ask_llm.core.translator import Translator
                from ask_llm.services.dry_run import estimate_translation_run
                from ask_llm.utils.path_resolver import resolve_trans_input_paths

                input_paths = resolve_trans_input_paths(
                    files,
                    trans_cfg.translatable_extensions,
                    trans_cfg.recursive_dir,
                )
                if not input_paths:
                    console.print_error("No input files matched.")
                    raise typer.Exit(1)
                # M10/2.25: the glossary widens the prompt and shrinks the
                # chunk budget — the dry run must size chunks the same way the
                # paid run does.
                glossary_pairs = Translator.load_glossary(glossary) if glossary else []
                dry_report = estimate_translation_run(
                    input_paths,
                    prelude.model,
                    prelude.provider,
                    target_language=target_lang or trans_cfg.target_language,
                    source_language=trans_cfg.source_language
                    if source_lang is None
                    else source_lang,
                    style=trans_cfg.style,
                    prompt_file=prompt_file,
                    glossary_pairs=glossary_pairs,
                    max_chunk_tokens=(
                        max_chunk_tokens
                        if max_chunk_tokens is not None
                        else trans_cfg.max_chunk_tokens
                    ),
                    balance_chunks=trans_cfg.balance_translation_chunks and not no_balance_chunks,
                    pricing_map=prelude.pricing_map,
                )
                for line in dry_report.render(pricing_source=prelude.pricing_source):
                    console.print(line)
                return

            options = TranslationOptions(
                target_language=target_lang or trans_cfg.target_language,
                source_language=trans_cfg.source_language if source_lang is None else source_lang,
                style=trans_cfg.style,
                threads=threads if threads is not None else trans_cfg.max_concurrent_api_calls,
                max_parallel_files=(
                    max_parallel_files
                    if max_parallel_files is not None
                    else trans_cfg.max_parallel_files
                ),
                retries=retries if retries is not None else trans_cfg.retries,
                balance_translation_chunks=trans_cfg.balance_translation_chunks
                and not no_balance_chunks,
                max_chunk_tokens=(
                    max_chunk_tokens if max_chunk_tokens is not None else trans_cfg.max_chunk_tokens
                ),
                max_output_tokens=trans_cfg.max_output_tokens,
                preserve_format=preserve_format,
                include_original=(
                    include_original if include_original is not None else trans_cfg.include_original
                ),
                temperature=temperature if temperature is not None else trans_cfg.temperature,
                translatable_extensions=trans_cfg.translatable_extensions,
                recursive_dir=trans_cfg.recursive_dir,
                prompt_file=prompt_file,
                resume=resume_enabled,
            )

            service = TranslationService(
                config_manager=prelude.config_manager,
                unified_config=prelude.load_result.unified_config,
                provider=prelude.provider,
                model=prelude.model,
                pricing_map=prelude.pricing_map,
                pricing_source=prelude.pricing_source,
            )

            session_result = service.translate_files(
                files,
                options,
                output=output,
                force=force,
                stream=stream,
                stream_api=stream_api,
                glossary=glossary,
                translated_suffix=translated_suffix,
            )
            service.export_report(report, session_result)

            if session_result.failed_files > 0:
                partial_note = (
                    f" ({session_result.partial_files} partial)"
                    if session_result.partial_files
                    else ""
                )
                console.print_error(
                    f"Translation finished with {session_result.failed_files} failed "
                    f"file(s){partial_note}"
                )
                raise typer.Exit(1)

    finally:
        logger.debug("trans CLI wall time: {:.2f}s", time.perf_counter() - _t0)
