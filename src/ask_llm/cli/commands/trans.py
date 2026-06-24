"""Typer command `trans` (split from former cli.py)."""

from __future__ import annotations

import time
from typing import Annotated

import typer
from loguru import logger

try:
    import llm_engine  # noqa: F401 — fail fast if engine missing
except ImportError:
    from ask_llm.utils.console import console

    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.config.cli_session import (
    apply_cli_overrides_and_gate_api_key,
    load_cli_session,
    resolve_provider_and_model_or_exit,
)
from ask_llm.services.translation_service import (
    TranslationOptions,
    TranslationService,
)
from ask_llm.utils.console import console
from ask_llm.utils.pricing import load_providers_pricing


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
            help="Target language code (from default_config.yml if not set)",
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
            max=100,
        ),
    ] = None,
    max_parallel_files: Annotated[
        int | None,
        typer.Option(
            "--max-parallel-files",
            help="Max files to translate in parallel (default: 3)",
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
            "Default search: ASK_LLM_PROVIDERS_YML, ./providers.yml, package root, ~/.config/ask_llm/providers.yml",
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
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Resume translation from per-file checkpoints (default: False)",
        ),
    ] = False,
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
    try:
        _t0 = time.perf_counter()
        load_result, config_manager = load_cli_session(config)
        trans_cfg = load_result.unified_config.translation

        pricing_map, pricing_source = load_providers_pricing(providers_pricing)
        if pricing_source:
            console.print_info(f"API pricing loaded from: {pricing_source}")
        else:
            console.print_info(
                "No providers.yml with pricing found; token counts will still be shown, "
                "cost estimate unavailable (add pricing_per_million_tokens or use --providers-pricing)"
            )

        final_provider, final_model = resolve_provider_and_model_or_exit(
            config_manager,
            cli_provider=provider,
            cli_model=model,
        )

        apply_cli_overrides_and_gate_api_key(
            config_manager,
            provider=final_provider,
            model=final_model,
            temperature=trans_cfg.temperature,
            skip_api_key_check=skip_api_key_check,
        )

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
            min_chunk_merge_tokens=trans_cfg.min_chunk_merge_tokens,
            max_output_tokens=trans_cfg.max_output_tokens,
            preserve_format=preserve_format,
            include_original=trans_cfg.include_original,
            temperature=trans_cfg.temperature,
            translatable_extensions=trans_cfg.translatable_extensions,
            recursive_dir=trans_cfg.recursive_dir,
            prompt_file=prompt_file,
            resume=resume,
        )

        service = TranslationService(
            config_manager=config_manager,
            unified_config=load_result.unified_config,
            provider=final_provider,
            model=final_model,
            pricing_map=pricing_map,
            pricing_source=pricing_source,
        )

        service.translate_files(
            files,
            options,
            output=output,
            force=force,
            stream=stream,
            stream_api=stream_api,
            glossary=glossary,
            translated_suffix=translated_suffix,
        )

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
        console.print("\nTranslation interrupted by user")
        raise typer.Exit(1) from None
    except Exception as e:
        raise_unexpected_cli_error("trans", e)
    finally:
        logger.debug("trans CLI wall time: {:.2f}s", time.perf_counter() - _t0)
