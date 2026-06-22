"""Typer command `trans` (split from former cli.py)."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import typer
from loguru import logger
from typing_extensions import Annotated

from ask_llm.config.cli_session import (
    apply_cli_overrides_and_gate_api_key,
    load_cli_session,
)
from ask_llm.config.context import get_config
from ask_llm.core.batch import (
    BatchResult,
    BatchTask,
    ModelConfig,
)
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.text_splitter import TextChunk, TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.utils.chunk_balance import plain_text_chunks_by_tokens, rebalance_translation_chunks
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.pricing import format_cost_estimate, load_providers_pricing
from ask_llm.utils.translation_exporter import TranslationExporter

try:
    import llm_engine  # noqa: F401 — fail fast if engine missing
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise

from ask_llm.cli.common import (
    _is_directory_output,
    _process_notebook_translation,
    _resolve_trans_input_paths,
)
from ask_llm.cli.errors import raise_unexpected_cli_error


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
        app_config = load_result.app_config
        trans_cfg = load_result.unified_config.translation

        pricing_map, pricing_source = load_providers_pricing(providers_pricing)
        if pricing_source:
            console.print_info(f"API pricing loaded from: {pricing_source}")
        else:
            console.print_info(
                "No providers.yml with pricing found; token counts will still be shown, "
                "cost estimate unavailable (add pricing_per_million_tokens or use --providers-pricing)"
            )

        # Build trans config from default_config.yml, override with CLI
        effective_threads = threads if threads is not None else trans_cfg.max_concurrent_api_calls
        effective_max_parallel = (
            max_parallel_files if max_parallel_files is not None else trans_cfg.max_parallel_files
        )
        trans_config = SimpleNamespace(
            target_language=target_lang or trans_cfg.target_language,
            source_language=trans_cfg.source_language if source_lang is None else source_lang,
            style=trans_cfg.style,
            threads=effective_threads,
            max_parallel_files=effective_max_parallel,
            retries=retries if retries is not None else trans_cfg.retries,
            balance_translation_chunks=trans_cfg.balance_translation_chunks
            and not no_balance_chunks,
            max_chunk_tokens=max_chunk_tokens
            if max_chunk_tokens is not None
            else trans_cfg.max_chunk_tokens,
            min_chunk_merge_tokens=trans_cfg.min_chunk_merge_tokens,
            max_output_tokens=trans_cfg.max_output_tokens,
            preserve_format=preserve_format,
            include_original=trans_cfg.include_original,
            provider=provider,
            model=model,
            prompt_file=prompt_file,
            prompt_template=None,
            temperature=trans_cfg.temperature,
            translatable_extensions=trans_cfg.translatable_extensions,
            recursive_dir=trans_cfg.recursive_dir,
        )

        # Determine provider and model
        final_provider = trans_config.provider or app_config.default_provider
        final_model = trans_config.model or config_manager.get_default_model()

        if not final_provider:
            console.print_error(
                "No provider specified. Use --provider or configure default provider."
            )
            raise typer.Exit(1)

        if not final_model:
            console.print_error("No model specified. Use --model or configure default model.")
            raise typer.Exit(1)

        # Load glossary if provided
        glossary_pairs: list[tuple[str, str]] = []
        if glossary:
            glossary_pairs = Translator.load_glossary(glossary)
            console.print_info(
                f"Glossary: {len(glossary_pairs)} term pair(s) loaded from {glossary}"
            )

        apply_cli_overrides_and_gate_api_key(
            config_manager,
            provider=final_provider,
            model=final_model,
            temperature=trans_config.temperature,
            skip_api_key_check=skip_api_key_check,
        )

        # Resolve file patterns (supports directory, glob, and file paths)
        resolved_files = _resolve_trans_input_paths(
            files,
            translatable_extensions=trans_config.translatable_extensions,
            recursive_dir=trans_config.recursive_dir,
        )

        if not resolved_files:
            console.print_error("No files found to translate")
            raise typer.Exit(1)

        console.print_info(f"Found {len(resolved_files)} file(s) to translate")
        console.print_info(
            f"File-level concurrency: {trans_config.max_parallel_files} file(s) in parallel"
        )
        console.print_info(
            f"Chunk-level concurrency: up to {trans_config.threads} concurrent API call(s) per file"
        )

        # Determine whether the output target is a directory. Create it once
        # before parallel processing to avoid races between worker threads.
        output_is_dir = bool(
            output and _is_directory_output(output, files, len(resolved_files))
        )
        if output_is_dir:
            Path(output).mkdir(parents=True, exist_ok=True)

        @dataclass
        class _TextTranslationJob:
            """Internal container for a prepared text/markdown translation job."""

            file_path: str
            file_type: str
            chunks: list[TextChunk]
            tasks: list[BatchTask]
            output_path: str

        def _resolve_output_path(file_path: str) -> str:
            """Determine the output path for a single input file."""
            effective_translated_suffix = (
                translated_suffix
                if translated_suffix is not None
                else get_config().unified_config.file.translated_suffix
            )
            if output:
                output_path = output
                if output_is_dir:
                    input_file = Path(file_path)
                    output_name = (
                        f"{input_file.stem}{effective_translated_suffix}{input_file.suffix}"
                    )
                    output_path = str(Path(output) / output_name)
            else:
                output_path = FileHandler.generate_output_path(
                    file_path, suffix=effective_translated_suffix
                )
            return output_path

        def _prepare_text_file(file_path: str) -> _TextTranslationJob | None:
            """Read, split, and build translation tasks for a text/markdown file."""
            console.print()
            console.print(f"[bold]Preparing: {file_path}[/bold]")

            file_type = TextSplitter.detect_file_type(file_path)
            if file_type not in ("markdown", "text"):
                console.print_warning(
                    f"Unsupported file type: {Path(file_path).suffix}. "
                    f"Only .txt, .md, and .ipynb files are supported. Skipping."
                )
                return None

            try:
                content = FileHandler.read(file_path, show_progress=not stream)
            except Exception as e:
                console.print_error(f"Failed to read file {file_path}: {e}")
                return None

            if not content.strip():
                console.print_warning(f"File {file_path} is empty. Skipping.")
                return None

            if file_type == "markdown":
                chunks = MarkdownTokenSplitter(final_model, trans_config.max_chunk_tokens).split(
                    content
                )
            else:
                chunks = plain_text_chunks_by_tokens(
                    content, final_model, trans_config.max_chunk_tokens
                )

            if not chunks:
                console.print_warning(f"No chunks created from {file_path}. Skipping.")
                return None

            n_before = len(chunks)
            chunks = rebalance_translation_chunks(
                chunks,
                final_model,
                max_chunk_tokens=trans_config.max_chunk_tokens,
                min_merge_tokens=trans_config.min_chunk_merge_tokens,
                enabled=trans_config.balance_translation_chunks,
            )
            if len(chunks) != n_before:
                console.print_info(
                    f"Token rebalance: {n_before} -> {len(chunks)} chunk(s) "
                    f"(cap ≈ {trans_config.max_chunk_tokens} tokens)"
                )
            else:
                console.print_info(f"Split into {len(chunks)} chunk(s)")

            translator = Translator(
                target_language=trans_config.target_language,
                source_language=trans_config.source_language,
                style=trans_config.style,
                custom_prompt_template=trans_config.prompt_template,
                prompt_file=trans_config.prompt_file,
                glossary_pairs=glossary_pairs,
            )

            model_config = ModelConfig(
                provider=final_provider,
                model=final_model,
                temperature=trans_config.temperature,
                max_tokens=trans_config.max_output_tokens,
            )

            tasks = translator.create_translation_tasks(chunks, model_config)
            output_path = _resolve_output_path(file_path)

            return _TextTranslationJob(
                file_path=file_path,
                file_type=file_type,
                chunks=chunks,
                tasks=tasks,
                output_path=output_path,
            )

        def _process_notebook_file(file_path: str) -> tuple[int, int] | None:
            """Process a single Jupyter notebook (keeps its own internal chunk parallelism)."""
            console.print()
            console.print(f"[bold]Processing: {file_path}[/bold]")

            try:
                return _process_notebook_translation(
                    file_path=file_path,
                    output=output,
                    trans_config=trans_config,
                    config_manager=config_manager,
                    final_provider=final_provider,
                    final_model=final_model,
                    force=force,
                    stream=stream,
                    stream_api=stream_api,
                    pricing_map=pricing_map,
                    pricing_source=pricing_source,
                    output_is_dir=output_is_dir,
                )
            except FileExistsError:
                console.print_error("Output file already exists. Use --force to overwrite.")
            except Exception as e:
                console.print_error(f"Failed to translate notebook {file_path}: {e}")
                logger.exception("Notebook translation error")
            return None

        def _export_text_file(
            job: _TextTranslationJob,
            results: list[BatchResult],
        ) -> tuple[int, int] | None:
            """Export translated chunks for a single text/markdown file."""
            failed_count = sum(1 for r in results if r.status.value == "failed")
            successful_chunks = sum(1 for r in results if r.status.value == "success")

            if failed_count > 0:
                console.print_warning(f"{failed_count} chunk(s) failed to translate")
            if successful_chunks == 0 and failed_count > 0:
                console.print_error(f"翻译失败: {job.file_path} 所有分块均失败。")
                return None

            exporter = TranslationExporter(
                chunks=job.chunks,
                results=results,
                preserve_format=preserve_format,
                include_original=trans_config.include_original,
            )

            output_ext = Path(job.output_path).suffix.lower()
            output_format = None
            if output_ext == ".json":
                output_format = "json"
            elif output_ext in (".md", ".markdown"):
                output_format = "markdown"

            try:
                output_file = Path(job.output_path)
                if output_file.exists() and not force:
                    raise FileExistsError(
                        f"Output file already exists: {job.output_path}. Use --force to overwrite."
                    )

                exported_path = exporter.export(job.output_path, format_type=output_format)
                console.print_success(f"Translation saved to: {exported_path}")

                console.print(f"  Successful: {successful_chunks}/{len(results)}")
                if failed_count > 0:
                    console.print_warning(f"  Failed: {failed_count}/{len(results)}")

                total_in = sum(
                    r.metadata.input_tokens
                    for r in results
                    if r.metadata and r.status.value == "success"
                )
                total_out = sum(
                    r.metadata.output_tokens
                    for r in results
                    if r.metadata and r.status.value == "success"
                )
                console.print(
                    format_cost_estimate(
                        final_provider,
                        final_model,
                        total_in,
                        total_out,
                        pricing_map,
                        pricing_source=pricing_source,
                    )
                )
                return (total_in, total_out)

            except FileExistsError:
                console.print_error(
                    f"Output file already exists: {job.output_path}. Use --force to overwrite."
                )
            except Exception as e:
                console.print_error(f"Failed to export translation: {e}")
                logger.exception("Export error")
            return None

        def _translate_and_export_text_file(
            job: _TextTranslationJob,
        ) -> tuple[int, int] | None:
            """Translate and immediately export a single text/markdown file.

            Runs inside the shared file-level thread pool so each file is saved as
            soon as its own chunks finish, without waiting for other files.
            """
            console.print()
            console.print(f"[bold]Translating: {job.file_path}[/bold]")

            results, processor = run_global_batch_tasks(
                job.tasks,
                config_manager,
                max_workers=trans_config.threads,
                max_retries=trans_config.retries,
                show_progress=not stream,
                clamp_workers_to_task_count=True,
                stream_api=stream_api,
            )

            failed_count = sum(1 for r in results if r.status.value == "failed")
            successful_chunks = sum(1 for r in results if r.status.value == "success")
            if failed_count > 0:
                console.print_warning(f"{failed_count} chunk(s) failed to translate")
            if (
                successful_chunks == 0
                and failed_count > 0
                and getattr(processor, "_auth_error_logged", False)
            ):
                console.print_error(
                    f"翻译失败: API 认证错误, {job.file_path} 未产生有效译文。"
                )
                return None

            return _export_text_file(job, results)

        # Classify files and prepare text/markdown jobs up front.
        notebook_files: list[str] = []
        text_jobs: list[_TextTranslationJob] = []

        for file_path in resolved_files:
            file_type = TextSplitter.detect_file_type(file_path)
            if file_type == "notebook":
                notebook_files.append(file_path)
            elif file_type in ("markdown", "text"):
                job = _prepare_text_file(file_path)
                if job:
                    text_jobs.append(job)
            else:
                console.print_warning(
                    f"Unsupported file type: {Path(file_path).suffix}. "
                    f"Only .txt, .md, and .ipynb files are supported. Skipping."
                )

        if not text_jobs and not notebook_files:
            console.print_error("No translatable files found")
            raise typer.Exit(1)

        prompt_preview = Translator(
            target_language=trans_config.target_language,
            source_language=trans_config.source_language,
            style=trans_config.style,
            custom_prompt_template=trans_config.prompt_template,
            prompt_file=trans_config.prompt_file,
            glossary_pairs=glossary_pairs,
        )
        prompt_template_tokens = prompt_preview.count_prompt_template_tokens(final_model)
        pf = trans_config.prompt_file
        if pf:
            prompt_label = pf if pf.startswith("@") else str(Path(pf).expanduser().name)
        else:
            prompt_label = f"内置样式 ({trans_config.style})"
        console.print_info(
            f"提示词模板「{prompt_label}」指令部分(已替换语言占位、不含待译正文)≈ "
            f"{prompt_template_tokens} tokens (tiktoken · {final_model})"
        )

        session_in = 0
        session_out = 0

        # Translate and export files in parallel. Each file is saved as soon as it
        # finishes; a failure in one file does not block others.
        if trans_config.max_parallel_files <= 1:
            for job in text_jobs:
                try:
                    usage = _translate_and_export_text_file(job)
                    if usage:
                        session_in += usage[0]
                        session_out += usage[1]
                except Exception as e:
                    console.print_error(f"Failed to translate {job.file_path}: {e}")
                    logger.exception("Translation error")
            for file_path in notebook_files:
                try:
                    usage = _process_notebook_file(file_path)
                    if usage:
                        session_in += usage[0]
                        session_out += usage[1]
                except Exception as e:
                    console.print_error(f"Failed to translate {file_path}: {e}")
                    logger.exception("Translation error")
        else:
            with ThreadPoolExecutor(max_workers=trans_config.max_parallel_files) as executor:
                futures: dict[Future, str] = {}
                for job in text_jobs:
                    futures[
                        executor.submit(_translate_and_export_text_file, job)
                    ] = job.file_path
                for file_path in notebook_files:
                    futures[executor.submit(_process_notebook_file, file_path)] = file_path

                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        usage = future.result()
                        if usage:
                            session_in += usage[0]
                            session_out += usage[1]
                    except Exception as e:
                        console.print_error(f"Failed to translate {file_path}: {e}")
                        logger.exception("Translation error")

        processed_file_count = len(text_jobs) + len(notebook_files)
        if processed_file_count > 1:
            console.print()
            console.print("[bold]Session total (all files)[/bold]")
            console.print(
                format_cost_estimate(
                    final_provider,
                    final_model,
                    session_in,
                    session_out,
                    pricing_map,
                    pricing_source=pricing_source,
                )
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
