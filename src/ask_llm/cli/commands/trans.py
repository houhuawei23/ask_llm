"""Typer command `trans` (split from former cli.py)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ModelConfig,
)
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.text_splitter import TextSplitter
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

from ask_llm.cli.common import _process_notebook_translation, _resolve_trans_input_paths
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
            help="Per-file concurrent API calls (from default_config.yml if not set)",
            min=1,
            max=100,
        ),
    ] = None,
    max_parallel_files: Annotated[
        int | None,
        typer.Option(
            "--max-parallel-files",
            help="Max files to process in parallel when translating a directory (default: 3)",
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
            help="Stream translation progress to console",
        ),
    ] = False,
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
        if trans_config.max_parallel_files > 1:
            console.print_info(
                f"Processing with max {trans_config.max_parallel_files} file(s) in parallel"
            )

        def _process_one_file(file_path: str) -> tuple[int, int] | None:
            """Process a single file (md, txt, or ipynb). Returns (input_tokens, output_tokens) for session total."""
            console.print()
            console.print(f"[bold]Processing: {file_path}[/bold]")

            # Check file type (auto-detect by extension)
            file_type = TextSplitter.detect_file_type(file_path)
            if file_type not in ("markdown", "text", "notebook"):
                console.print_warning(
                    f"Unsupported file type: {Path(file_path).suffix}. "
                    f"Only .txt, .md, and .ipynb files are supported. Skipping."
                )
                return None

            # Handle .ipynb notebook translation (markdown cells only, code cells preserved)
            if file_type == "notebook":
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
                        pricing_map=pricing_map,
                        pricing_source=pricing_source,
                    )
                except FileExistsError:
                    console.print_error("Output file already exists. Use --force to overwrite.")
                except Exception as e:
                    console.print_error(f"Failed to translate notebook {file_path}: {e}")
                    logger.exception("Notebook translation error")
                return None

            # Read file content (for .txt and .md)
            try:
                content = FileHandler.read(file_path, show_progress=not stream)
            except Exception as e:
                console.print_error(f"Failed to read file {file_path}: {e}")
                return None

            if not content.strip():
                console.print_warning(f"File {file_path} is empty. Skipping.")
                return None

            # Split by token budget (structure-aware for Markdown)
            if TextSplitter.detect_file_type(file_path) == "markdown":
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

            # Create translator
            translator = Translator(
                target_language=trans_config.target_language,
                source_language=trans_config.source_language,
                style=trans_config.style,
                custom_prompt_template=trans_config.prompt_template,
                prompt_file=trans_config.prompt_file,
            )

            # Create model config
            model_config = ModelConfig(
                provider=final_provider,
                model=final_model,
                temperature=trans_config.temperature,
                max_tokens=trans_config.max_output_tokens,
            )

            # Create translation tasks
            tasks = translator.create_translation_tasks(chunks, model_config)

            console.print_info(
                f"Translating {len(tasks)} chunk(s) with {trans_config.threads} thread(s)..."
            )
            results, processor = run_global_batch_tasks(
                tasks,
                config_manager,
                max_workers=trans_config.threads,
                max_retries=trans_config.retries,
                show_progress=not stream,
                clamp_workers_to_task_count=False,
            )

            # Check for failures
            failed_count = sum(1 for r in results if r.status.value == "failed")
            successful_chunks = sum(1 for r in results if r.status.value == "success")
            if failed_count > 0:
                console.print_warning(f"{failed_count} chunk(s) failed to translate")
            if (
                successful_chunks == 0
                and failed_count > 0
                and getattr(processor, "_auth_error_logged", False)
            ):
                console.print_error("翻译失败: API 认证错误, 未产生有效译文。")
                raise typer.Exit(1)

            # Determine output path
            if output:
                output_path = output
                # If output is a directory, create file-specific name
                if Path(output).is_dir():
                    input_file = Path(file_path)
                    translated_suffix = get_config().unified_config.file.translated_suffix
                    output_name = f"{input_file.stem}{translated_suffix}{input_file.suffix}"
                    output_path = str(Path(output) / output_name)
            else:
                # Auto-generate output path
                output_path = FileHandler.generate_output_path(
                    file_path, suffix=get_config().unified_config.file.translated_suffix
                )

            # Export results
            exporter = TranslationExporter(
                chunks=chunks,
                results=results,
                preserve_format=preserve_format,
                include_original=trans_config.include_original,
            )

            # Detect output format from extension
            output_format = None
            output_ext = Path(output_path).suffix.lower()
            if output_ext == ".json":
                output_format = "json"
            elif output_ext in (".md", ".markdown"):
                output_format = "markdown"

            try:
                # Check if output file exists
                output_file = Path(output_path)
                if output_file.exists() and not force:
                    raise FileExistsError(
                        f"Output file already exists: {output_path}. Use --force to overwrite."
                    )

                exported_path = exporter.export(output_path, format_type=output_format)
                console.print_success(f"Translation saved to: {exported_path}")

                # Display statistics
                console.print(f"  Successful: {successful_chunks}/{len(results)}")
                if failed_count > 0:
                    console.print_warning(f"  Failed: {failed_count}/{len(results)}")

                by_model = processor.calculate_statistics(results)
                model_key = f"{final_provider}/{final_model}"
                st = by_model.get(model_key)
                if st is None and by_model:
                    st = next(iter(by_model.values()))
                total_in = st.total_input_tokens if st else 0
                total_out = st.total_output_tokens if st else 0
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
                    f"Output file already exists: {output_path}. Use --force to overwrite."
                )
            except Exception as e:
                console.print_error(f"Failed to export translation: {e}")
                logger.exception("Export error")
            return None

        prompt_preview = Translator(
            target_language=trans_config.target_language,
            source_language=trans_config.source_language,
            style=trans_config.style,
            custom_prompt_template=trans_config.prompt_template,
            prompt_file=trans_config.prompt_file,
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

        # Run file processing (sequential or parallel)
        session_in = 0
        session_out = 0
        if trans_config.max_parallel_files <= 1:
            for file_path in resolved_files:
                usage = _process_one_file(file_path)
                if usage:
                    session_in += usage[0]
                    session_out += usage[1]
        else:
            with ThreadPoolExecutor(max_workers=trans_config.max_parallel_files) as executor:
                futures = {executor.submit(_process_one_file, fp): fp for fp in resolved_files}
                for future in as_completed(futures):
                    try:
                        usage = future.result()
                        if usage:
                            session_in += usage[0]
                            session_out += usage[1]
                    except Exception as e:
                        file_path = futures[future]
                        console.print_error(f"Failed to translate {file_path}: {e}")
                        logger.exception("Translation error")

        if len(resolved_files) > 1:
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
