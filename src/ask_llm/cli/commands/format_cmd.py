"""Typer command `format_cmd` (split from former cli.py)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from typing_extensions import Annotated

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.format_markdown_file import format_one_markdown_file
from ask_llm.core.processor import RequestProcessor
from ask_llm.utils.console import console
from ask_llm.utils.md_path_discovery import discover_markdown_files

try:
    from llm_engine import create_provider_adapter
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise


def _default_file_workers() -> int:
    """Default parallel file workers for I/O-bound LLM calls."""
    cpu = os.cpu_count() or 4
    return max(1, min(16, cpu * 2))


def _output_is_single_file_path(output: str) -> bool:
    """True if ``-o`` clearly targets one file (not a directory)."""
    p = Path(output)
    if p.exists():
        return p.is_file()
    # Non-existent path: treat as file if it looks like a single markdown file
    suf = p.suffix.lower()
    return suf in (".md", ".markdown") and not output.endswith(os.sep)


def _validate_batch_output(output: str | None, file_count: int, inplace: bool) -> None:
    if inplace or file_count <= 1 or not output:
        return
    if _output_is_single_file_path(output):
        raise typer.BadParameter(
            "多个输入文件不能使用单个 Markdown 文件作为 -o/--output；请指定目录或省略 -o 使用默认命名。"
        )


def format_cmd(
    files: Annotated[
        list[str],
        typer.Argument(
            help="Markdown 文件、目录或 glob（目录下递归处理所有 .md / .markdown）",
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="输出文件或目录（默认按配置生成带后缀的新文件）",
        ),
    ] = None,
    config_path: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="配置文件路径",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-a",
            help="API 服务商",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="模型名称",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            "-t",
            help="采样温度 (0.0–2.0)",
            min=0.0,
            max=2.0,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="覆盖已存在的输出文件",
        ),
    ] = False,
    inplace: Annotated[
        bool,
        typer.Option(
            "--inplace",
            "-i",
            help="原地覆盖源文件",
        ),
    ] = False,
    heading_batch_size: Annotated[
        int | None,
        typer.Option(
            "--heading-batch-size",
            help="单次 API 调用最多格式化的标题数（默认见配置，过大可能截断）",
        ),
    ] = None,
    heading_concurrency: Annotated[
        int | None,
        typer.Option(
            "--heading-concurrency",
            help="单文件内标题批次的并发 API 数（默认见配置，1 为串行）",
        ),
    ] = None,
    prompt_file: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="提示词模板文件（支持 @ 项目相对路径，如 @prompts/md-heading-format.md）",
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            help="目录参数是否递归包含子目录中的 Markdown（默认：递归）",
        ),
    ] = True,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-j",
            min=1,
            help="并行处理文件数（多文件/目录批处理时生效；默认随 CPU 调整）",
        ),
    ] = None,
) -> None:
    """
    使用 LLM 规范化 Markdown 标题层级。

    支持传入文件、glob、**目录**（将并发处理目录内全部 Markdown）。

    示例::

        ask-llm format document.md
        ask-llm format ./notes_dir
        ask-llm format ./notes_dir -j 8 -o ./formatted_out/
        ask-llm format *.md --no-recursive
        ask-llm format doc.md --inplace
    """
    try:
        load_result = ConfigLoader.load(config_path)
        set_config(load_result)
        config_manager = ConfigManager(load_result.app_config)

        if provider:
            config_manager.set_provider(provider)

        config_manager.apply_overrides(
            model=model,
            temperature=temperature,
        )

        provider_config = config_manager.get_provider_config()
        default_model = config_manager.get_model_override() or config_manager.get_default_model()

        if not default_model:
            console.print_error("未指定模型。请使用 --model 或在配置中设置默认模型。")
            raise typer.Exit(1)

        llm_provider = create_provider_adapter(provider_config, default_model=default_model)
        processor = RequestProcessor(llm_provider)

        resolved_paths = discover_markdown_files(files, recursive=recursive)
        resolved_files: list[str] = [str(p) for p in resolved_paths]

        if not resolved_files:
            console.print_error("未找到可格式化的 Markdown 文件")
            raise typer.Exit(1)

        _validate_batch_output(output, len(resolved_files), inplace)

        fh_config = load_result.unified_config.format_heading
        prompt_resolved = prompt_file or fh_config.default_prompt_file

        file_workers = workers if workers is not None else _default_file_workers()

        console.print_info(
            f"待处理 {len(resolved_files)} 个 Markdown 文件"
            f"（目录递归={recursive}，并行数={file_workers}）"
        )

        use_parallel = len(resolved_files) > 1 and file_workers > 1

        if use_parallel:
            _run_parallel_format(
                resolved_files,
                processor=processor,
                prompt_file_resolved=prompt_resolved,
                heading_batch_size=heading_batch_size,
                heading_concurrency=heading_concurrency,
                output=output,
                inplace=inplace,
                force=force,
                max_workers=file_workers,
            )
        else:
            _run_sequential_format(
                resolved_files,
                processor=processor,
                prompt_file_resolved=prompt_resolved,
                heading_batch_size=heading_batch_size,
                heading_concurrency=heading_concurrency,
                output=output,
                inplace=inplace,
                force=force,
            )

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API 错误: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\n用户中断")
        raise typer.Exit(1) from None
    except Exception as e:
        raise_unexpected_cli_error("format", e)


def _run_sequential_format(
    resolved_files: list[str],
    *,
    processor: RequestProcessor,
    prompt_file_resolved: str,
    heading_batch_size: int | None,
    heading_concurrency: int | None,
    output: str | None,
    inplace: bool,
    force: bool,
) -> None:
    """Single-worker path: verbose per-file logging (legacy UX)."""
    successful_count = 0
    failed_count = 0
    skipped_count = 0

    for file_path in resolved_files:
        console.print()
        console.print(f"[bold]处理: {file_path}[/bold]")
        outcome = format_one_markdown_file(
            file_path,
            processor=processor,
            prompt_file_resolved=prompt_file_resolved,
            heading_batch_size=heading_batch_size,
            heading_concurrency=heading_concurrency,
            output=output,
            inplace=inplace,
            force=force,
        )
        if outcome.ok:
            successful_count += 1
            if inplace:
                console.print_success(f"已原地写入: {outcome.output_path}")
            else:
                console.print_success(f"已保存: {outcome.output_path}")
            console.print(f"  共格式化 {outcome.heading_count} 个标题")
        elif outcome.skipped:
            skipped_count += 1
            console.print_warning(f"跳过 {file_path}: {outcome.message}")
        else:
            failed_count += 1
            console.print_error(f"{file_path}: {outcome.message}")

    _print_format_summary(successful_count, failed_count, skipped_count)


def _run_parallel_format(
    resolved_files: list[str],
    *,
    processor: RequestProcessor,
    prompt_file_resolved: str,
    heading_batch_size: int | None,
    heading_concurrency: int | None,
    output: str | None,
    inplace: bool,
    force: bool,
    max_workers: int,
) -> None:
    """Process many files with a thread pool and a Rich progress bar."""
    successful_count = 0
    failed_count = 0
    skipped_count = 0

    workers = min(max_workers, len(resolved_files))

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with Progress(*progress_columns, console=console.rich_console, transient=False) as progress:
        task_id = progress.add_task(
            "[cyan]格式化 Markdown[/cyan]",
            total=len(resolved_files),
        )
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="format-md") as pool:
            future_map = {
                pool.submit(
                    format_one_markdown_file,
                    fp,
                    processor=processor,
                    prompt_file_resolved=prompt_file_resolved,
                    heading_batch_size=heading_batch_size,
                    heading_concurrency=heading_concurrency,
                    output=output,
                    inplace=inplace,
                    force=force,
                ): fp
                for fp in resolved_files
            }
            for fut in as_completed(future_map):
                fp = future_map[fut]
                try:
                    outcome = fut.result()
                except Exception as exc:
                    logger.exception("未捕获异常: {}", fp)
                    failed_count += 1
                    console.print_error(f"{fp}: {exc}")
                else:
                    if outcome.ok:
                        successful_count += 1
                        logger.info(
                            "完成 [{}] -> {} ({} 标题)",
                            fp,
                            outcome.output_path,
                            outcome.heading_count,
                        )
                    elif outcome.skipped:
                        skipped_count += 1
                        logger.warning("跳过 [{}]: {}", fp, outcome.message)
                    else:
                        failed_count += 1
                        logger.error("{}: {}", fp, outcome.message)
                progress.advance(task_id)

    _print_format_summary(successful_count, failed_count, skipped_count)


def _print_format_summary(successful: int, failed: int, skipped: int) -> None:
    console.print()
    if successful:
        console.print_success(f"成功: {successful} 个文件")
    if skipped:
        console.print_warning(f"跳过: {skipped} 个文件")
    if failed:
        console.print_warning(f"失败: {failed} 个文件")
