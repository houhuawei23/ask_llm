"""Typer command `format_cmd` (split from former cli.py)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.processor import RequestProcessor
from ask_llm.services.format_service import (
    FormatService,
    run_parallel_format,
    run_sequential_format,
)
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
    type_: Annotated[
        str,
        typer.Option(
            "--type",
            "-T",
            help="格式化类型: title (标题层级) 或 body (正文排版)",
        ),
    ] = "title",
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
            help="单文件内标题批次的并发 API 数（默认见配置，1 为串行；仅 title 模式生效）",
        ),
    ] = None,
    body_max_chunk_tokens: Annotated[
        int | None,
        typer.Option(
            "--body-max-chunk-tokens",
            help="正文格式化时每块最大 token 数（默认见配置；仅 body 模式生效）",
        ),
    ] = None,
    body_concurrency: Annotated[
        int | None,
        typer.Option(
            "--body-concurrency",
            help="单文件内正文 chunk 的并发 API 数（默认见配置，1 为串行；仅 body 模式生效）",
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
    max_depth: Annotated[
        int | None,
        typer.Option(
            "--max-depth",
            "-d",
            help="目录递归的最大深度（0=仅当前目录，1=当前+一级子目录，默认无限制）",
            min=0,
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-j",
            min=1,
            help="并行处理文件数（多文件/目录批处理时生效；默认随 CPU 调整）",
        ),
    ] = None,
    retries: Annotated[
        int | None,
        typer.Option(
            "--retries",
            "-r",
            min=0,
            help="每个 chunk/batch 的最大重试次数（默认见配置）",
        ),
    ] = None,
    retry_delay: Annotated[
        float | None,
        typer.Option(
            "--retry-delay",
            help="初始重试延迟秒数（默认见配置）",
            min=0.0,
        ),
    ] = None,
    retry_delay_max: Annotated[
        float | None,
        typer.Option(
            "--retry-delay-max",
            help="最大重试延迟秒数（默认见配置）",
            min=0.0,
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            "-R",
            help="从 checkpoint 文件恢复未完成的格式化",
        ),
    ] = None,
) -> None:
    """
    使用 LLM 格式化 Markdown 文档。

    支持两种格式化模式：
    - title: 规范化标题层级（默认）
    - body: 优化正文排版

    支持传入文件、glob、**目录**（将并发处理目录内全部 Markdown）。

    示例::

        ask-llm format document.md
        ask-llm format document.md --type title
        ask-llm format document.md --type body
        ask-llm format ./notes_dir
        ask-llm format ./notes_dir -j 8 -o ./formatted_out/
        ask-llm format *.md --no-recursive
        ask-llm format doc.md --inplace
        ask-llm format ./notes_dir --max-depth 1
        ask-llm format doc.md --type body --resume doc.md.body_checkpoint.json
    """
    try:
        # Load config and apply CLI overrides first so that --resume respects
        # --config, --provider, --model, and --temperature.
        load_result = ConfigLoader.load(config_path)
        set_config(load_result)
        config_manager = ConfigManager(load_result.app_config, load_result.unified_config)

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
        format_service = FormatService(processor=processor, model=default_model)

        # Handle --resume mode after config and processor are ready
        if resume:
            format_service.resume_from_checkpoint(
                resume,
                output=output,
                inplace=inplace,
                force=force,
            )
            raise typer.Exit(0)

        type_lower = type_.lower()
        if type_lower not in ("title", "body"):
            console.print_error(f"不支持的格式化类型: {type_}。请使用 title 或 body。")
            raise typer.Exit(1)

        resolved_paths = discover_markdown_files(files, recursive=recursive, max_depth=max_depth)
        resolved_files: list[str] = [str(p) for p in resolved_paths]

        if not resolved_files:
            console.print_error("未找到可格式化的 Markdown 文件")
            raise typer.Exit(1)

        _validate_batch_output(output, len(resolved_files), inplace)

        if type_lower == "title":
            fh_config = load_result.unified_config.format_heading
            prompt_resolved = prompt_file or fh_config.default_prompt_file
        else:
            fb_config = load_result.unified_config.format_body
            prompt_resolved = prompt_file or fb_config.default_prompt_file

        file_workers = workers if workers is not None else _default_file_workers()

        depth_str = f"，最大深度={max_depth}" if max_depth is not None else ""
        console.print_info(
            f"待处理 {len(resolved_files)} 个 Markdown 文件"
            f"（类型={type_lower}，目录递归={recursive}{depth_str}，并行数={file_workers}）"
        )

        use_parallel = len(resolved_files) > 1 and file_workers > 1

        if use_parallel:
            run_parallel_format(
                resolved_files,
                format_type=type_lower,
                processor=processor,
                model=default_model,
                prompt_file_resolved=prompt_resolved,
                heading_batch_size=heading_batch_size,
                heading_concurrency=heading_concurrency,
                body_max_chunk_tokens=body_max_chunk_tokens,
                body_concurrency=body_concurrency,
                output=output,
                inplace=inplace,
                force=force,
                max_workers=file_workers,
                retries=retries,
                retry_delay=retry_delay,
                retry_delay_max=retry_delay_max,
            )
        else:
            run_sequential_format(
                resolved_files,
                format_type=type_lower,
                processor=processor,
                model=default_model,
                prompt_file_resolved=prompt_resolved,
                heading_batch_size=heading_batch_size,
                heading_concurrency=heading_concurrency,
                body_max_chunk_tokens=body_max_chunk_tokens,
                body_concurrency=body_concurrency,
                output=output,
                inplace=inplace,
                force=force,
                retries=retries,
                retry_delay=retry_delay,
                retry_delay_max=retry_delay_max,
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
