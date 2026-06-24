"""Formatting orchestration service.

This module moves file-level formatting orchestration out of the CLI command so
that the command module stays focused on argument parsing and error handling.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ask_llm.core.format_markdown_file import (
    FormatMarkdownOutcome,
    format_body_markdown_file,
    format_one_markdown_file,
)
from ask_llm.core.processor import RequestProcessor
from ask_llm.utils.console import console


def _print_format_summary(
    successful_count: int,
    failed_count: int,
    skipped_count: int,
    total_input_tokens: int,
    total_output_tokens: int,
) -> None:
    """Print a formatted summary of the formatting run."""
    console.print()
    console.print("[bold]格式化完成[/bold]")
    console.print(f"  成功: {successful_count}")
    if failed_count:
        console.print(f"  失败: {failed_count}")
    if skipped_count:
        console.print(f"  跳过: {skipped_count}")
    if total_input_tokens or total_output_tokens:
        console.print(f"  总消耗 tokens: {total_input_tokens} -> {total_output_tokens}")


def _handle_outcome(outcome: FormatMarkdownOutcome, format_type: str) -> tuple[bool, int, int]:
    """Print per-file outcome and return (success, input_tokens, output_tokens)."""
    if outcome.ok:
        console.print_success(f"已保存: {outcome.output_path}")
        if format_type == "title":
            console.print(f"  共格式化 {outcome.heading_count} 个标题")
        else:
            if outcome.total_input_tokens or outcome.total_output_tokens:
                console.print(
                    f"  消耗 tokens: {outcome.total_input_tokens} -> {outcome.total_output_tokens}"
                )
        if outcome.failed_chunks:
            console.print_warning(
                f"  部分失败: {len(outcome.failed_chunks)} 个 chunk/batch 失败，原始内容已保留"
            )
            if outcome.checkpoint_path:
                console.print_info(f"  可使用 --resume {outcome.checkpoint_path} 再次尝试")
        return True, outcome.total_input_tokens, outcome.total_output_tokens
    elif outcome.skipped:
        console.print_warning(f"跳过 {outcome.source_path}: {outcome.message}")
        return False, 0, 0
    else:
        console.print_error(f"{outcome.source_path}: {outcome.message}")
        return False, 0, 0


def run_sequential_format(
    resolved_files: list[str],
    *,
    format_type: str,
    processor: RequestProcessor,
    model: str,
    prompt_file_resolved: str,
    heading_batch_size: int | None,
    heading_concurrency: int | None,
    body_max_chunk_tokens: int | None,
    body_concurrency: int | None,
    output: str | None,
    inplace: bool,
    force: bool,
    retries: int | None,
    retry_delay: float | None,
    retry_delay_max: float | None,
) -> None:
    """Single-worker path: verbose per-file logging (legacy UX)."""
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for file_path in resolved_files:
        console.print()
        console.print(f"[bold]处理: {file_path}[/bold]")

        if format_type == "title":
            outcome = format_one_markdown_file(
                file_path,
                processor=processor,
                prompt_file_resolved=prompt_file_resolved,
                heading_batch_size=heading_batch_size,
                heading_concurrency=heading_concurrency,
                retries=retries,
                retry_delay=retry_delay,
                retry_delay_max=retry_delay_max,
                output=output,
                inplace=inplace,
                force=force,
            )
        else:
            outcome = format_body_markdown_file(
                file_path,
                processor=processor,
                model=model,
                prompt_file_resolved=prompt_file_resolved,
                body_max_chunk_tokens=body_max_chunk_tokens,
                body_concurrency=body_concurrency,
                retries=retries,
                retry_delay=retry_delay,
                retry_delay_max=retry_delay_max,
                output=output,
                inplace=inplace,
                force=force,
            )

        ok, in_toks, out_toks = _handle_outcome(outcome, format_type)
        if ok:
            successful_count += 1
            total_input_tokens += in_toks
            total_output_tokens += out_toks
        elif outcome.skipped:
            skipped_count += 1
        else:
            failed_count += 1

    _print_format_summary(
        successful_count, failed_count, skipped_count, total_input_tokens, total_output_tokens
    )


def run_parallel_format(
    resolved_files: list[str],
    *,
    format_type: str,
    processor: RequestProcessor,
    model: str,
    prompt_file_resolved: str,
    heading_batch_size: int | None,
    heading_concurrency: int | None,
    body_max_chunk_tokens: int | None,
    body_concurrency: int | None,
    output: str | None,
    inplace: bool,
    force: bool,
    max_workers: int,
    retries: int | None,
    retry_delay: float | None,
    retry_delay_max: float | None,
) -> None:
    """Process many files with a thread pool and a Rich progress bar."""
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    workers = min(max_workers, len(resolved_files))

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    def _submit_file(pool: ThreadPoolExecutor, fp: str) -> Any:
        if format_type == "title":
            return pool.submit(
                format_one_markdown_file,
                fp,
                processor=processor,
                prompt_file_resolved=prompt_file_resolved,
                heading_batch_size=heading_batch_size,
                heading_concurrency=heading_concurrency,
                retries=retries,
                retry_delay=retry_delay,
                retry_delay_max=retry_delay_max,
                output=output,
                inplace=inplace,
                force=force,
            )
        else:
            return pool.submit(
                format_body_markdown_file,
                fp,
                processor=processor,
                model=model,
                prompt_file_resolved=prompt_file_resolved,
                body_max_chunk_tokens=body_max_chunk_tokens,
                body_concurrency=body_concurrency,
                retries=retries,
                retry_delay=retry_delay,
                retry_delay_max=retry_delay_max,
                output=output,
                inplace=inplace,
                force=force,
            )

    with Progress(*progress_columns, console=console.rich_console, transient=False) as progress:
        task_id = progress.add_task(
            "[cyan]格式化 Markdown[/cyan]",
            total=len(resolved_files),
        )
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="format-md") as pool:
            future_map = {_submit_file(pool, fp): fp for fp in resolved_files}
            for fut in as_completed(future_map):
                fp = future_map[fut]
                try:
                    outcome = fut.result()
                except Exception as exc:
                    console.print_error(f"{fp}: {exc}")
                    failed_count += 1
                    progress.advance(task_id)
                    continue

                ok, in_toks, out_toks = _handle_outcome(outcome, format_type)
                if ok:
                    successful_count += 1
                    total_input_tokens += in_toks
                    total_output_tokens += out_toks
                elif outcome.skipped:
                    skipped_count += 1
                else:
                    failed_count += 1
                progress.advance(task_id)

    _print_format_summary(
        successful_count, failed_count, skipped_count, total_input_tokens, total_output_tokens
    )
