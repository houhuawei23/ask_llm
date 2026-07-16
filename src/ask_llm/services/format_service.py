"""Formatting orchestration service.

This module moves file-level formatting orchestration out of the CLI command so
that the command module stays focused on argument parsing and error handling.
"""

from __future__ import annotations

import os
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

from ask_llm.config.context import get_config_or_none
from ask_llm.core.format_checkpoint import FormatCheckpoint
from ask_llm.core.format_markdown_file import (
    FormatMarkdownOutcome,
    format_body_markdown_file,
    format_one_markdown_file,
)
from ask_llm.core.md_body_formatter import BodyFormatter
from ask_llm.core.md_heading_formatter import (
    HeadingApplier,
    HeadingExtractor,
    HeadingFormatter,
)
from ask_llm.core.processor import RequestProcessor
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler

# Built-in default matching default_config.yml so FormatService can resume
# without an active CLI config (e.g. library / embedded use).
_DEFAULT_FORMATTED_SUFFIX = "_formatted"


def format_one(
    file_path: str,
    *,
    format_type: str,
    processor: RequestProcessor,
    model: str,
    prompt_file_resolved: str,
    heading_batch_size: int | None = None,
    heading_concurrency: int | None = None,
    body_max_chunk_tokens: int | None = None,
    body_concurrency: int | None = None,
    output: str | None = None,
    inplace: bool = False,
    force: bool = False,
    retries: int | None = None,
    retry_delay: float | None = None,
    retry_delay_max: float | None = None,
) -> FormatMarkdownOutcome:
    """Single dispatcher for per-file formatting (P3.5).

    The title/body branch lives here exactly once; sequential and parallel
    runners both call this instead of duplicating the if/else.
    """
    if format_type == "title":
        return format_one_markdown_file(
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
    return format_body_markdown_file(
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

        outcome = format_one(
            file_path,
            format_type=format_type,
            processor=processor,
            model=model,
            prompt_file_resolved=prompt_file_resolved,
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
        return pool.submit(
            format_one,
            fp,
            format_type=format_type,
            processor=processor,
            model=model,
            prompt_file_resolved=prompt_file_resolved,
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


class FormatService:
    """High-level service for format command orchestration, including resume."""

    def __init__(
        self,
        *,
        processor: RequestProcessor,
        model: str,
    ) -> None:
        """Initialize the format service.

        Args:
            processor: Active request processor (provider already configured).
            model: Resolved model name.
        """
        self.processor = processor
        self.model = model

    def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        *,
        output: str | None,
        inplace: bool,
        force: bool,
    ) -> None:
        """Resume formatting from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint JSON file.
            output: Explicit output path.
            inplace: Overwrite the source file.
            force: Overwrite existing output file.

        Supports both body and title checkpoints (P3.5; title resume was
        previously rejected).

        Raises:
            RuntimeError: If writing output fails, or the resumed heading count
                does not match the source file.
        """
        checkpoint = FormatCheckpoint.load(checkpoint_path)
        source_file = checkpoint.source_file

        console.print_info(f"从 checkpoint 恢复: {checkpoint_path}")
        console.print_info(f"源文件: {source_file}")
        console.print_info(
            f"失败 chunk 数: {len(checkpoint.failed_chunks)}, "
            f"成功 chunk 数: {len(checkpoint.successful_chunks)}"
        )

        if checkpoint.format_type == "body":
            result = BodyFormatter.resume_from_checkpoint(
                checkpoint_path,
                processor=self.processor,
                model=self.model,
            )
            final_text = result.text
            still_failed = result.failed_chunks
            updated_checkpoint = result.checkpoint_path
        else:
            # Title resume (P3.5): re-process failed heading batches, then
            # re-apply the merged heading list onto the source document.
            result = HeadingFormatter.resume_from_checkpoint(
                checkpoint_path,
                processor=self.processor,
            )
            source_text = FileHandler.read(source_file)
            headings = HeadingExtractor.extract(source_text)
            if len(result.formatted_headings) != len(headings):
                raise RuntimeError(
                    f"恢复结果标题数 ({len(result.formatted_headings)}) 与源文件标题数 "
                    f"({len(headings)}) 不一致，无法安全合并；请直接重新运行 format 命令。"
                )
            final_text = HeadingApplier().apply(
                source_text, headings, result.formatted_headings
            )
            still_failed = result.failed_batches
            updated_checkpoint = result.checkpoint_path

        if inplace:
            out_path = source_file
        elif output:
            out_path = output
        else:
            lr = get_config_or_none()
            suffix = (
                lr.unified_config.file.formatted_suffix
                if lr is not None
                else _DEFAULT_FORMATTED_SUFFIX
            )
            out_path = FileHandler.generate_output_path(source_file, suffix=suffix)

        try:
            FileHandler.write(out_path, final_text, force=force or inplace)
        except Exception as exc:
            raise RuntimeError(f"写入失败: {exc}") from exc

        if still_failed:
            console.print_warning(
                f"部分成功: {len(still_failed)} 个 chunk/batch 仍失败，原始内容已保留"
            )
            if updated_checkpoint:
                console.print_info(f"更新后的 checkpoint: {updated_checkpoint}")
        else:
            console.print_success(f"全部完成！已保存: {out_path}")
            try:
                os.remove(checkpoint_path)
                console.print_info(f"已删除 checkpoint: {checkpoint_path}")
            except OSError as e:
                # B11: don't swallow silently -- surface the residue so the user
                # knows the checkpoint wasn't removed (and can delete it manually).
                console.print_warning(
                    f"全部完成，但未能删除 checkpoint {checkpoint_path}: {e}（可手动删除）"
                )
