"""Formatting orchestration service.

This module moves file-level formatting orchestration out of the CLI command so
that the command module stays focused on argument parsing and error handling.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
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
from ask_llm.core.format_checkpoint import (
    CHECKPOINT_VERSION,
    FormatCheckpoint,
    compute_format_digest,
)
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
from ask_llm.utils.prompt_resolver import load_prompt_template

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


@dataclass
class FormatRunStats:
    """F-lite convention: every long-running service returns a result carrying
    a ``failed_count`` so the CLI can map it to a non-zero exit code (H2)."""

    successful_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def run_format(
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
    max_workers: int = 1,
    retries: int | None,
    retry_delay: float | None,
    retry_delay_max: float | None,
) -> FormatRunStats:
    """Format all files sequentially (``max_workers <= 1``) or via a thread pool.

    Sequential mode keeps the legacy verbose per-file logging; parallel mode
    shows a single Rich progress bar.

    Returns per-run counts so the CLI owns the exit code (P4.2/H2).
    """
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def _record(outcome: FormatMarkdownOutcome) -> None:
        nonlocal successful_count, failed_count, skipped_count, total_input_tokens
        nonlocal total_output_tokens
        ok, in_toks, out_toks = _handle_outcome(outcome, format_type)
        if ok:
            successful_count += 1
            total_input_tokens += in_toks
            total_output_tokens += out_toks
        elif outcome.skipped:
            skipped_count += 1
        else:
            failed_count += 1

    format_kwargs: dict[str, Any] = {
        "format_type": format_type,
        "processor": processor,
        "model": model,
        "prompt_file_resolved": prompt_file_resolved,
        "heading_batch_size": heading_batch_size,
        "heading_concurrency": heading_concurrency,
        "body_max_chunk_tokens": body_max_chunk_tokens,
        "body_concurrency": body_concurrency,
        "output": output,
        "inplace": inplace,
        "force": force,
        "retries": retries,
        "retry_delay": retry_delay,
        "retry_delay_max": retry_delay_max,
    }

    use_parallel = len(resolved_files) > 1 and max_workers > 1
    if not use_parallel:
        for file_path in resolved_files:
            console.print()
            console.print(f"[bold]处理: {file_path}[/bold]")
            # M9/2.25: mirror the parallel path's per-file guard — one
            # unexpected error used to abort the whole sequential run with no
            # summary for the remaining files.
            try:
                _record(format_one(file_path, **format_kwargs))
            except Exception as exc:
                console.print_error(f"{file_path}: {exc}")
                failed_count += 1
    else:
        workers = min(max_workers, len(resolved_files))
        progress_columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        with Progress(*progress_columns, console=console.rich_console, transient=False) as progress:
            task_id = progress.add_task("[cyan]格式化 Markdown[/cyan]", total=len(resolved_files))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="format-md") as pool:
                future_map = {
                    pool.submit(format_one, fp, **format_kwargs): fp for fp in resolved_files
                }
                for fut in as_completed(future_map):
                    fp = future_map[fut]
                    try:
                        outcome = fut.result()
                    except Exception as exc:
                        console.print_error(f"{fp}: {exc}")
                        failed_count += 1
                        progress.advance(task_id)
                        continue
                    _record(outcome)
                    progress.advance(task_id)

    _print_format_summary(
        successful_count, failed_count, skipped_count, total_input_tokens, total_output_tokens
    )
    return FormatRunStats(
        successful_count=successful_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


@dataclass
class FormatResumeOutcome:
    """Result of a ``format --resume`` run (F-lite: carries the failure count
    so the CLI can map it to a non-zero exit code — audit 2.7)."""

    output_path: str
    still_failed_count: int = 0
    checkpoint_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.still_failed_count == 0


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
        current_model: str | None = None,
        current_prompt_file: str | None = None,
        current_max_chunk_tokens: int | None = None,
        current_format_type: str | None = None,
    ) -> FormatResumeOutcome:
        """Resume formatting from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint JSON file.
            output: Explicit output path.
            inplace: Overwrite the source file.
            force: Overwrite existing output file.
            current_model: The run's model (M7/2.25) — the digest is computed
                from the *current* run's options, not the checkpoint's own
                stored values, so resuming with a different ``--model``/
                ``--type``/prompt file/chunk budget is refused instead of
                silently retrying failed chunks under new settings. ``None``
                falls back to the checkpoint's stored value (pre-2.25
                callers), which only detects source-file edits.
            current_prompt_file: Resolved prompt path for the current run.
            current_max_chunk_tokens: Explicit chunk-budget override for the
                current run (config-default runs pass ``None``).
            current_format_type: ``"body"``/``"title"`` for the current run.

        Supports both body and title checkpoints (P3.5; title resume was
        previously rejected). Refuses pre-v4 checkpoints and checkpoints whose
        input digest no longer matches the current source file (M8): a stale
        resume rebuilding output from the old body — and with ``--inplace``
        clobbering the edited source — is worse than rerunning. When resuming
        ``--inplace`` with chunks still failing, a one-shot ``.bak`` of the
        source is kept (audit 2.6).

        Returns:
            FormatResumeOutcome with the output path and remaining failure count.

        Raises:
            RuntimeError: If writing output fails, the resumed heading count
                does not match the source file, or the checkpoint is stale.
        """
        checkpoint = FormatCheckpoint.load(checkpoint_path)
        if checkpoint.version < CHECKPOINT_VERSION:
            raise RuntimeError(
                f"checkpoint 版本过旧 (v{checkpoint.version}，当前 v{CHECKPOINT_VERSION})，"
                "缺少输入一致性摘要，无法安全恢复；请删除该 checkpoint 后重新运行 format 命令。"
            )
        source_file = checkpoint.source_file
        if checkpoint.config_digest:
            # M7/2.25: prefer the current run's options; a None argument
            # (option not explicitly set) falls back to the stored value.
            current_template = checkpoint.prompt_template
            if current_prompt_file:
                current_template = load_prompt_template(current_prompt_file)
            current_digest = compute_format_digest(
                source_file,
                prompt_template=current_template,
                model=current_model if current_model is not None else checkpoint.model,
                max_chunk_tokens=(
                    current_max_chunk_tokens
                    if current_max_chunk_tokens is not None
                    else checkpoint.max_chunk_tokens
                ),
                format_type=(
                    current_format_type
                    if current_format_type is not None
                    else checkpoint.format_type
                ),
            )
            if checkpoint.config_digest != current_digest:
                raise RuntimeError(
                    f"checkpoint 与当前运行不一致（源文件在 checkpoint 创建后被修改过，或 "
                    f"model/prompt/chunk 预算/格式化类型发生了变化）：{source_file}。"
                    "为避免把旧结果错拼到新内容或新设置上，已拒绝恢复；"
                    "请删除该 checkpoint 后重新运行。"
                )

        console.print_info(f"从 checkpoint 恢复: {checkpoint_path}")
        console.print_info(f"源文件: {source_file}")
        console.print_info(
            f"失败 chunk 数: {len(checkpoint.failed_chunks)}, "
            f"成功 chunk 数: {len(checkpoint.successful_chunks)}"
        )

        if checkpoint.format_type == "body":
            body_result = BodyFormatter.resume_from_checkpoint(
                checkpoint_path,
                processor=self.processor,
                model=self.model,
            )
            final_text = body_result.text
            still_failed = body_result.failed_chunks
            updated_checkpoint = body_result.checkpoint_path
        else:
            # Title resume (P3.5): re-process failed heading batches, then
            # re-apply the merged heading list onto the source document.
            heading_result = HeadingFormatter.resume_from_checkpoint(
                checkpoint_path,
                processor=self.processor,
            )
            source_text = FileHandler.read(source_file)
            headings = HeadingExtractor.extract(source_text)
            if len(heading_result.formatted_headings) != len(headings):
                raise RuntimeError(
                    f"恢复结果标题数 ({len(heading_result.formatted_headings)}) 与源文件标题数 "
                    f"({len(headings)}) 不一致，无法安全合并；请直接重新运行 format 命令。"
                )
            final_text = HeadingApplier().apply(
                source_text, headings, heading_result.formatted_headings
            )
            still_failed = heading_result.failed_batches
            updated_checkpoint = heading_result.checkpoint_path

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
            if inplace and still_failed:
                # 2.6: the source is the user's only copy; a partial resume must
                # not destroy it without recourse. One-shot backup, no rotation.
                backup_path = Path(source_file + ".bak")
                shutil.copyfile(source_file, backup_path)
                console.print_warning(
                    f"仍有失败 chunk，--inplace 覆盖前已备份原文件: {backup_path}"
                )
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

        return FormatResumeOutcome(
            output_path=out_path,
            still_failed_count=len(still_failed),
            checkpoint_path=updated_checkpoint,
        )
