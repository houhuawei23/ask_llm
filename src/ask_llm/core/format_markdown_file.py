"""Single-file Markdown heading format workflow (used by CLI batch and sequential paths)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ask_llm.config.context import get_config
from ask_llm.core.md_heading_formatter import (
    HeadingApplier,
    HeadingExtractor,
    HeadingFormatter,
)
from ask_llm.core.processor import RequestProcessor
from ask_llm.utils.file_handler import FileHandler


@dataclass(frozen=True)
class FormatMarkdownOutcome:
    """Result of formatting one Markdown file."""

    source_path: str
    ok: bool
    skipped: bool
    message: str
    output_path: str | None = None
    heading_count: int = 0


def format_one_markdown_file(
    file_path: str,
    *,
    processor: RequestProcessor,
    prompt_file_resolved: str,
    heading_batch_size: int | None,
    heading_concurrency: int | None,
    output: str | None,
    inplace: bool,
    force: bool,
) -> FormatMarkdownOutcome:
    """
    Read a Markdown file, format headings via LLM, and write the result.

    Args:
        file_path: Input path
        processor: Shared request processor (thread-safe for typical HTTP backends)
        prompt_file_resolved: Resolved prompt path (same as previous CLI default)
        heading_batch_size: Optional override for batch size
        heading_concurrency: Optional override for per-file API concurrency
        output: Optional ``-o`` path (file or directory)
        inplace: Overwrite source
        force: Overwrite existing output when not inplace

    Returns:
        :class:`FormatMarkdownOutcome` with success/skip/error information
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in (".md", ".markdown"):
        return FormatMarkdownOutcome(
            source_path=file_path,
            ok=False,
            skipped=True,
            message=f"Unsupported type {suffix!r}; only .md and .markdown",
        )

    try:
        content = FileHandler.read(file_path, show_progress=False)
    except Exception as exc:
        logger.exception("Failed to read {}", file_path)
        return FormatMarkdownOutcome(
            source_path=file_path,
            ok=False,
            skipped=False,
            message=str(exc),
        )

    if not content.strip():
        return FormatMarkdownOutcome(
            source_path=file_path,
            ok=False,
            skipped=True,
            message="File is empty",
        )

    headings = HeadingExtractor.extract(content)
    if not headings:
        return FormatMarkdownOutcome(
            source_path=file_path,
            ok=False,
            skipped=True,
            message="No headings found",
        )

    try:
        formatter = HeadingFormatter(
            processor=processor,
            prompt_file=prompt_file_resolved,
            batch_size=heading_batch_size,
            concurrency=heading_concurrency,
        )
        formatted_headings = formatter.format_headings(headings)
    except Exception as exc:
        logger.exception("Heading format failed for {}", file_path)
        return FormatMarkdownOutcome(
            source_path=file_path,
            ok=False,
            skipped=False,
            message=str(exc),
        )

    try:
        applier = HeadingApplier()
        formatted_content = applier.apply(content, headings, formatted_headings)
    except Exception as exc:
        logger.exception("Apply headings failed for {}", file_path)
        return FormatMarkdownOutcome(
            source_path=file_path,
            ok=False,
            skipped=False,
            message=str(exc),
        )

    if inplace:
        output_path = file_path
    elif output:
        output_path = output
        if Path(output).is_dir():
            input_file = Path(file_path)
            formatted_suffix = get_config().unified_config.file.formatted_suffix
            output_name = f"{input_file.stem}{formatted_suffix}{input_file.suffix}"
            output_path = str(Path(output) / output_name)
    else:
        output_path = FileHandler.generate_output_path(
            file_path,
            suffix=get_config().unified_config.file.formatted_suffix,
        )

    out_p = Path(output_path)
    try:
        if out_p.exists() and not force and not inplace:
            return FormatMarkdownOutcome(
                source_path=file_path,
                ok=False,
                skipped=False,
                message=f"Output exists: {output_path} (use --force)",
            )
        FileHandler.write(output_path, formatted_content, force=force or inplace)
    except Exception as exc:
        logger.exception("Write failed for {}", output_path)
        return FormatMarkdownOutcome(
            source_path=file_path,
            ok=False,
            skipped=False,
            message=str(exc),
        )

    return FormatMarkdownOutcome(
        source_path=file_path,
        ok=True,
        skipped=False,
        message="OK",
        output_path=output_path,
        heading_count=len(headings),
    )
