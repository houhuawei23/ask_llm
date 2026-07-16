"""Markdown heading formatter using LLM API.

Extracts headings from markdown text, formats them using LLM API,
and applies the formatted headings back to the original text.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ask_llm.config.context import get_config_or_none
from ask_llm.core.chunked_llm_job import ChunkedLLMJob
from ask_llm.core.format_checkpoint import (
    FailedChunkInfo,
    FormatCheckpoint,
    SuccessfulChunkInfo,
)
from ask_llm.core.markdown_structure import (
    CODE_FENCE_PATTERN,
    HEADING_PATTERN,
    MarkdownStructure,
)
from ask_llm.core.processor import RequestProcessor


@dataclass(frozen=True)
class _FormatHeadingDefaults:
    """Built-in defaults matching default_config.yml."""

    batch_size: int = 160
    concurrency: int = 8
    context_heading_count: int = 5
    retries: int = 3
    retry_delay: float = 1.0
    retry_delay_max: float = 10.0


_DEFAULT_FORMAT_HEADING = _FormatHeadingDefaults()


@dataclass
class HeadingMatch:
    """Represents a matched heading in markdown text."""

    raw_text: str  # Original heading line including # and text
    start_pos: int  # Start position in original text
    end_pos: int  # End position in original text
    level: int  # Current heading level (1-6)
    title: str  # Heading title text (without #)


class HeadingExtractor:
    """Extract headings from markdown text, excluding code blocks and frontmatter.

    Delegates structure parsing to :class:`MarkdownStructure` (P3.1): fence
    ranges, frontmatter range, and heading spans are computed in one pass
    there; this class only adapts the spans to ``HeadingMatch`` objects.
    """

    # Kept for backward compatibility (external references/tests); the
    # canonical definitions live in ask_llm.core.markdown_structure.
    HEADING_PATTERN = HEADING_PATTERN
    CODE_FENCE_PATTERN = CODE_FENCE_PATTERN

    @classmethod
    def _find_code_block_ranges(cls, text: str) -> list[tuple[int, int]]:
        """Find ranges of code blocks in markdown text.

        Returns:
            List of (start, end) tuples for each code block.
        """
        return MarkdownStructure.parse(text).fence_ranges

    @classmethod
    def extract(cls, text: str) -> list[HeadingMatch]:
        """
        Extract all headings from markdown text, excluding those inside code
        blocks and YAML frontmatter.

        Args:
            text: Markdown text content

        Returns:
            List of HeadingMatch objects in order of appearance
        """
        structure = MarkdownStructure.parse(text)

        headings = []
        for match in HEADING_PATTERN.finditer(text):
            start_pos = match.start()

            # Skip headings inside code blocks / frontmatter
            if structure.is_protected(start_pos):
                continue

            level = len(match.group(1))  # Number of # characters
            title = match.group(2).strip()
            raw_text = match.group(0)  # Full match including # and title
            end_pos = match.end()

            headings.append(
                HeadingMatch(
                    raw_text=raw_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    level=level,
                    title=title,
                )
            )

        logger.debug(f"Extracted {len(headings)} headings from text")
        return headings


@dataclass
class HeadingFormatResult:
    """Result of heading formatting operation."""

    formatted_headings: list[str]
    stats: "HeadingFormatStats"
    failed_batches: list[FailedChunkInfo] = field(default_factory=list)
    checkpoint_path: str | None = None


@dataclass
class HeadingFormatStats:
    """Statistics for heading formatting operation."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency: float = 0.0
    batches_processed: int = 0
    batches_failed: int = 0


class HeadingFormatter(ChunkedLLMJob):
    """Format headings using LLM API."""

    # Fallback if the prompt file is unavailable (e.g. broken symlink in a
    # minimal install). Canonical source: prompts/md-heading-context-batch.md.
    _CONTEXT_BATCH_INSTRUCTION_FALLBACK = (
        "注意：以下输入用 --- 分隔。前部分为前文标题（原始格式，供层级和编号规律参考），"
        "后部分为待格式化标题。请根据前文与后文的衔接关系，为后部分分配合适级别。"
        "只输出后部分的格式化结果，每行一个，保持顺序。\n\n"
    )
    _context_batch_instruction_cache: str | None = None

    @classmethod
    def context_batch_instruction(cls) -> str:
        """Context-batch instruction, loaded from the prompt file (P3.6).

        Prompt text lives in ``prompts/md-heading-context-batch.md`` instead of
        the class body; the embedded string is only a defensive fallback.
        """
        if cls._context_batch_instruction_cache is not None:
            return cls._context_batch_instruction_cache
        instruction: str | None = None
        try:
            prompt_file = Path(__file__).resolve().parent.parent / "prompts" / (
                "md-heading-context-batch.md"
            )
            if prompt_file.is_file():
                instruction = prompt_file.read_text(encoding="utf-8")
        except OSError:
            instruction = None
        if not instruction or not instruction.strip():
            instruction = cls._CONTEXT_BATCH_INSTRUCTION_FALLBACK
        # Preserve the historical trailing blank line before 【前文标题】.
        cls._context_batch_instruction_cache = instruction.rstrip("\n") + "\n\n"
        return cls._context_batch_instruction_cache

    def __init__(
        self,
        processor: RequestProcessor,
        prompt_template: str | None = None,
        prompt_file: str | None = None,
        batch_size: int | None = None,
        concurrency: int | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        retry_delay_max: float | None = None,
    ):
        """
        Initialize heading formatter.

        Args:
            processor: RequestProcessor instance for LLM API calls
            prompt_template: Custom prompt template (overrides default)
            prompt_file: Path to prompt template file (overrides prompt_template)
            batch_size: Max headings per API call (default 80). Use smaller value
                if LLM output is truncated.
            concurrency: Max concurrent API calls (default 4). Set to 1 to disable.
            retries: Max retry attempts per batch (default from config)
            retry_delay: Initial retry delay in seconds (default from config)
            retry_delay_max: Max retry delay cap in seconds (default from config)
        """
        lr = get_config_or_none()
        fh_config = lr.unified_config.format_heading if lr is not None else _DEFAULT_FORMAT_HEADING
        super().__init__(
            processor,
            prompt_template=prompt_template,
            prompt_file=prompt_file,
            concurrency=self._pick(concurrency, fh_config.concurrency),
            retries=self._pick(retries, fh_config.retries),
            retry_delay=self._pick(retry_delay, fh_config.retry_delay),
            retry_delay_max=self._pick(retry_delay_max, fh_config.retry_delay_max),
        )
        self.batch_size = self._pick(batch_size, fh_config.batch_size)
        self._context_heading_count = fh_config.context_heading_count

    def _process_batch(
        self,
        batch: list[HeadingMatch],
        template: str,
        start_idx: int,
        total: int,
        context_headings: list[str] | None = None,
    ) -> list[str]:
        """Process a single batch. When context_headings provided, prepend for level consistency."""
        batch_text = "\n".join(h.raw_text for h in batch)

        if context_headings:
            content = (
                self.context_batch_instruction()
                + "【前文标题，供层级参考】\n"
                + "\n".join(context_headings)
                + "\n\n---\n\n【待格式化，只输出此部分】\n\n"
                + batch_text
            )
        else:
            content = batch_text

        result = self.processor.process_with_metadata(
            content=content,
            prompt_template=template,
        )
        return self._parse_formatted_headings(
            result.content.strip(),
            len(batch),
            take_last_only=bool(context_headings),
        )

    @dataclass
    class _BatchResult:
        """Internal result for a single batch processing attempt."""

        batch_idx: int
        success: bool
        formatted: list[str] = field(default_factory=list)
        original_headings: list[str] = field(default_factory=list)
        meta: Any = None
        retry_count: int = 0
        failed_info: FailedChunkInfo = field(
            default_factory=lambda: FailedChunkInfo(0, "", "", "", 0)
        )

    def _process_batch_worker(
        self,
        batch_item: tuple[int, list[HeadingMatch], list[str] | None, int],
        retry_count: int,
    ) -> "HeadingFormatter._BatchResult":
        """Process a single batch. Retry/backoff is handled by the shared runner."""
        batch_idx, batch, context_headings, total = batch_item
        template = self.prompt_template
        assert template is not None
        batch_text = "\n".join(h.raw_text for h in batch)
        try:
            formatted = self._process_batch(batch, template, batch_idx, total, context_headings)
            return HeadingFormatter._BatchResult(
                batch_idx=batch_idx,
                success=True,
                formatted=formatted,
                original_headings=[h.raw_text for h in batch],
                retry_count=retry_count,
            )
        except Exception as e:
            return HeadingFormatter._BatchResult(
                batch_idx=batch_idx,
                success=False,
                original_headings=[h.raw_text for h in batch],
                retry_count=retry_count,
                failed_info=FailedChunkInfo(
                    chunk_id=batch_idx,
                    content=batch_text,
                    prompt_template=template,
                    error=str(e),
                    retry_count=retry_count,
                ),
            )

    def format_headings(
        self,
        headings: list[HeadingMatch],
        source_file: str | None = None,
    ) -> HeadingFormatResult:
        """
        Format headings using LLM API.

        When headings exceed batch_size, processes in batches concurrently.
        Each batch (except the first) receives the last N original headings
        from the previous batch as context (enables concurrent + level consistency).

        Failed batches (after all retries) retain original headings.

        Args:
            headings: List of heading matches to format
            source_file: Optional source file path for checkpoint naming

        Returns:
            HeadingFormatResult with formatted headings and stats

        Raises:
            ValueError: If prompt template is missing
        """
        if not headings:
            return HeadingFormatResult(
                formatted_headings=[],
                stats=HeadingFormatStats(),
                failed_batches=[],
            )

        template = self._resolve_template(
            "Prompt template required for heading formatting. "
            "Set format_heading.default_prompt_file in default_config.yml."
        )

        batches: list[tuple[int, list[HeadingMatch], list[str] | None]] = []
        for i in range(0, len(headings), self.batch_size):
            batch = headings[i : i + self.batch_size]
            context: list[str] | None = None
            if i > 0:
                prev_batch = headings[i - self.batch_size : i]
                context = [h.raw_text for h in prev_batch[-self._context_heading_count :]]
            batches.append((i, batch, context))

        # Run batches through the shared bounded runner (single queue, unified retry/backoff).
        batch_items = [(i, batch, ctx, len(headings)) for i, batch, ctx in batches]
        max_workers = min(self.concurrency, len(batches))

        if max_workers <= 1:
            logger.info(
                f"Formatting {len(headings)} headings in {len(batches)} batches sequentially"
            )
        else:
            logger.info(
                f"Formatting {len(headings)} headings in {len(batches)} batches "
                f"(concurrency: {max_workers})"
            )

        batch_results_list = self._run_units(
            batch_items,
            self._process_batch_worker,
            is_failed=lambda r: not r.success,
            error_message=lambda r: r.failed_info.error or "",
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.batch_idx,
        )
        batch_results: dict[int, HeadingFormatter._BatchResult] = {
            r.batch_idx: r for r in batch_results_list
        }

        # Collect results in order
        stats = HeadingFormatStats()
        failed_batches: list[FailedChunkInfo] = []
        all_formatted: list[str] = []

        for bidx in sorted(batch_results.keys()):
            res = batch_results[bidx]
            if res.success:
                all_formatted.extend(res.formatted)
                stats.batches_processed += 1
                logger.info(
                    f"[HeadingFormat] batch {bidx + 1}/{len(batches)} completed: "
                    f"{len(res.formatted)} headings"
                )
            else:
                all_formatted.extend(res.original_headings)
                stats.batches_failed += 1
                failed_batches.append(res.failed_info)
                logger.warning(
                    f"[HeadingFormat] batch {bidx + 1}/{len(batches)} FAILED: {res.failed_info.error}"
                )

        logger.info(
            f"[HeadingFormat] all {len(batches)} batches done: "
            f"success={stats.batches_processed}, failed={stats.batches_failed}"
        )

        # Save checkpoint if any batches failed
        successful = []
        offset = 0
        for bidx in sorted(batch_results.keys()):
            res = batch_results[bidx]
            if res.success:
                for fh in res.formatted:
                    successful.append(SuccessfulChunkInfo(chunk_id=offset, formatted_content=fh))
                    offset += 1
            else:
                offset += len(res.original_headings)

        checkpoint_path = self._save_checkpoint(
            source_file=source_file,
            format_type="title",
            model="",
            prompt_template=template,
            max_chunk_tokens=None,
            failed_chunks=failed_batches,
            successful_chunks=successful,
        )

        return HeadingFormatResult(
            formatted_headings=all_formatted,
            stats=stats,
            failed_batches=failed_batches,
            checkpoint_path=checkpoint_path,
        )

    @staticmethod
    def _lines_to_matches(lines: list[str]) -> list[HeadingMatch]:
        """Rebuild minimal HeadingMatch objects from raw heading lines (resume)."""
        matches = []
        for line in lines:
            stripped = line.lstrip("#")
            level = len(line) - len(stripped) if line.startswith("#") else 0
            matches.append(
                HeadingMatch(
                    raw_text=line,
                    start_pos=0,
                    end_pos=len(line),
                    level=level,
                    title=stripped.strip(),
                )
            )
        return matches

    def _resume_batch_worker(
        self, unit: tuple[int, list[HeadingMatch]], retry_count: int
    ) -> "HeadingFormatter._BatchResult":
        """Re-process one failed batch from a checkpoint. Runner handles retry/backoff."""
        offset, matches = unit
        template = self.prompt_template
        assert template is not None
        try:
            formatted = self._process_batch(matches, template, offset, 0, None)
            return HeadingFormatter._BatchResult(
                batch_idx=offset,
                success=True,
                formatted=formatted,
                original_headings=[h.raw_text for h in matches],
                retry_count=retry_count,
            )
        except Exception as e:
            return HeadingFormatter._BatchResult(
                batch_idx=offset,
                success=False,
                original_headings=[h.raw_text for h in matches],
                retry_count=retry_count,
                failed_info=FailedChunkInfo(
                    chunk_id=offset,
                    content="\n".join(h.raw_text for h in matches),
                    prompt_template=template,
                    error=str(e),
                    retry_count=retry_count,
                ),
            )

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path: str,
        processor: RequestProcessor,
    ) -> HeadingFormatResult:
        """Resume title formatting from a checkpoint file.

        Symmetric with ``BodyFormatter.resume_from_checkpoint`` (P3.3): loads
        the checkpoint, re-processes only the failed batches, and merges with
        previously successful headings by heading ordinal.

        Args:
            checkpoint_path: Path to checkpoint JSON file
            processor: RequestProcessor instance

        Returns:
            HeadingFormatResult with complete formatted headings
        """
        checkpoint = FormatCheckpoint.load(checkpoint_path)
        formatter = cls(processor=processor, prompt_template=checkpoint.prompt_template)

        stats = HeadingFormatStats()
        still_failed: list[FailedChunkInfo] = []

        # Ordinal -> formatted heading line, seeded from prior successes.
        result_map: dict[int, str] = {
            sc.chunk_id: sc.formatted_content for sc in checkpoint.successful_chunks
        }

        # Rebuild failed batches: failed content is "\n"-joined raw heading
        # lines; chunk_id is the first heading ordinal of the batch.
        units: list[tuple[int, list[HeadingMatch]]] = []
        for fc in checkpoint.failed_chunks:
            lines = [ln for ln in fc.content.split("\n") if ln.strip()]
            if lines:
                units.append((fc.chunk_id, cls._lines_to_matches(lines)))

        retry_results = formatter._retry_failed_units(
            checkpoint,
            units,
            formatter._resume_batch_worker,
            is_failed=lambda r: not r.success,
            error_message=lambda r: r.failed_info.error or "",
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.batch_idx,
        )
        for res in retry_results:
            if res.success:
                for j, formatted in enumerate(res.formatted):
                    result_map[res.batch_idx + j] = formatted
                stats.batches_processed += 1
                logger.info(f"[HeadingFormat] resumed batch at {res.batch_idx} succeeded")
            else:
                for j, original in enumerate(res.original_headings):
                    result_map[res.batch_idx + j] = original
                stats.batches_failed += 1
                still_failed.append(res.failed_info)
                logger.warning(
                    f"[HeadingFormat] resumed batch at {res.batch_idx} still failed: "
                    f"{res.failed_info.error}"
                )

        all_ids = sorted(result_map.keys())
        all_formatted = [result_map[i] for i in all_ids]

        # Save updated checkpoint if still failing
        successful = [
            SuccessfulChunkInfo(chunk_id=cid, formatted_content=result_map[cid])
            for cid in all_ids
            if not any(
                f.chunk_id <= cid < f.chunk_id + len(f.content.split("\n")) for f in still_failed
            )
        ]
        checkpoint_path_updated = formatter._save_checkpoint(
            source_file=checkpoint.source_file,
            format_type=checkpoint.format_type,
            model=checkpoint.model,
            prompt_template=checkpoint.prompt_template,
            max_chunk_tokens=checkpoint.max_chunk_tokens,
            failed_chunks=still_failed,
            successful_chunks=successful,
            checkpoint_path=checkpoint_path,
        )

        return HeadingFormatResult(
            formatted_headings=all_formatted,
            stats=stats,
            failed_batches=still_failed,
            checkpoint_path=checkpoint_path_updated,
        )

    def _parse_formatted_headings(
        self,
        formatted_text: str,
        expected_count: int,
        take_last_only: bool = False,
    ) -> list[str]:
        """
        Parse formatted headings from LLM response.

        Args:
            formatted_text: LLM response text
            expected_count: Expected number of headings
            take_last_only: When True (context-aware batch), take only the last
                expected_count headings (LLM may have output context headings too)

        Returns:
            List of formatted heading strings

        Raises:
            ValueError: If parsing fails or count mismatch
        """
        # Extract lines that match heading pattern
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = heading_pattern.findall(formatted_text)

        if not matches:
            # Try to extract headings from lines (may have extra text)
            lines = formatted_text.split("\n")
            formatted_headings = []
            for line in lines:
                line = line.strip()
                if heading_pattern.match(line):
                    formatted_headings.append(line)

            if not formatted_headings:
                raise ValueError(
                    f"Could not parse any headings from LLM response. "
                    f"Expected {expected_count} headings."
                )
        else:
            # Reconstruct heading strings from matches
            # matches is list of tuples: (hashes_string, title)
            formatted_headings = [f"{hashes} {title}" for hashes, title in matches]

        if take_last_only and len(formatted_headings) > expected_count:
            formatted_headings = formatted_headings[-expected_count:]

        # Validate count
        if len(formatted_headings) != expected_count:
            logger.warning(
                f"Expected {expected_count} headings, got {len(formatted_headings)}. "
                f"Response: {formatted_text[:200]}..."
            )
            # If count mismatch but we have some headings, use what we got
            # Pad with original format if needed
            if len(formatted_headings) < expected_count:
                raise ValueError(
                    f"Insufficient headings in LLM response: "
                    f"expected {expected_count}, got {len(formatted_headings)}"
                )

        logger.debug(f"Parsed {len(formatted_headings)} formatted headings")
        return formatted_headings

    @staticmethod
    def _load_prompt_from_file(prompt_path: str) -> str:
        """
        Load prompt template from file.

        Supports @ prefix for relative paths from project root.

        Args:
            prompt_path: Path to prompt file (may start with @)

        Returns:
            Prompt template content

        Raises:
            FileNotFoundError: If prompt file not found
            OSError: If file cannot be read
        """
        from ask_llm.utils.prompt_resolver import load_prompt_template

        return load_prompt_template(prompt_path)


class HeadingApplier:
    """Apply formatted headings back to original text."""

    @staticmethod
    def apply(
        text: str, original_headings: list[HeadingMatch], formatted_headings: list[str]
    ) -> str:
        """
        Apply formatted headings to original text.

        Args:
            text: Original markdown text
            original_headings: List of original heading matches
            formatted_headings: List of formatted heading strings

        Returns:
            Text with formatted headings applied

        Raises:
            ValueError: If heading counts don't match
        """
        if len(original_headings) != len(formatted_headings):
            raise ValueError(
                f"Heading count mismatch: {len(original_headings)} original, "
                f"{len(formatted_headings)} formatted"
            )

        # Replace from end to start to avoid position offset issues
        result = text
        for original, formatted in reversed(
            list(zip(original_headings, formatted_headings, strict=False))
        ):
            # Ensure formatted heading ends with newline if original did
            if original.raw_text.endswith("\n") and not formatted.endswith("\n"):
                formatted = formatted + "\n"
            elif not original.raw_text.endswith("\n") and formatted.endswith("\n"):
                formatted = formatted.rstrip("\n")

            # Replace in reverse order
            before = result[: original.start_pos]
            after = result[original.end_pos :]
            result = before + formatted + after

        logger.debug("Applied formatted headings to text")
        return result
