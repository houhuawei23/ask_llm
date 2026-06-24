"""Markdown heading formatter using LLM API.

Extracts headings from markdown text, formats them using LLM API,
and applies the formatted headings back to the original text.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from ask_llm.config.context import get_config
from ask_llm.core.concurrent import run_bounded_with_retries
from ask_llm.core.format_checkpoint import (
    FailedChunkInfo,
    FormatCheckpoint,
    SuccessfulChunkInfo,
    generate_checkpoint_path,
)
from ask_llm.core.processor import RequestProcessor


@dataclass
class HeadingMatch:
    """Represents a matched heading in markdown text."""

    raw_text: str  # Original heading line including # and text
    start_pos: int  # Start position in original text
    end_pos: int  # End position in original text
    level: int  # Current heading level (1-6)
    title: str  # Heading title text (without #)


class HeadingExtractor:
    """Extract headings from markdown text, excluding code blocks."""

    # Regex to match markdown headings: # Title, ## Title, etc.
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    # Regex to match markdown code fences: ``` or ~~~ (with optional language)
    CODE_FENCE_PATTERN = re.compile(r"^(```|~~~).*$", re.MULTILINE)

    @classmethod
    def _find_code_block_ranges(cls, text: str) -> list[tuple[int, int]]:
        """Find ranges of code blocks in markdown text.

        Returns:
            List of (start, end) tuples for each code block.
        """
        ranges = []
        in_code_block = False
        block_start = 0

        for match in cls.CODE_FENCE_PATTERN.finditer(text):
            if not in_code_block:
                block_start = match.start()
                in_code_block = True
            else:
                # End of code block (match.end() includes the newline after fence)
                ranges.append((block_start, match.end()))
                in_code_block = False

        # Handle unclosed code block (treat rest of text as code block)
        if in_code_block:
            ranges.append((block_start, len(text)))

        return ranges

    @classmethod
    def extract(cls, text: str) -> list[HeadingMatch]:
        """
        Extract all headings from markdown text, excluding those inside code blocks.

        Args:
            text: Markdown text content

        Returns:
            List of HeadingMatch objects in order of appearance
        """
        code_ranges = cls._find_code_block_ranges(text)

        headings = []
        for match in cls.HEADING_PATTERN.finditer(text):
            start_pos = match.start()

            # Skip headings inside code blocks
            if any(start <= start_pos < end for start, end in code_ranges):
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


class HeadingFormatter:
    """Format headings using LLM API."""

    # Instruction for context-aware batch processing (uses original headings, enables concurrent)
    CONTEXT_BATCH_INSTRUCTION = """注意：以下输入用 --- 分隔。前部分为前文标题（原始格式，供层级和编号规律参考），后部分为待格式化标题。请根据前文与后文的衔接关系，为后部分分配合适级别。只输出后部分的格式化结果，每行一个，保持顺序。

"""

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
        self.processor = processor
        self.prompt_template = prompt_template
        self.prompt_file = prompt_file
        fh_config = get_config().unified_config.format_heading
        self.batch_size = batch_size if batch_size is not None else fh_config.batch_size
        self.concurrency = concurrency if concurrency is not None else fh_config.concurrency
        self.retries = retries if retries is not None else fh_config.retries
        self.retry_delay = retry_delay if retry_delay is not None else fh_config.retry_delay
        self.retry_delay_max = (
            retry_delay_max if retry_delay_max is not None else fh_config.retry_delay_max
        )
        self._context_heading_count = fh_config.context_heading_count

        # Load prompt from file if specified
        if prompt_file:
            self.prompt_template = self._load_prompt_from_file(prompt_file)

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
                self.CONTEXT_BATCH_INSTRUCTION
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

        template = self.prompt_template
        if not template and self.prompt_file:
            template = self._load_prompt_from_file(self.prompt_file)
        if not template:
            raise ValueError(
                "Prompt template required for heading formatting. "
                "Set format_heading.default_prompt_file in default_config.yml."
            )
        self.prompt_template = template
        assert self.prompt_template is not None

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

        batch_results_list = run_bounded_with_retries(
            batch_items,
            self._process_batch_worker,
            max_workers=max_workers,
            max_retries=self.retries,
            retry_delay=self.retry_delay,
            retry_delay_max=self.retry_delay_max,
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
        checkpoint_path: str | None = None
        if failed_batches and source_file:
            checkpoint_path = str(generate_checkpoint_path(source_file, "title"))
            successful = []
            offset = 0
            for bidx in sorted(batch_results.keys()):
                res = batch_results[bidx]
                if res.success:
                    for fh in res.formatted:
                        successful.append(
                            SuccessfulChunkInfo(chunk_id=offset, formatted_content=fh)
                        )
                        offset += 1
                else:
                    offset += len(res.original_headings)

            checkpoint = FormatCheckpoint(
                version=1,
                source_file=source_file,
                format_type="title",
                model="",
                prompt_template=template,
                max_chunk_tokens=None,
                created_at=datetime.now().isoformat(),
                failed_chunks=failed_batches,
                successful_chunks=successful,
            )
            checkpoint.save(checkpoint_path)

        return HeadingFormatResult(
            formatted_headings=all_formatted,
            stats=stats,
            failed_batches=failed_batches,
            checkpoint_path=checkpoint_path,
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
