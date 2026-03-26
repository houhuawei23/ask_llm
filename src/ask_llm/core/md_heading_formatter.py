"""Markdown heading formatter using LLM API.

Extracts headings from markdown text, formats them using LLM API,
and applies the formatted headings back to the original text.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

from ask_llm.config.context import get_config
from ask_llm.core.processor import RequestProcessor
from ask_llm.utils.file_handler import FileHandler


@dataclass
class HeadingMatch:
    """Represents a matched heading in markdown text."""

    raw_text: str  # Original heading line including # and text
    start_pos: int  # Start position in original text
    end_pos: int  # End position in original text
    level: int  # Current heading level (1-6)
    title: str  # Heading title text (without #)


class HeadingExtractor:
    """Extract headings from markdown text."""

    # Regex to match markdown headings: # Title, ## Title, etc.
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    @classmethod
    def extract(cls, text: str) -> List[HeadingMatch]:
        """
        Extract all headings from markdown text.

        Args:
            text: Markdown text content

        Returns:
            List of HeadingMatch objects in order of appearance
        """
        headings = []
        for match in cls.HEADING_PATTERN.finditer(text):
            level = len(match.group(1))  # Number of # characters
            title = match.group(2).strip()
            raw_text = match.group(0)  # Full match including # and title
            start_pos = match.start()
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


class HeadingFormatter:
    """Format headings using LLM API."""

    # Instruction for context-aware batch processing (uses original headings, enables concurrent)
    CONTEXT_BATCH_INSTRUCTION = """注意：以下输入用 --- 分隔。前部分为前文标题（原始格式，供层级和编号规律参考），后部分为待格式化标题。请根据前文与后文的衔接关系，为后部分分配合适级别。只输出后部分的格式化结果，每行一个，保持顺序。

"""

    def __init__(
        self,
        processor: RequestProcessor,
        prompt_template: Optional[str] = None,
        prompt_file: Optional[str] = None,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
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
        """
        self.processor = processor
        self.prompt_template = prompt_template
        self.prompt_file = prompt_file
        fh_config = get_config().unified_config.format_heading
        self.batch_size = batch_size if batch_size is not None else fh_config.batch_size
        self.concurrency = concurrency if concurrency is not None else fh_config.concurrency
        self._context_heading_count = fh_config.context_heading_count

        # Load prompt from file if specified
        if prompt_file:
            self.prompt_template = self._load_prompt_from_file(prompt_file)

    def _process_batch(
        self,
        batch: List[HeadingMatch],
        template: str,
        start_idx: int,
        total: int,
        context_headings: Optional[List[str]] = None,
    ) -> List[str]:
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

    def format_headings(self, headings: List[HeadingMatch]) -> List[str]:
        """
        Format headings using LLM API.

        When headings exceed batch_size, processes in batches concurrently.
        Each batch (except the first) receives the last N original headings
        from the previous batch as context (enables concurrent + level consistency).

        Args:
            headings: List of heading matches to format

        Returns:
            List of formatted heading strings (one per heading)

        Raises:
            ValueError: If LLM response cannot be parsed
            RuntimeError: If LLM API call fails
        """
        if not headings:
            return []

        template = self.prompt_template
        if not template and self.prompt_file:
            template = self._load_prompt_from_file(self.prompt_file)
        if not template:
            raise ValueError(
                "Prompt template required for heading formatting. "
                "Set format_heading.default_prompt_file in default_config.yml."
            )

        # Build batch list with context (original headings from prev batch, computed upfront)
        batches: List[tuple[int, List[HeadingMatch], Optional[List[str]]]] = []
        for i in range(0, len(headings), self.batch_size):
            batch = headings[i : i + self.batch_size]
            context: Optional[List[str]] = None
            if i > 0:
                prev_batch = headings[i - self.batch_size : i]
                context = [h.raw_text for h in prev_batch[-self._context_heading_count :]]
            batches.append((i, batch, context))

        max_workers = min(self.concurrency, len(batches))

        def process_batch(
            batch_idx: int,
            batch: List[HeadingMatch],
            context_headings: Optional[List[str]],
        ) -> tuple[int, List[str]]:
            logger.info(
                f"Formatting headings {batch_idx + 1}-{batch_idx + len(batch)} "
                f"of {len(headings)} (batch size: {len(batch)})"
            )
            formatted = self._process_batch(
                batch,
                template,
                batch_idx,
                len(headings),
                context_headings=context_headings,
            )
            return batch_idx, formatted

        try:
            if max_workers <= 1:
                all_formatted = []
                for i, batch, ctx in batches:
                    _, formatted = process_batch(i, batch, ctx)
                    all_formatted.extend(formatted)
                return all_formatted

            logger.info(
                f"Formatting {len(headings)} headings in {len(batches)} batches "
                f"(concurrency: {max_workers})"
            )
            batch_results: dict[int, List[str]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_batch, i, batch, ctx): i for i, batch, ctx in batches
                }
                for future in as_completed(futures):
                    batch_idx, formatted = future.result()
                    batch_results[batch_idx] = formatted

            return [
                heading for idx in sorted(batch_results.keys()) for heading in batch_results[idx]
            ]
        except Exception as e:
            logger.error(f"Failed to format headings batch: {e}")
            raise RuntimeError(f"LLM API call failed: {e}") from e

    def _parse_formatted_headings(
        self,
        formatted_text: str,
        expected_count: int,
        take_last_only: bool = False,
    ) -> List[str]:
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
        # Handle @ prefix (relative to project root)
        if prompt_path.startswith("@"):
            relative_path = prompt_path[1:]
            current_dir = Path.cwd()
            project_root = None
            markers = get_config().unified_config.project_root_markers
            for marker in markers:
                for parent in [current_dir, *list(current_dir.parents)]:
                    if (parent / marker).exists():
                        project_root = parent
                        break
                if project_root:
                    break

            if project_root:
                prompt_file = project_root / relative_path.lstrip("/")
            else:
                prompt_file = Path(relative_path.lstrip("/"))
        else:
            prompt_file = Path(prompt_path)

        # Resolve absolute path
        if not prompt_file.is_absolute():
            prompt_file = prompt_file.resolve()

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        logger.debug(f"Loading prompt template from: {prompt_file}")
        try:
            content = FileHandler.read(str(prompt_file))
            return content.strip()
        except Exception as e:
            raise OSError(f"Failed to read prompt file {prompt_file}: {e}") from e


class HeadingApplier:
    """Apply formatted headings back to original text."""

    @staticmethod
    def apply(
        text: str, original_headings: List[HeadingMatch], formatted_headings: List[str]
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
        for original, formatted in reversed(list(zip(original_headings, formatted_headings))):
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
