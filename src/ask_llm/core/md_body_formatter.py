"""Markdown body formatter using LLM API.

Splits markdown text into chunks using heading-aware token-based splitting,
formats each chunk concurrently via LLM API, and merges the results.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from ask_llm.config.context import get_config
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.text_splitter import TextChunk
from ask_llm.utils.file_handler import FileHandler


class BodyFormatter:
    """Format markdown body using LLM API."""

    def __init__(
        self,
        processor: RequestProcessor,
        model: str,
        prompt_template: str | None = None,
        prompt_file: str | None = None,
        max_chunk_tokens: int | None = None,
        concurrency: int | None = None,
    ):
        """Initialize body formatter.

        Args:
            processor: RequestProcessor instance for LLM API calls
            model: Model name for tiktoken token counting in splitter
            prompt_template: Custom prompt template (overrides default)
            prompt_file: Path to prompt template file (overrides prompt_template)
            max_chunk_tokens: Max tokens per chunk (default from config)
            concurrency: Max concurrent API calls (default from config)
        """
        self.processor = processor
        self.model = model
        self.prompt_template = prompt_template
        self.prompt_file = prompt_file

        fb_config = get_config().unified_config.format_body
        self.max_chunk_tokens = max_chunk_tokens if max_chunk_tokens is not None else fb_config.max_chunk_tokens
        self.concurrency = concurrency if concurrency is not None else fb_config.concurrency

        # Load prompt from file if specified
        if prompt_file:
            self.prompt_template = self._load_prompt_from_file(prompt_file)

    def format_body(self, text: str) -> str:
        """Format markdown body by splitting into chunks and processing each.

        Uses MarkdownTokenSplitter for heading-aware splitting, then processes
        chunks concurrently via ThreadPoolExecutor, and merges results in order.

        Args:
            text: Markdown text content

        Returns:
            Formatted markdown text

        Raises:
            ValueError: If prompt template is missing
            RuntimeError: If LLM API call fails
        """
        if not text.strip():
            return ""

        template = self.prompt_template
        if not template and self.prompt_file:
            template = self._load_prompt_from_file(self.prompt_file)
        if not template:
            raise ValueError(
                "Prompt template required for body formatting. "
                "Set format_body.default_prompt_file in default_config.yml."
            )

        # Split text into chunks
        splitter = MarkdownTokenSplitter(
            model=self.model,
            max_chunk_tokens=self.max_chunk_tokens,
        )
        chunks = splitter.split(text)

        if not chunks:
            return text

        if len(chunks) == 1:
            logger.debug("Single chunk, processing directly")
            _, formatted = self._process_chunk(chunks[0], template)
            return formatted

        # Process chunks concurrently
        max_workers = min(self.concurrency, len(chunks))

        def process_chunk_wrapper(chunk: TextChunk) -> tuple[int, str]:
            logger.info(
                f"Formatting body chunk {chunk.chunk_id + 1}/{len(chunks)} "
                f"(tokens: ~{splitter._tok(chunk.content)}, pos: {chunk.start_pos}-{chunk.end_pos})"
            )
            return self._process_chunk(chunk, template)

        try:
            if max_workers <= 1:
                # Sequential processing
                results = {}
                for chunk in chunks:
                    chunk_id, formatted = process_chunk_wrapper(chunk)
                    results[chunk_id] = formatted
                sorted_chunks = [results[i] for i in range(len(chunks))]
                return self._join_chunks(sorted_chunks)

            logger.info(
                f"Formatting body in {len(chunks)} chunks "
                f"(concurrency: {max_workers}, max_chunk_tokens: {self.max_chunk_tokens})"
            )

            results: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="format-body") as executor:
                futures = {
                    executor.submit(process_chunk_wrapper, chunk): chunk.chunk_id
                    for chunk in chunks
                }
                for future in as_completed(futures):
                    chunk_id, formatted = future.result()
                    results[chunk_id] = formatted

            # Merge in original order
            sorted_chunks = [results[i] for i in range(len(chunks))]
            return self._join_chunks(sorted_chunks)

        except Exception as e:
            logger.error(f"Failed to format body chunk: {e}")
            raise RuntimeError(f"LLM API call failed: {e}") from e

    def _process_chunk(self, chunk: TextChunk, template: str) -> tuple[int, str]:
        """Process a single chunk via LLM API.

        Args:
            chunk: Text chunk to format
            template: Prompt template

        Returns:
            Tuple of (chunk_id, formatted_content)
        """
        result = self.processor.process_with_metadata(
            content=chunk.content,
            prompt_template=template,
        )
        return chunk.chunk_id, result.content.rstrip()

    @staticmethod
    def _join_chunks(chunks: list[str]) -> str:
        """Merge formatted chunks ensuring proper paragraph separation.

        When text is split by paragraphs, the original blank-line separators
        may be stripped away during splitting and LLM processing. This method
        restores at least one blank line between chunks that would otherwise
        be concatenated without separation, preventing markdown elements like
        headings from being glued to preceding content.

        Args:
            chunks: List of formatted chunk strings.

        Returns:
            Merged markdown text with proper paragraph boundaries.
        """
        if not chunks:
            return ""
        result = chunks[0]
        for chunk in chunks[1:]:
            # Normalize trailing/leading newlines and ensure a blank line
            result = result.rstrip("\n") + "\n\n" + chunk.lstrip("\n")
        return result

    @staticmethod
    def _load_prompt_from_file(prompt_path: str) -> str:
        """Load prompt template from file.

        Supports @ prefix for relative paths from project root.

        Args:
            prompt_path: Path to prompt file (may start with @)

        Returns:
            Prompt template content

        Raises:
            FileNotFoundError: If prompt file not found
            OSError: If file cannot be read
        """
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
