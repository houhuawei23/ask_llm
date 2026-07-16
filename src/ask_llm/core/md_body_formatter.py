"""Markdown body formatter using LLM API.

Splits markdown text into chunks using heading-aware token-based splitting,
formats each chunk concurrently via LLM API, and merges the results.

Orchestration skeleton (config fallback, prompt loading, runner wiring,
checkpoint save/resume) lives in :class:`ChunkedLLMJob` (P3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from ask_llm.config.context import get_config_or_none
from ask_llm.core.chunked_llm_job import ChunkedLLMJob
from ask_llm.core.format_checkpoint import (
    FailedChunkInfo,
    FormatCheckpoint,
    SuccessfulChunkInfo,
)
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.models import RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.text_splitter import TextChunk


@dataclass(frozen=True)
class _FormatBodyDefaults:
    """Built-in defaults matching default_config.yml."""

    max_chunk_tokens: int = 2400
    concurrency: int = 32
    retries: int = 3
    retry_delay: float = 1.0
    retry_delay_max: float = 10.0


_DEFAULT_FORMAT_BODY = _FormatBodyDefaults()


@dataclass
class BodyFormatStats:
    """Statistics for body formatting operation."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency: float = 0.0
    chunks_processed: int = 0
    chunks_failed: int = 0


@dataclass
class BodyFormatResult:
    """Result of a body formatting operation, including any failed chunks."""

    text: str
    stats: BodyFormatStats
    failed_chunks: list[FailedChunkInfo] = field(default_factory=list)
    checkpoint_path: str | None = None


@dataclass
class _ChunkResult:
    """Internal result for a single chunk processing attempt."""

    chunk_id: int
    success: bool
    formatted: str = ""
    original: str = ""
    meta: RequestMetadata | None = None
    retry_count: int = 0
    failed_info: FailedChunkInfo = field(default_factory=lambda: FailedChunkInfo(0, "", "", "", 0))


class BodyFormatter(ChunkedLLMJob):
    """Format markdown body using LLM API."""

    def __init__(
        self,
        processor: RequestProcessor,
        model: str,
        prompt_template: str | None = None,
        prompt_file: str | None = None,
        max_chunk_tokens: int | None = None,
        concurrency: int | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        retry_delay_max: float | None = None,
    ):
        """Initialize body formatter.

        Args:
            processor: RequestProcessor instance for LLM API calls
            model: Model name for tiktoken token counting in splitter
            prompt_template: Custom prompt template (overrides default)
            prompt_file: Path to prompt template file (overrides prompt_template)
            max_chunk_tokens: Max tokens per chunk (default from config)
            concurrency: Max concurrent API calls (default from config)
            retries: Max retry attempts per chunk (default from config)
            retry_delay: Initial retry delay in seconds (default from config)
            retry_delay_max: Max retry delay cap in seconds (default from config)
        """
        lr = get_config_or_none()
        fb_config = lr.unified_config.format_body if lr is not None else _DEFAULT_FORMAT_BODY
        super().__init__(
            processor,
            prompt_template=prompt_template,
            prompt_file=prompt_file,
            concurrency=self._pick(concurrency, fb_config.concurrency),
            retries=self._pick(retries, fb_config.retries),
            retry_delay=self._pick(retry_delay, fb_config.retry_delay),
            retry_delay_max=self._pick(retry_delay_max, fb_config.retry_delay_max),
        )
        self.model = model
        self.max_chunk_tokens = self._pick(max_chunk_tokens, fb_config.max_chunk_tokens)

    def format_body(
        self,
        text: str,
        source_file: str | None = None,
    ) -> BodyFormatResult:
        """Format markdown body by splitting into chunks and processing each.

        Uses MarkdownTokenSplitter for heading-aware splitting, then processes
        chunks concurrently via ThreadPoolExecutor, and merges results in order.

        Failed chunks (after all retries) are recorded and the original content
        is preserved. A checkpoint file is saved if any chunks fail.

        Args:
            text: Markdown text content
            source_file: Optional source file path for checkpoint naming

        Returns:
            BodyFormatResult with formatted text, stats, and failed chunk info

        Raises:
            ValueError: If prompt template is missing
        """
        stats = BodyFormatStats()
        failed_chunks: list[FailedChunkInfo] = []

        if not text.strip():
            return BodyFormatResult(text="", stats=stats, failed_chunks=[])

        template = self._resolve_template(
            "Prompt template required for body formatting. "
            "Set format_body.default_prompt_file in default_config.yml."
        )

        splitter = MarkdownTokenSplitter(
            model=self.model,
            max_chunk_tokens=self.max_chunk_tokens,
        )
        chunks = splitter.split(text)

        if not chunks:
            return BodyFormatResult(text=text, stats=stats, failed_chunks=[])

        # Calculate total document tokens
        total_doc_tokens = sum(splitter._tok(chunk.content) for chunk in chunks)
        logger.info(
            f"[BodyFormat] model={self.model}, total_doc_tokens≈{total_doc_tokens}, "
            f"chunks={len(chunks)}, concurrency={min(self.concurrency, len(chunks))}, "
            f"retries={self.retries}"
        )

        if len(chunks) > 1:
            logger.info(
                f"[BodyFormat] processing {len(chunks)} chunks concurrently "
                f"(max_workers={min(self.concurrency, len(chunks))}, "
                f"max_chunk_tokens={self.max_chunk_tokens})"
            )

        sorted_results: list[_ChunkResult] = self._run_units(
            chunks,
            self._process_chunk_worker,
            is_failed=lambda r: not r.success,
            error_message=lambda r: r.failed_info.error or "",
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.chunk_id,
        )

        # Collect results in order
        final_chunks: list[str] = []
        for res in sorted_results:
            if res.success and res.meta is not None:
                final_chunks.append(res.formatted)
                stats.total_input_tokens += res.meta.input_tokens
                stats.total_output_tokens += res.meta.output_tokens
                stats.total_latency += res.meta.latency
                stats.chunks_processed += 1
                logger.info(
                    f"[BodyFormat] chunk {res.chunk_id + 1}/{len(chunks)} completed: "
                    f"{res.meta.input_tokens} -> {res.meta.output_tokens} tokens "
                    f"in {res.meta.latency:.2f}s"
                )
            else:
                final_chunks.append(res.original)
                stats.chunks_failed += 1
                failed_chunks.append(res.failed_info)
                logger.warning(
                    f"[BodyFormat] chunk {res.chunk_id + 1}/{len(chunks)} FAILED: "
                    f"{res.failed_info.error}"
                )

        logger.info(
            f"[BodyFormat] all {len(chunks)} chunks done: "
            f"success={stats.chunks_processed}, failed={stats.chunks_failed}, "
            f"total {stats.total_input_tokens} -> {stats.total_output_tokens} tokens "
            f"in {stats.total_latency:.2f}s"
        )

        formatted_text = self._join_chunks(final_chunks)

        # Save checkpoint if any chunks failed
        successful = [
            SuccessfulChunkInfo(
                chunk_id=i,
                formatted_content=sr.formatted,
            )
            for i, sr in enumerate(sorted_results)
            if sr.success
        ]
        checkpoint_path = self._save_checkpoint(
            source_file=source_file,
            format_type="body",
            model=self.model,
            prompt_template=template,
            max_chunk_tokens=self.max_chunk_tokens,
            failed_chunks=failed_chunks,
            successful_chunks=successful,
        )

        return BodyFormatResult(
            text=formatted_text,
            stats=stats,
            failed_chunks=failed_chunks,
            checkpoint_path=checkpoint_path,
        )

    def _process_chunk_worker(self, chunk: TextChunk, retry_count: int) -> _ChunkResult:
        """Process a single chunk. Retry/backoff is handled by the shared runner."""
        template = self.prompt_template
        assert template is not None
        try:
            chunk_id, formatted, meta = self._process_chunk(chunk, template)
            return _ChunkResult(
                chunk_id=chunk_id,
                success=True,
                formatted=formatted,
                meta=meta,
                retry_count=retry_count,
            )
        except Exception as e:
            return _ChunkResult(
                chunk_id=chunk.chunk_id,
                success=False,
                original=chunk.content,
                retry_count=retry_count,
                failed_info=FailedChunkInfo(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    prompt_template=template,
                    error=str(e),
                    retry_count=retry_count,
                ),
            )

    def _process_chunk(self, chunk: TextChunk, template: str) -> tuple[int, str, RequestMetadata]:
        """Process a single chunk via LLM API (no retries).

        Args:
            chunk: Text chunk to format
            template: Prompt template

        Returns:
            Tuple of (chunk_id, formatted_content, metadata)
        """
        result = self.processor.process_with_metadata(
            content=chunk.content,
            prompt_template=template,
        )
        assert result.metadata is not None
        return chunk.chunk_id, result.content.rstrip(), result.metadata

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

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path: str,
        processor: RequestProcessor,
        model: str,
    ) -> BodyFormatResult:
        """Resume formatting from a checkpoint file.

        Loads the checkpoint, re-processes only the failed chunks, and merges
        with previously successful results.

        Args:
            checkpoint_path: Path to checkpoint JSON file
            processor: RequestProcessor instance
            model: Model name

        Returns:
            BodyFormatResult with complete formatted text
        """
        checkpoint = FormatCheckpoint.load(checkpoint_path)
        logger.info(
            f"[BodyFormat] Resuming from checkpoint: {checkpoint_path}, "
            f"failed_chunks={len(checkpoint.failed_chunks)}, "
            f"successful_chunks={len(checkpoint.successful_chunks)}"
        )

        formatter = cls(
            processor=processor,
            model=model or checkpoint.model,
            prompt_template=checkpoint.prompt_template,
            max_chunk_tokens=checkpoint.max_chunk_tokens,
        )

        stats = BodyFormatStats()
        still_failed: list[FailedChunkInfo] = []

        # Build result map from successful chunks
        result_map: dict[int, str] = {
            sc.chunk_id: sc.formatted_content for sc in checkpoint.successful_chunks
        }

        # Retry failed chunks through the shared bounded runner.
        failed_chunks_inputs = [
            TextChunk(content=fc.content, chunk_id=fc.chunk_id) for fc in checkpoint.failed_chunks
        ]
        retry_results = formatter._retry_failed_units(
            checkpoint,
            failed_chunks_inputs,
            formatter._process_chunk_worker,
            is_failed=lambda r: not r.success,
            error_message=lambda r: r.failed_info.error or "",
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.chunk_id,
        )
        for res in retry_results:
            fc = next(f for f in checkpoint.failed_chunks if f.chunk_id == res.chunk_id)
            if res.success and res.meta is not None:
                result_map[res.chunk_id] = res.formatted
                stats.total_input_tokens += res.meta.input_tokens
                stats.total_output_tokens += res.meta.output_tokens
                stats.total_latency += res.meta.latency
                stats.chunks_processed += 1
                logger.info(f"[BodyFormat] resumed chunk {res.chunk_id + 1} succeeded")
            else:
                result_map[res.chunk_id] = fc.content
                stats.chunks_failed += 1
                still_failed.append(res.failed_info)
                logger.warning(
                    f"[BodyFormat] resumed chunk {res.chunk_id + 1} still failed: {res.failed_info.error}"
                )

        # Merge in order
        all_ids = sorted(result_map.keys())
        final_chunks = [result_map[i] for i in all_ids]
        formatted_text = cls._join_chunks(final_chunks)

        # Save updated checkpoint if still failing
        successful = [
            SuccessfulChunkInfo(chunk_id=cid, formatted_content=result_map[cid])
            for cid in all_ids
            if cid not in {f.chunk_id for f in still_failed}
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

        return BodyFormatResult(
            text=formatted_text,
            stats=stats,
            failed_chunks=still_failed,
            checkpoint_path=checkpoint_path_updated,
        )
