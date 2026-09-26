"""Integration tests: Ctrl-C interruption must not lose paid content (P0 #1).

Regression tests for the format pipeline's interrupt semantics:

- An interrupted chunked run must return exactly one result per unit, with
  abandoned units converted into explicit failed results.
- The output text must retain the original content of abandoned units.
- A checkpoint covering the abandoned units must be written, with chunk ids
  taken from the results themselves (not positional enumeration).
- Resuming from that checkpoint completes the document.
"""

import os
import signal
from unittest.mock import MagicMock

from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.core.format_checkpoint import FormatCheckpoint
from ask_llm.core.md_body_formatter import BodyFormatter
from ask_llm.core.models import ProcessingResult, RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol

_BODY_TEMPLATE = "Format this text: {content}"

# Enough paragraphs to guarantee several chunks (each paragraph is unique).
_PARAGRAPHS = [f"Paragraph number {i:03d} with some filler text." for i in range(12)]
_BODY_TEXT = "\n\n".join(_PARAGRAPHS)


def _make_processor(responder) -> RequestProcessor:
    """Build a processor whose LLM call is delegated to *responder*."""
    mock_provider = MagicMock(spec=LLMProviderProtocol)
    mock_provider.name = "mock"
    mock_provider.default_model = "mock-model"
    mock_provider.config = MagicMock()
    mock_provider.config.api_temperature = 0.7
    processor = RequestProcessor(mock_provider)

    def mock_process_with_metadata(*args, **kwargs):
        content = responder(kwargs["content"])
        return ProcessingResult(
            content=content,
            metadata=RequestMetadata(
                provider="mock",
                model="mock-model",
                temperature=0.7,
                input_words=10,
                input_tokens=20,
                output_words=10,
                output_tokens=20,
                latency=0.1,
            ),
        )

    processor.process_with_metadata = mock_process_with_metadata
    return processor


def _raise_sigint(_content: str) -> str:
    os.kill(os.getpid(), signal.SIGINT)
    return "formatted"


def _format_all(content: str) -> str:
    return f"formatted[{content[-20:]}]"


class TestBodyFormatInterrupt:
    """Interrupted body formatting keeps every chunk and can resume."""

    def test_interrupt_preserves_all_chunks_and_resumes(self, tmp_path):
        set_config(ConfigLoader.load())
        source = tmp_path / "doc.md"
        source.write_text(_BODY_TEXT, encoding="utf-8")

        calls = {"n": 0}

        def responder(content: str) -> str:
            calls["n"] += 1
            if calls["n"] > 1:
                # Deliver SIGINT to the main thread while the runner waits.
                os.kill(os.getpid(), signal.SIGINT)
            return _format_all(content)

        formatter = BodyFormatter(
            _make_processor(responder),
            model="mock-model",
            prompt_template=_BODY_TEMPLATE,
            max_chunk_tokens=40,
            concurrency=1,
            retries=1,
        )
        result = formatter.format_body(_BODY_TEXT, source_file=str(source))

        # 1. Output must retain every chunk: formatted chunks appear as
        # ``formatted[...]``, abandoned ones keep their original text.
        for para in _PARAGRAPHS:
            assert (
                para in result.text or f"formatted[{para[-20:]}]" in result.text
            ), f"lost chunk content: {para!r}"

        # 2. A checkpoint must exist covering the abandoned chunks.
        assert result.checkpoint_path is not None
        checkpoint = FormatCheckpoint.load(result.checkpoint_path)
        assert checkpoint.failed_chunks, "abandoned chunks must be recorded as failed"

        # 3. Chunk ids must be the real ones (no positional renumbering): the
        # successful id set and failed id set must not overlap and must not
        # collide after resume.
        success_ids = {sc.chunk_id for sc in checkpoint.successful_chunks}
        failed_ids = {fc.chunk_id for fc in checkpoint.failed_chunks}
        assert success_ids and failed_ids
        assert success_ids.isdisjoint(failed_ids)

        # 4. Resume must complete the document.
        ok_formatter = BodyFormatter(
            _make_processor(_format_all),
            model="mock-model",
            prompt_template=_BODY_TEMPLATE,
            max_chunk_tokens=120,
            concurrency=1,
        )
        resumed = BodyFormatter.resume_from_checkpoint(
            result.checkpoint_path,
            ok_formatter.processor,
            model="mock-model",
        )
        assert resumed.failed_chunks == []
        # Every original chunk is either formatted or present verbatim.
        for para in _PARAGRAPHS:
            assert (
                para in resumed.text or f"formatted[{para[-20:]}]" in resumed.text
            ), f"resume lost chunk ending {para[-20:]!r}"

    def test_no_interrupt_writes_no_checkpoint(self, tmp_path):
        set_config(ConfigLoader.load())
        source = tmp_path / "ok.md"
        source.write_text(_BODY_TEXT, encoding="utf-8")
        formatter = BodyFormatter(
            _make_processor(_format_all),
            model="mock-model",
            prompt_template=_BODY_TEMPLATE,
            max_chunk_tokens=120,
            concurrency=2,
        )
        result = formatter.format_body(_BODY_TEXT, source_file=str(source))
        assert result.failed_chunks == []
        assert result.checkpoint_path is None
        assert "formatted[" in result.text


class TestTitleCheckpointModel:
    """Title checkpoints must record the real model (P0 #2)."""

    def test_checkpoint_carries_model_for_digest_verify(self, tmp_path):
        from ask_llm.core.format_checkpoint import compute_format_digest, generate_checkpoint_path
        from ask_llm.core.md_heading_formatter import HeadingExtractor, HeadingFormatter

        set_config(ConfigLoader.load())
        source = tmp_path / "paper.md"
        text = "# a\n\n# b\n\n# c\n"
        source.write_text(text, encoding="utf-8")

        n = {"i": 0}

        def responder(_content: str) -> str:
            n["i"] += 1
            if n["i"] == 1:
                # Kill during the FIRST batch so later batches are abandoned
                # before ever being submitted.
                os.kill(os.getpid(), signal.SIGINT)
            return "# formatted"

        formatter = HeadingFormatter(
            _make_processor(responder),
            prompt_template="Format: {content}",
            model="real-model",
            batch_size=1,
            concurrency=1,
            retries=1,
        )
        headings = HeadingExtractor.extract(text)
        result = formatter.format_headings(headings, source_file=str(source))

        assert result.checkpoint_path is not None
        checkpoint = FormatCheckpoint.load(result.checkpoint_path)
        assert checkpoint.model == "real-model", (
            "title checkpoint must store the real model so the resume digest matches"
        )

        # The digest the CLI would recompute on resume must match the stored one.
        expected = compute_format_digest(
            checkpoint.source_file,
            prompt_template=checkpoint.prompt_template,
            model="real-model",
            max_chunk_tokens=checkpoint.max_chunk_tokens,
            format_type="title",
        )
        assert checkpoint.config_digest == expected
        assert str(generate_checkpoint_path(str(source), "title")).endswith(
            ".title_checkpoint.json"
        )


def test_sigint_delivery_contract():
    """Sanity: SIGINT from a worker thread reaches the main-thread handler."""
    received = []
    prev = signal.getsignal(signal.SIGINT)

    def handler(_signum, _frame):
        received.append(_signum)
        signal.signal(signal.SIGINT, prev)

    signal.signal(signal.SIGINT, handler)
    try:
        os.kill(os.getpid(), signal.SIGINT)
        # Handler runs asynchronously on the main thread; spin briefly.
        import time

        for _ in range(100):
            if received:
                break
            time.sleep(0.01)
    finally:
        signal.signal(signal.SIGINT, prev)
    assert received == [signal.SIGINT]
