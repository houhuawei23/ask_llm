"""Unit tests for the extracted streaming collector (P1.3 / StreamCollector)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ask_llm.core.batch_models import ModelConfig
from ask_llm.core.protocols import ReasoningChunk
from ask_llm.core.stream_collector import stream_and_collect


def _config() -> ModelConfig:
    return ModelConfig(provider="p", model="m")


def test_plain_text_stream_is_concatenated_and_counted():
    # Force the word-count fallback (no tiktoken encoding) for determinism.
    with patch("ask_llm.core.stream_collector.TokenCounter.get_encoding", return_value=None):
        response, reasoning, tokens, latency = stream_and_collect(
            iter(["hello", " world"]),
            _config(),
            progress=None,
            progress_task_id=None,
            description_prefix="x",
        )
    assert response == "hello world"
    assert reasoning is None
    assert tokens >= 1
    assert latency >= 0


def test_reasoning_stream_separates_content_and_reasoning():
    chunks = [
        ReasoningChunk(content="answer", reasoning="because"),
        ReasoningChunk(content=" done", reasoning=" so"),
    ]
    with patch("ask_llm.core.stream_collector.TokenCounter.get_encoding", return_value=None):
        response, reasoning, tokens, _latency = stream_and_collect(
            iter(chunks),
            _config(),
            progress=None,
            progress_task_id=None,
            description_prefix="x",
            return_reasoning=True,
        )
    assert response == "answer done"
    assert reasoning == "because so"
    assert tokens >= 1


def test_progress_updates_are_throttled():
    progress = MagicMock()
    with (
        patch("ask_llm.core.stream_collector.TokenCounter.get_encoding", return_value=None),
        patch("ask_llm.core.stream_collector.PROGRESS_UPDATE_INTERVAL", 0.0),
    ):
        stream_and_collect(
            iter(["a", "b", "c"]),
            _config(),
            progress=progress,
            progress_task_id=1,
            description_prefix="p",
        )
    # With interval 0, every chunk updates the progress bar.
    assert progress.update.call_count == 3
