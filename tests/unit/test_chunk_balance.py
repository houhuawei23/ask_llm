"""Tests for token-based translation chunk rebalancing."""

from ask_llm.core.text_splitter import TextChunk
from ask_llm.utils.chunk_balance import rebalance_translation_chunks
from ask_llm.utils.token_counter import TokenCounter


def test_rebalance_disabled_returns_same_list() -> None:
    chunks = [
        TextChunk(content="a" * 100, chunk_id=0, start_pos=0, end_pos=100, metadata={}),
    ]
    out = rebalance_translation_chunks(chunks, "deepseek-chat", enabled=False)
    assert out is chunks


def test_rebalance_splits_token_heavy_chunk() -> None:
    # Short string, tight token cap → forces multiple splits without huge tiktoken cost
    heavy = "alpha " * 120
    chunks = [TextChunk(content=heavy, chunk_id=0, start_pos=0, end_pos=len(heavy), metadata={})]
    out = rebalance_translation_chunks(
        chunks,
        "deepseek-chat",
        max_chunk_tokens=40,
        min_merge_tokens=10,
        enabled=True,
    )
    assert len(out) >= 2
    for c in out:
        assert c.chunk_id >= 0


def test_rebalance_merges_tiny_chunks() -> None:
    chunks = [
        TextChunk(content="hi", chunk_id=0, start_pos=0, end_pos=2, metadata={}),
        TextChunk(content="there", chunk_id=1, start_pos=2, end_pos=7, metadata={}),
    ]
    out = rebalance_translation_chunks(
        chunks,
        "deepseek-chat",
        max_chunk_tokens=2400,
        min_merge_tokens=500,
        enabled=True,
    )
    assert len(out) == 1


def test_rebalance_merges_two_non_tiny_adjacent_when_combined_fits() -> None:
    """Both sides > legacy min_merge threshold but sum under cap → still one merged body."""
    model = "deepseek-chat"
    n = 400
    a = "alpha " * n
    b = "beta " * n
    chunks = [
        TextChunk(content=a, chunk_id=0, start_pos=0, end_pos=len(a), metadata={}),
        TextChunk(
            content=b,
            chunk_id=1,
            start_pos=len(a),
            end_pos=len(a) + len(b),
            metadata={},
        ),
    ]
    out = rebalance_translation_chunks(
        chunks,
        model,
        max_chunk_tokens=2400,
        min_merge_tokens=400,
        enabled=True,
    )
    assert len(out) == 1
    assert "alpha" in out[0].content and "beta" in out[0].content


def test_rebalance_each_output_within_max_tokens() -> None:
    heavy = "alpha " * 120
    chunks = [TextChunk(content=heavy, chunk_id=0, start_pos=0, end_pos=len(heavy), metadata={})]
    cap = 40
    model = "deepseek-chat"
    out = rebalance_translation_chunks(
        chunks,
        model,
        max_chunk_tokens=cap,
        min_merge_tokens=10,
        enabled=True,
    )
    for c in out:
        assert TokenCounter.count_tokens(c.content, model) <= cap
