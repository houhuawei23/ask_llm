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
        enabled=True,
    )
    for c in out:
        assert TokenCounter.count_tokens(c.content, model) <= cap


def test_rebalance_keeps_fenced_block_atomic_when_it_fits() -> None:
    """D4: a fenced code block that fits the budget is never split mid-fence,
    even when surrounding text forces the document into multiple chunks. The
    rebalance path now routes through BinarySplitter, which is fence-aware."""
    model = "gpt-4"
    fence = "```python\nx = 1\ny = 2\nprint(x + y)\n```"
    text = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda\n\n"
        f"{fence}\n\n" + "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega alpha\n" * 6
    )
    chunks = [TextChunk(content=text, chunk_id=0, start_pos=0, end_pos=len(text), metadata={})]
    out = rebalance_translation_chunks(chunks, model, max_chunk_tokens=40, enabled=True)
    assert len(out) >= 2
    # The fenced block must appear intact inside exactly one chunk and no chunk
    # may carry an unbalanced fence marker.
    assert any(fence in c.content for c in out)
    for c in out:
        assert c.content.count("```") % 2 == 0, f"fence split across chunk: {c.content[:40]!r}"


def test_rebalance_respects_prompt_overhead() -> None:
    """D2: prompt_overhead reduces the usable content cap, so the same text
    splits into more (smaller) chunks when the template is larger."""
    model = "gpt-4"
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa\n" * 4
    chunks = [TextChunk(content=text, chunk_id=0, start_pos=0, end_pos=len(text), metadata={})]
    small = rebalance_translation_chunks(
        chunks, model, max_chunk_tokens=60, enabled=True, prompt_overhead=2
    )
    big = rebalance_translation_chunks(
        chunks, model, max_chunk_tokens=60, enabled=True, prompt_overhead=40
    )
    assert len(big) > len(small)


class TestAudit43MergeMetaAndSpans:
    """Audit 4.3: merges keep both sides' metadata; spans are source-relative."""

    def test_merge_preserves_both_sides_metadata(self):
        from ask_llm.utils.chunk_balance import _merge_meta

        left = {"type": "heading_section", "heading": "Intro"}
        right = {"type": "heading_section", "heading": "Methods", "rebalanced": False}
        merged = _merge_meta(left, right)

        assert merged["heading"] == ["Intro", "Methods"]  # both survive
        assert merged["type"] == "heading_section"  # equal values collapse
        assert merged["rebalanced"] is False

    def test_merged_chunk_spans_cover_both_sources(self):
        """A merged chunk's span runs from the first piece's source start to
        the last piece's source end — not cumulative offsets in a synthetic
        stream."""
        heavy = "para " * 400  # oversized: gets split
        tail_a = "tail-a content"
        tail_b = "tail-b content"
        sep_len = len("\n\n")
        chunks = [
            TextChunk(content=heavy, chunk_id=0, start_pos=0, end_pos=len(heavy), metadata={}),
            TextChunk(
                content=tail_a,
                chunk_id=1,
                start_pos=len(heavy) + 10,
                end_pos=len(heavy) + 10 + len(tail_a),
                metadata={},
            ),
            TextChunk(
                content=tail_b,
                chunk_id=2,
                start_pos=len(heavy) + 50,
                end_pos=len(heavy) + 50 + len(tail_b),
                metadata={},
            ),
        ]

        out = rebalance_translation_chunks(chunks, model="gpt-4", max_chunk_tokens=200)

        assert out
        for c in out:
            assert c.start_pos <= c.end_pos
        # Every output span stays within the true source extent.
        assert all(c.end_pos <= len(heavy) + 50 + len(tail_b) for c in out)

    def test_merged_chunk_carries_all_source_metadata(self):
        """Metadata of chunks merged together survives on the merged chunk."""
        meta_a = {"heading": "Section A"}
        meta_b = {"heading": "Section B"}
        chunks = [
            TextChunk(content="alpha beta", chunk_id=0, start_pos=0, end_pos=10, metadata=meta_a),
            TextChunk(content="gamma delta", chunk_id=1, start_pos=12, end_pos=23, metadata=meta_b),
        ]

        out = rebalance_translation_chunks(chunks, model="gpt-4", max_chunk_tokens=2000)

        assert len(out) == 1, "tiny neighbors must merge under a generous budget"
        assert out[0].metadata["heading"] == ["Section A", "Section B"]
        assert out[0].metadata["rebalanced"] is True
        # Source-relative span: origin chunk 0 start through origin chunk 1 end.
        assert out[0].start_pos == 0
        assert out[0].end_pos == 23
