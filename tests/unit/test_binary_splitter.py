"""Unit tests for BinarySplitter + TokenBudget (P3.2)."""

from ask_llm.core.binary_splitter import BinarySplitter, TokenBudget
from ask_llm.utils.token_counter import TokenCounter

MODEL = "deepseek-chat"


def _split(text: str, max_tokens: int, overhead: int = 0):
    budget = TokenBudget(model=MODEL, max_tokens=max_tokens, prompt_overhead=overhead)
    return BinarySplitter(budget).split(text)


class TestTokenBudget:
    def test_prompt_overhead_shrinks_content_budget(self):
        budget = TokenBudget(model=MODEL, max_tokens=100, prompt_overhead=30)
        assert budget.content_max_tokens == 70

    def test_prompt_overhead_clamped_to_one(self):
        budget = TokenBudget(model=MODEL, max_tokens=10, prompt_overhead=50)
        assert budget.content_max_tokens == 1

    def test_fits_respects_overhead(self):
        text = "word " * 60  # ~60+ tokens
        no_overhead = TokenBudget(model=MODEL, max_tokens=100)
        with_overhead = TokenBudget(model=MODEL, max_tokens=100, prompt_overhead=50)
        assert no_overhead.fits(text)
        assert not with_overhead.fits(text)


class TestBinarySplitter:
    def test_empty_text(self):
        assert _split("", 100) == []
        assert _split("   \n  ", 100) == []

    def test_full_document_single_chunk(self):
        text = "# Title\n\nshort body\n"
        chunks = _split(text, 1000)
        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "full_document"

    def test_splits_by_headings(self):
        sections = "\n\n".join(
            f"## Section {i}\n\n" + ("content " * 40) for i in range(4)
        )
        text = f"# Doc\n\n{sections}\n"
        chunks = _split(text, 80)
        assert len(chunks) > 1
        joined = "".join(c.content for c in chunks)
        for i in range(4):
            assert f"Section {i}" in joined

    def test_chunks_respect_budget(self):
        text = "\n\n".join(f"Paragraph {i}. " + "text " * 50 for i in range(8))
        budget = TokenBudget(model=MODEL, max_tokens=60)
        chunks = BinarySplitter(budget).split(text)
        assert len(chunks) > 1
        for c in chunks:
            assert TokenCounter.count_tokens(c.content, MODEL) <= budget.content_max_tokens

    def test_overhead_forces_more_chunks(self):
        text = "\n\n".join(f"Paragraph {i}. " + "text " * 40 for i in range(6))
        plain = _split(text, 100)
        with_overhead = _split(text, 100, overhead=60)
        assert len(with_overhead) >= len(plain)

    def test_fence_not_cut(self):
        """A fenced block that fits the budget stays intact in one chunk."""
        code = "\n".join(f"line {i} of code" for i in range(6))
        text = f"Intro paragraph.\n\n```python\n{code}\n```\n\nOutro paragraph.\n"
        budget = TokenBudget(model=MODEL, max_tokens=60)
        chunks = BinarySplitter(budget).split(text)
        fence_chunks = [c for c in chunks if "```python" in c.content]
        assert len(fence_chunks) == 1
        assert code in fence_chunks[0].content
        # fence is closed in the same chunk
        assert fence_chunks[0].content.count("```") == 2

    def test_oversized_fence_hard_split_is_last_resort(self):
        """A fence alone exceeding the budget may be hard-split (documented)."""
        code = "\n".join(f"line {i} of code" for i in range(30))
        text = f"```python\n{code}\n```\n"
        budget = TokenBudget(model=MODEL, max_tokens=40)
        chunks = BinarySplitter(budget).split(text)
        assert len(chunks) >= 1
        # all content preserved
        assert "".join(c.content for c in chunks).replace(" ", "") .replace("\n", "") >= (
            text.replace(" ", "").replace("\n", "")
        )

    def test_compat_wrapper_matches(self):
        """MarkdownTokenSplitter delegates to BinarySplitter identically."""
        from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter

        text = "# A\n\n" + ("body " * 100) + "\n\n## B\n\n" + ("more " * 100)
        via_wrapper = MarkdownTokenSplitter(MODEL, 50).split(text)
        via_impl = _split(text, 50)
        assert [c.content for c in via_wrapper] == [c.content for c in via_impl]
        assert [c.chunk_id for c in via_wrapper] == [c.chunk_id for c in via_impl]

    def test_wrapper_prompt_overhead(self):
        from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter

        text = "\n\n".join(f"Paragraph {i}. " + "text " * 40 for i in range(6))
        splitter = MarkdownTokenSplitter(MODEL, 100, prompt_overhead_tokens=60)
        assert splitter._budget.prompt_overhead == 60
        chunks = splitter.split(text)
        plain = MarkdownTokenSplitter(MODEL, 100).split(text)
        assert len(chunks) >= len(plain)


class TestChunkIdConvention:
    """P3.7: all producers emit dense zero-based ids in document order."""

    def test_binary_splitter_ids_dense_ordered(self):
        text = "\n\n".join(f"Paragraph {i}. " + "text " * 50 for i in range(8))
        chunks = _split(text, 60)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
        # positions non-decreasing in id order
        starts = [c.start_pos for c in chunks]
        assert starts == sorted(starts)

    def test_rebalance_ids_dense_ordered(self):
        from ask_llm.core.text_splitter import TextChunk
        from ask_llm.utils.chunk_balance import rebalance_translation_chunks

        chunks = [
            TextChunk(content="word " * 80, chunk_id=0, start_pos=0, end_pos=400),
            TextChunk(content="tiny", chunk_id=1, start_pos=400, end_pos=404),
            TextChunk(content="tiny2", chunk_id=2, start_pos=404, end_pos=409),
        ]
        out = rebalance_translation_chunks(
            chunks, model=MODEL, max_chunk_tokens=40, min_merge_tokens=10, enabled=True
        )
        assert [c.chunk_id for c in out] == list(range(len(out)))
