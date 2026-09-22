"""Unit tests for BinarySplitter + TokenBudget (P3.2)."""

from itertools import pairwise

from ask_llm.core.binary_splitter import BinarySplitter, TokenBudget
from ask_llm.utils.token_counter import TokenCounter

MODEL = "deepseek-chat"


def _split(text: str, max_tokens: int, overhead: int = 0):
    budget = TokenBudget(model=MODEL, max_tokens=max_tokens, prompt_overhead=overhead)
    return BinarySplitter(budget).split(text)


class TestTokenBudget:
    def test_prompt_overhead_shrinks_content_budget(self):
        # Use an exact-model (gpt-4) to isolate overhead from the approximate-
        # model safety factor (see test_approximate_model_applies_safety_factor).
        budget = TokenBudget(model="gpt-4", max_tokens=100, prompt_overhead=30)
        assert budget.content_max_tokens == 70

    def test_prompt_overhead_clamped_to_one(self):
        budget = TokenBudget(model="gpt-4", max_tokens=10, prompt_overhead=50)
        assert budget.content_max_tokens == 1

    def test_fits_respects_overhead(self):
        text = "word " * 60  # ~60+ tokens
        no_overhead = TokenBudget(model="gpt-4", max_tokens=100)
        with_overhead = TokenBudget(model="gpt-4", max_tokens=100, prompt_overhead=50)
        assert no_overhead.fits(text)
        assert not with_overhead.fits(text)

    def test_approximate_model_applies_safety_factor(self):
        """D1: approximate models (DeepSeek/Qwen) get a reduced content cap and
        a stricter ``fits()`` so the fast-path "whole input fits" no longer
        admits chunks that overflow the real context window."""
        from ask_llm.core.constants import APPROX_TOKEN_SAFETY_FACTOR

        approx = TokenBudget(model=MODEL, max_tokens=100)  # MODEL = deepseek-chat
        exact = TokenBudget(model="gpt-4", max_tokens=100)
        assert approx.content_max_tokens == int(100 * APPROX_TOKEN_SAFETY_FACTOR)
        assert exact.content_max_tokens == 100

        text = "word " * 90  # ~90 tokens: fits the exact cap, exceeds the reduced one
        assert exact.fits(text)
        assert not approx.fits(text)


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
        sections = "\n\n".join(f"## Section {i}\n\n" + ("content " * 40) for i in range(4))
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
        assert "".join(c.content for c in chunks).replace(" ", "").replace("\n", "") >= (
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
        out = rebalance_translation_chunks(chunks, model=MODEL, max_chunk_tokens=40, enabled=True)
        assert [c.chunk_id for c in out] == list(range(len(out)))


class TestSentenceSplitLosslessness:
    """K1 regression guard: the sentence-pairing loop in
    ``_split_long_paragraph`` used ``range(0, len(sentences) - 1, 2)``, which
    dropped the final tail element of ``re.split`` with a capture group —
    silently losing the last sentence of every budget-split paragraph."""

    @staticmethod
    def _long_paragraph(n_sentences: int, end_with_separator: bool) -> str:
        parts = [
            f"Sentence number {i} explains one more aspect of the topic in detail."
            for i in range(n_sentences)
        ]
        text = " ".join(parts)
        if end_with_separator:
            text += " "
        return text

    @staticmethod
    def _normalized_join(chunks) -> str:
        return " ".join(" ".join(c.content.split()) for c in chunks).strip()

    def test_tail_sentence_survives_without_trailing_separator(self):
        para = self._long_paragraph(60, end_with_separator=False)
        chunks = _split(para, 100)
        assert len(chunks) > 1, "paragraph must actually hit the sentence-split path"
        assert chunks[-1].content.rstrip().endswith("detail.")
        assert self._normalized_join(chunks) == " ".join(para.split())

    def test_trailing_separator_variant_is_lossless(self):
        para = self._long_paragraph(60, end_with_separator=True)
        chunks = _split(para, 100)
        assert len(chunks) > 1
        assert self._normalized_join(chunks) == " ".join(para.split())


class TestAudit41NoProgressFallback:
    """Audit 4.1: find-miss on merged $$ paragraphs must not recurse forever."""

    @staticmethod
    def _dollar_doc(blocks: int = 8, filler: int = 80) -> str:
        """Paragraph pairs separated by '\\n \\n' (custom separator).

        Each display-math block merges with its predecessor into a *synthetic*
        string ("para\\n\\n$$...$$") that does not appear verbatim in the
        source, so the find-based split-point search misses on every candidate.
        """
        parts = []
        for i in range(blocks):
            parts.append(f"Paragraph {i} " + ("content " * filler))
            parts.append("$$\ne^{i\\pi} + 1 = 0\n$$")
        return "\n \n".join(parts)

    @staticmethod
    def _normalized_join(chunks) -> str:
        return " ".join(" ".join(c.content.split()) for c in chunks).strip()

    def test_dollar_blocks_with_custom_separator_no_recursion_error(self):
        text = self._dollar_doc()
        chunks = _split(text, 150)  # must not raise RecursionError
        assert len(chunks) > 1
        # Losslessness invariant: split+join equals the source (normalized).
        assert self._normalized_join(chunks) == " ".join(text.split())

    def test_no_progress_falls_back_to_offset_split(self):
        """A total find-miss splits at a character offset instead of recursing
        on identical input — both halves strictly smaller than the whole."""
        text = self._dollar_doc(blocks=6, filler=40)
        chunks = _split(text, 120)
        assert chunks, "must produce chunks"
        assert self._normalized_join(chunks) == " ".join(text.split())
        # The fallback genuinely made progress: more than one chunk.
        assert len(chunks) >= 2

    def test_depth_cap_degrades_to_forced_split(self):
        """The explicit depth cap hard-splits instead of recursing further."""
        splitter = BinarySplitter(TokenBudget(model=MODEL, max_tokens=10))
        text = "AAAA " * 500  # never fits, no sentence separators
        # Call past the cap directly: must return hard-token-split chunks.
        chunks = splitter._split_by_paragraphs_binary(text, 0, 0, splitter._MAX_PARAGRAPH_DEPTH + 1)
        assert chunks
        assert all(c.metadata["type"] == "hard_token_split" for c in chunks)
        # Forced token splits may cut mid-token and trim boundary whitespace;
        # content equality is checked whitespace-insensitively.
        joined = "".join(c.content for c in chunks)
        assert " ".join(joined.split()) == " ".join(text.split())


def test_spans_point_exactly_at_content_after_budget_enforcement():
    """M2/2.25: every chunk span must satisfy original[start:end] == content —
    stripped sentence groups and hard-split pieces used to drift."""
    from ask_llm.core.binary_splitter import BinarySplitter, TokenBudget

    budget = TokenBudget(model="gpt-4", max_tokens=40)
    splitter = BinarySplitter(budget)
    text = (
        "Alpha paragraph one. Alpha paragraph two. Alpha three.\n\n"
        "Beta paragraph one. Beta paragraph two. Beta three.\n\n"
        "Gamma paragraph one. Gamma paragraph two. Gamma three."
    ) * 4
    chunks = splitter.split(text)
    assert len(chunks) > 1
    for c in chunks:
        assert text[c.start_pos : c.end_pos] == c.content
    # Spans are ordered and non-overlapping.
    spans = [(c.start_pos, c.end_pos) for c in sorted(chunks, key=lambda c: c.chunk_id)]
    for (_, prev_end), (next_start, _) in pairwise(spans):
        assert prev_end <= next_start
