"""Markdown splitting by tiktoken budget (translation).

Thin compatibility wrapper (P3.2): the split algorithm lives in
``ask_llm.core.binary_splitter.BinarySplitter`` with a ``TokenBudget``
policy. This class keeps the historical constructor used by existing
callers/tests.
"""

from __future__ import annotations

from ask_llm.core.binary_splitter import BinarySplitter, TokenBudget
from ask_llm.core.markdown_structure import MarkdownStructure
from ask_llm.core.text_splitter import TextChunk, TextSplitter


class MarkdownTokenSplitter(TextSplitter):
    """Split Markdown using heading/paragraph binary strategy with a token cap."""

    def __init__(self, model: str, max_chunk_tokens: int, prompt_overhead_tokens: int = 0):
        super().__init__(max_chunk_size=max_chunk_tokens)
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens
        self.prompt_overhead_tokens = prompt_overhead_tokens
        self._budget = TokenBudget(
            model=model, max_tokens=max_chunk_tokens, prompt_overhead=prompt_overhead_tokens
        )
        self._impl = BinarySplitter(self._budget)

    @staticmethod
    def _find_code_fence_ranges(text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` char ranges of fenced code blocks (inclusive).

        An unclosed fence extends to end-of-text. See ARCHITECTURE_REVIEW.md bug B4:
        without this, a ``#`` inside a code fence is treated as a heading and used
        as a split point, and a long fenced block is cut mid-fence.
        """
        return MarkdownStructure.parse(text).fence_ranges

    def split(self, text: str) -> list[TextChunk]:
        return self._impl.split(text)
