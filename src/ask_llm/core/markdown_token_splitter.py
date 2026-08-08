"""Markdown splitting by tiktoken budget (translation).

Thin compatibility wrapper (P3.2): the split algorithm lives in
``ask_llm.core.binary_splitter.BinarySplitter`` with a ``TokenBudget``
policy. This class keeps the historical constructor and helper shims used by
existing callers/tests.
"""

from __future__ import annotations

import re

from ask_llm.core.binary_splitter import BinarySplitter, TokenBudget
from ask_llm.core.markdown_structure import (
    CODE_FENCE_PATTERN,
    HEADING_PATTERN,
    MarkdownStructure,
)
from ask_llm.core.text_splitter import TextChunk, TextSplitter
from ask_llm.utils.token_counter import TokenCounter


class MarkdownTokenSplitter(TextSplitter):
    """Split Markdown using heading/paragraph binary strategy with a token cap."""

    # Kept for backward compatibility; canonical definitions live in
    # ask_llm.core.markdown_structure.
    HEADING_PATTERN = HEADING_PATTERN
    CODE_FENCE_PATTERN = CODE_FENCE_PATTERN
    # Matches a display-math block: $$ ... $$ potentially spanning multiple lines.
    DISPLAY_MATH_PATTERN = re.compile(r"^\$\$[\s\S]*?\$\$", re.MULTILINE)

    def __init__(self, model: str, max_chunk_tokens: int, prompt_overhead_tokens: int = 0):
        super().__init__(max_chunk_size=max_chunk_tokens)
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens
        self.prompt_overhead_tokens = prompt_overhead_tokens
        self._budget = TokenBudget(
            model=model, max_tokens=max_chunk_tokens, prompt_overhead=prompt_overhead_tokens
        )
        self._impl = BinarySplitter(self._budget)

    def _tok(self, s: str) -> int:
        return TokenCounter.count_tokens(s, self.model)

    def _fits(self, s: str) -> bool:
        return self._budget.fits(s)

    @classmethod
    def _find_code_fence_ranges(cls, text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` char ranges of fenced code blocks (inclusive).

        An unclosed fence extends to end-of-text. See ARCHITECTURE_REVIEW.md bug B4:
        without this, a ``#`` inside a code fence is treated as a heading and used
        as a split point, and a long fenced block is cut mid-fence.
        """
        return MarkdownStructure.parse(text).fence_ranges

    @classmethod
    def _pos_in_code_fence(cls, ranges: list[tuple[int, int]], pos: int) -> bool:
        return any(start <= pos < end for start, end in ranges)

    def split(self, text: str) -> list[TextChunk]:
        return self._impl.split(text)
