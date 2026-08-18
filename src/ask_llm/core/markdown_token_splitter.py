"""Markdown splitting by tiktoken budget (translation).

Thin compatibility wrapper (P3.2): the split algorithm lives in
``ask_llm.core.binary_splitter.BinarySplitter`` with a ``TokenBudget``
policy. This class keeps the historical constructor used by existing
callers/tests.
"""

from __future__ import annotations

from ask_llm.core.binary_splitter import BinarySplitter, TokenBudget
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

    def split(self, text: str) -> list[TextChunk]:
        return self._impl.split(text)
