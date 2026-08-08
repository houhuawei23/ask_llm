"""Text splitting primitives for translation.

Keeps the shared :class:`TextChunk` model and the :class:`TextSplitter` base
(with ``detect_file_type``). The concrete char-based ``MarkdownSplitter`` /
``PlainTextSplitter`` / ``create_splitter`` were dead in production and were
removed in v2.17 (P3.2); the live split algorithm is
``ask_llm.core.binary_splitter.BinarySplitter`` with a ``TokenBudget``
(see ``markdown_token_splitter.MarkdownTokenSplitter`` for the compat
wrapper).
"""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """A chunk of text with metadata.

    Chunk-id convention (P3.7, single convention): every producer —
    ``BinarySplitter``, ``plain_text_chunks_by_tokens``, and
    ``rebalance_translation_chunks`` — emits **dense, zero-based ids in
    document order** (``0..n-1``). Rebalancing may renumber, but only ever
    to another dense zero-based sequence, so ``chunk_id`` is always a valid
    positional index into the chunk list it came from.
    """

    content: str = Field(..., description="Chunk content")
    chunk_id: int = Field(..., description="Chunk ID")
    start_pos: int = Field(default=0, description="Start position in original text")
    end_pos: int = Field(default=0, description="End position in original text")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class TextSplitter(ABC):
    """Base class for text splitters."""

    def __init__(self, max_chunk_size: int = 2000):
        """
        Initialize text splitter.

        Args:
            max_chunk_size: Maximum chunk size (interpretation depends on the
                concrete splitter: characters historically, tokens for the
                token-budget splitters)
        """
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def split(self, text: str) -> list[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """

    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """
        Detect file type based on extension.

        Args:
            file_path: Path to file

        Returns:
            File type ('markdown', 'text', or 'notebook')
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".ipynb":
            return "notebook"
        if ext in (".md", ".markdown"):
            return "markdown"
        return "text"
