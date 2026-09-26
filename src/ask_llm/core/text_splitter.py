"""Text splitting primitives for translation.

Keeps the shared :class:`TextChunk` model and :func:`detect_file_type`. The
concrete splitters and the ``TextSplitter`` ABC were removed (P3.2 and the
2.25 refactor): the only live split algorithm is
``ask_llm.core.binary_splitter.BinarySplitter`` with a ``TokenBudget``
(construct directly, or via ``create_markdown_splitter``).
"""

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
