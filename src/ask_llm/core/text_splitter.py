"""Text splitting primitives for translation.

Keeps the shared :class:`TextChunk` model, :func:`detect_file_type`, and the
shared position-aware chunk joiner. The concrete splitters and the
``TextSplitter`` ABC were removed (P3.2 and the 2.25 refactor): the only live
split algorithm is ``ask_llm.core.binary_splitter.BinarySplitter`` with a
``TokenBudget`` (construct directly, or via ``create_markdown_splitter``).
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


# Chunk types produced by artificial contiguous cuts (hard splits). Between
# two such chunks an empty separator means "cut mid-content": rejoin
# verbatim, not with a blank line. ``fence_aware_group`` belongs here too —
# its chunks are concatenated directly while packing, so their spans are
# contiguous in the original text.
HARD_SPLIT_TYPES = frozenset({"character_split", "hard_token_split", "fence_aware_group"})


def join_chunks_position_aware(
    parts: list[str],
    spans: list[tuple[int, int]],
    original_text: str,
    types: list[str] | None = None,
) -> str | None:
    """Join per-chunk texts using separators recovered from the original text.

    Position-aware reassembly (P3.4, review §4.4.4): the splitter records each
    chunk's ``start_pos``/``end_pos`` in the original document, so the exact
    original inter-chunk whitespace (single newline between list items, blank
    lines, etc.) can be restored instead of forcing ``\\n\\n`` everywhere.
    Between two hard-split chunks (contiguous artificial cut) an empty
    separator rejoins verbatim.

    Shared by the body formatter, the translation exporters and the notebook
    translator — all of them reassemble per-chunk LLM output back into one
    document and must not inject blank lines the original did not have.

    Args:
        parts: Per-chunk texts (translations or formatted bodies), in order.
        spans: ``(start, end)`` span of each chunk in ``original_text``.
        original_text: The document the spans refer to.
        types: Optional per-chunk ``metadata["type"]`` values, used to detect
            verbatim hard-cut boundaries.

    Returns:
        The joined text, or ``None`` when the spans do not describe a clean
        ordered partition of ``original_text`` (callers fall back to a simple
        separator join).
    """
    if not parts:
        return ""
    if len(parts) != len(spans):
        return None
    if types is not None and len(types) != len(parts):
        return None
    result = parts[0]
    for i in range(1, len(parts)):
        prev_end, cur_start = spans[i - 1][1], spans[i][0]
        if not (0 <= prev_end <= cur_start <= len(original_text)):
            return None
        sep = original_text[prev_end:cur_start]
        if sep.strip():
            # Non-whitespace between two chunks: positions are not a clean
            # partition; do not guess.
            return None
        if sep == "":
            both_hard = (
                types is not None
                and types[i - 1] in HARD_SPLIT_TYPES
                and types[i] in HARD_SPLIT_TYPES
            )
            if both_hard:
                # Artificial contiguous cut: rejoin verbatim, no stripping.
                result = result + parts[i]
                continue
            sep = "\n\n"
        result = result.rstrip("\n") + sep + parts[i].lstrip("\n")
    return result


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
