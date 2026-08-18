"""Re-balance translation chunks by estimated tokens to reduce parallel tail latency."""

from __future__ import annotations

from loguru import logger

from ask_llm.core.binary_splitter import BinarySplitter, TokenBudget
from ask_llm.core.text_splitter import TextChunk

_Meta = dict


def _split_by_token_budget(
    text: str, model: str, max_tokens: int, prompt_overhead: int = 0
) -> list[str]:
    """Split *text* into pieces that each fit the token budget.

    Delegates to :class:`BinarySplitter` with a :class:`TokenBudget` so the
    translation rebalance path shares the single split algorithm and its
    correctness guarantees: fence-aware splitting (no cut mid-code-block,
    review V2 D4 / V1 B4), the approximate-model safety factor, and the
    prompt-overhead reservation (review V2 D1/D2). The previous local
    implementation duplicated the splitter and protected neither fences nor
    CJK undercount.
    """
    text = text.strip()
    if not text:
        return []
    budget = TokenBudget(model=model, max_tokens=max_tokens, prompt_overhead=prompt_overhead)
    return [c.content for c in BinarySplitter(budget).split(text)]


def _merge_adjacent_greedy(
    items: list[tuple[str, _Meta]],
    model: str,
    max_tokens: int,
    prompt_overhead: int = 0,
) -> list[tuple[str, _Meta]]:
    """Merge adjacent translation bodies left-to-right while the combined body fits the budget.

    Chunks are merged as raw markdown/text only; the translation prompt is applied later per
    merged chunk (see ``Translator.prompt_template_for_batch``). The fit test goes through
    :class:`TokenBudget` so merged chunks respect the safety factor and prompt
    overhead, keeping the merge consistent with the split.
    """
    if not items:
        return []
    budget = TokenBudget(model=model, max_tokens=max_tokens, prompt_overhead=prompt_overhead)
    merged: list[tuple[str, _Meta]] = []
    buf_s, buf_m = items[0]
    sep = "\n\n"
    for nxt_s, nxt_m in items[1:]:
        if budget.fits(buf_s + sep + nxt_s):
            buf_s = buf_s + sep + nxt_s
            buf_m = {**buf_m, **nxt_m, "rebalanced": True}
        else:
            merged.append((buf_s, buf_m))
            buf_s, buf_m = nxt_s, nxt_m
    merged.append((buf_s, buf_m))
    return merged


def plain_text_chunks_by_tokens(
    text: str, model: str, max_chunk_tokens: int, prompt_overhead: int = 0
) -> list[TextChunk]:
    """Split plain text into TextChunks; each piece fits the token budget (before merge pass)."""
    parts = _split_by_token_budget(text, model, max_chunk_tokens, prompt_overhead)
    out: list[TextChunk] = []
    start = 0
    for i, content in enumerate(parts):
        end = start + len(content)
        out.append(
            TextChunk(
                content=content,
                chunk_id=i,
                start_pos=start,
                end_pos=end,
                metadata={"type": "token_budget"},
            )
        )
        start = end
    return out


def rebalance_translation_chunks(
    chunks: list[TextChunk],
    model: str,
    *,
    max_chunk_tokens: int = 2400,
    enabled: bool = True,
    prompt_overhead: int = 0,
) -> list[TextChunk]:
    """Split oversized chunks and merge tiny neighbors so estimated input tokens are more uniform.

    Reduces long-tail API latency when many chunks are translated in parallel (wall-clock is
    dominated by the slowest request).

    Args:
        chunks: Chunks from TextSplitter
        model: Model name for tiktoken mapping
        max_chunk_tokens: Max body tokens per chunk after split+merge (prompt is added per chunk)
        enabled: When False, return chunks unchanged
        prompt_overhead: Tokens reserved for the per-chunk translation prompt template (review V2 D2)

    Returns:
        New list of TextChunk with sequential chunk_id 0..n-1
    """
    if not enabled or not chunks:
        return chunks

    pieces: list[tuple[str, _Meta]] = []
    for c in sorted(chunks, key=lambda x: x.chunk_id):
        base_meta = dict(c.metadata)
        for part in _split_by_token_budget(c.content, model, max_chunk_tokens, prompt_overhead):
            pieces.append((part, {**base_meta, "rebalanced": True}))

    pieces = _merge_adjacent_greedy(pieces, model, max_chunk_tokens, prompt_overhead)

    out: list[TextChunk] = []
    start = 0
    for i, (content, meta) in enumerate(pieces):
        end = start + len(content)
        out.append(
            TextChunk(
                content=content,
                chunk_id=i,
                start_pos=start,
                end_pos=end,
                metadata=meta,
            )
        )
        start = end

    if len(out) != len(chunks):
        logger.debug(
            f"Translation chunk rebalance: {len(chunks)} -> {len(out)} chunks "
            f"(max_chunk_tokens={max_chunk_tokens})"
        )
    return out
