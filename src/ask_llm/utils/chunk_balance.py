"""Re-balance translation chunks by estimated tokens to reduce parallel tail latency."""

from __future__ import annotations

from loguru import logger

from ask_llm.core.binary_splitter import BinarySplitter, TokenBudget
from ask_llm.core.text_splitter import TextChunk

_Meta = dict


def _merge_meta(a: _Meta, b: _Meta) -> _Meta:
    """Union of two chunk metadata dicts preserving BOTH sides (audit 4.3).

    ``{**a, **b}`` let the right side silently overwrite shared keys — merging
    two chunks that each carry a ``heading`` (or any other context) dropped the
    left one. Conflicting values are collected into ordered lists instead.
    """
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
        elif out[k] == v:
            continue
        else:
            acc = out[k]
            if not isinstance(acc, list):
                acc = [acc]
            additions = v if isinstance(v, list) else [v]
            for item in additions:
                if item not in acc:
                    acc.append(item)
            out[k] = acc
    return out


def _locate_pieces(source: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Map each piece to ``(start, length)`` within *source* (audit 4.3).

    Uses a monotonic find-cursor; a piece that can't be located verbatim (the
    splitter strips or synthesizes content) falls back to the whole source
    span rather than reporting a made-up offset.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for part in pieces:
        pos = source.find(part, cursor) if part else -1
        if pos != -1:
            cursor = pos + len(part)
            spans.append((pos, len(part)))
        else:
            spans.append((0, len(source)))
    return spans


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
    items: list[tuple[str, _Meta, int, int]],
    model: str,
    max_tokens: int,
    prompt_overhead: int = 0,
) -> list[tuple[str, _Meta, int, int]]:
    """Merge adjacent translation bodies left-to-right while the combined body fits the budget.

    Chunks are merged as raw markdown/text only; the translation prompt is applied later per
    merged chunk (see ``Translator.prompt_template_for_batch``). The fit test goes through
    :class:`TokenBudget` so merged chunks respect the safety factor and prompt
    overhead, keeping the merge consistent with the split.

    Each item is ``(content, meta, src_start, src_end)``; the merged entry's
    source span covers first piece's start through last piece's end (audit 4.3)
    and metadata is a loss-free union of both sides.
    """
    if not items:
        return []
    budget = TokenBudget(model=model, max_tokens=max_tokens, prompt_overhead=prompt_overhead)
    merged: list[tuple[str, _Meta, int, int]] = []
    buf_s, buf_m, buf_start, buf_end = items[0]
    sep = "\n\n"
    for nxt_s, nxt_m, nxt_start, nxt_end in items[1:]:
        if budget.fits(buf_s + sep + nxt_s):
            buf_s = buf_s + sep + nxt_s
            buf_m = {**_merge_meta(buf_m, nxt_m), "rebalanced": True}
            buf_end = nxt_end
        else:
            merged.append((buf_s, buf_m, buf_start, buf_end))
            buf_s, buf_m, buf_start, buf_end = nxt_s, nxt_m, nxt_start, nxt_end
    merged.append((buf_s, buf_m, buf_start, buf_end))
    return merged


def plain_text_chunks_by_tokens(
    text: str, model: str, max_chunk_tokens: int, prompt_overhead: int = 0
) -> list[TextChunk]:
    """Split plain text into TextChunks; each piece fits the token budget (before merge pass).

    Spans are source-relative (audit 4.3): each piece is located inside the
    original text instead of being assigned cumulative offsets in a synthetic
    stream that drifts as soon as the splitter strips whitespace.
    """
    parts = _split_by_token_budget(text, model, max_chunk_tokens, prompt_overhead)
    stripped = text.strip()
    prefix = text.find(stripped) if stripped else 0
    out: list[TextChunk] = []
    for i, (content, (rel_start, rel_len)) in enumerate(
        zip(parts, _locate_pieces(stripped, parts), strict=False)
    ):
        out.append(
            TextChunk(
                content=content,
                chunk_id=i,
                start_pos=prefix + rel_start,
                end_pos=prefix + rel_start + rel_len,
                metadata={"type": "token_budget"},
            )
        )
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

    pieces: list[tuple[str, _Meta, int, int]] = []
    for c in sorted(chunks, key=lambda x: x.chunk_id):
        base_meta = dict(c.metadata)
        parts = _split_by_token_budget(c.content, model, max_chunk_tokens, prompt_overhead)
        stripped = c.content.strip()
        prefix = c.content.find(stripped) if stripped else 0
        for part, (rel_start, rel_len) in zip(parts, _locate_pieces(stripped, parts), strict=False):
            pieces.append(
                (
                    part,
                    {**base_meta, "rebalanced": True},
                    c.start_pos + prefix + rel_start,
                    c.start_pos + prefix + rel_start + rel_len,
                )
            )

    pieces = _merge_adjacent_greedy(pieces, model, max_chunk_tokens, prompt_overhead)

    out: list[TextChunk] = []
    for i, (content, meta, src_start, src_end) in enumerate(pieces):
        out.append(
            TextChunk(
                content=content,
                chunk_id=i,
                start_pos=src_start,
                end_pos=src_end,
                metadata=meta,
            )
        )

    if len(out) != len(chunks):
        logger.debug(
            f"Translation chunk rebalance: {len(chunks)} -> {len(out)} chunks "
            f"(max_chunk_tokens={max_chunk_tokens})"
        )
    return out
