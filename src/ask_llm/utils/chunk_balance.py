"""Re-balance translation chunks by estimated tokens to reduce parallel tail latency."""

from __future__ import annotations

import re

from loguru import logger

from ask_llm.core.text_splitter import TextChunk
from ask_llm.utils.token_counter import TokenCounter

_Meta = dict


def _split_by_token_budget(text: str, model: str, max_tokens: int) -> list[str]:
    """Split text into pieces each with at most max_tokens (tiktoken estimate)."""
    text = text.strip()
    if not text:
        return []
    if TokenCounter.count_tokens(text, model) <= max_tokens:
        return [text]

    paras = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in paras if p.strip()]
    if len(paras) > 1:
        result: list[str] = []
        buf = ""
        for p in paras:
            cand = f"{buf}\n\n{p}" if buf else p
            if TokenCounter.count_tokens(cand, model) <= max_tokens:
                buf = cand
            else:
                if buf:
                    result.extend(_split_by_token_budget(buf, model, max_tokens))
                if TokenCounter.count_tokens(p, model) <= max_tokens:
                    buf = p
                else:
                    result.extend(TokenCounter.split_hard_by_max_tokens(p, max_tokens, model))
                    buf = ""
        if buf:
            result.extend(_split_by_token_budget(buf, model, max_tokens))
        return result

    return TokenCounter.split_hard_by_max_tokens(text, max_tokens, model)


def _merge_adjacent_greedy(
    items: list[tuple[str, _Meta]],
    model: str,
    max_tokens: int,
) -> list[tuple[str, _Meta]]:
    """
    Merge adjacent translation bodies left-to-right while combined tiktoken count ≤ max_tokens.

    Chunks are merged as raw markdown/text only; the translation prompt is applied later per
    merged chunk (see ``Translator.generate_prompt``).
    """
    if not items:
        return []
    merged: list[tuple[str, _Meta]] = []
    buf_s, buf_m = items[0]
    sep = "\n\n"
    for nxt_s, nxt_m in items[1:]:
        tc = TokenCounter.count_tokens(buf_s + sep + nxt_s, model)
        if tc <= max_tokens:
            buf_s = buf_s + sep + nxt_s
            buf_m = {**buf_m, **nxt_m, "rebalanced": True}
        else:
            merged.append((buf_s, buf_m))
            buf_s, buf_m = nxt_s, nxt_m
    merged.append((buf_s, buf_m))
    return merged


def plain_text_chunks_by_tokens(text: str, model: str, max_chunk_tokens: int) -> list[TextChunk]:
    """Split plain text into TextChunks; each piece is at most max_chunk_tokens (before merge pass)."""
    parts = _split_by_token_budget(text, model, max_chunk_tokens)
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
    min_merge_tokens: int = 400,
    enabled: bool = True,
) -> list[TextChunk]:
    """
    Split oversized chunks and merge tiny neighbors so estimated input tokens are more uniform.

    Reduces long-tail API latency when many chunks are translated in parallel (wall-clock is
    dominated by the slowest request).

    Args:
        chunks: Chunks from TextSplitter
        model: Model name for tiktoken mapping
        max_chunk_tokens: Max body tokens per chunk after split+merge (prompt is added per chunk)
        min_merge_tokens: Unused; kept for call-site compatibility
        enabled: When False, return chunks unchanged

    Returns:
        New list of TextChunk with sequential chunk_id 0..n-1
    """
    if not enabled or not chunks:
        return chunks

    _ = min_merge_tokens  # API / config compat; merge is greedy up to max_chunk_tokens only

    pieces: list[tuple[str, _Meta]] = []
    for c in sorted(chunks, key=lambda x: x.chunk_id):
        base_meta = dict(c.metadata)
        for part in _split_by_token_budget(c.content, model, max_chunk_tokens):
            pieces.append((part, {**base_meta, "rebalanced": True}))

    pieces = _merge_adjacent_greedy(pieces, model, max_chunk_tokens)

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
