"""Budget-pluggable binary Markdown splitter (P3.2).

Single split algorithm, pluggable budget policy. The historical pair of
~80%-duplicate splitters (char-based ``MarkdownSplitter`` in text_splitter.py
and token-based ``MarkdownTokenSplitter``) is replaced by this module:
the char-based classes were dead in production and are deleted; the token
policy lives here as :class:`TokenBudget`.

``TokenBudget`` accepts a ``prompt_overhead`` so the budget covers
``prompt + content`` instead of content alone (review §4.4.4: a large
template plus a near-full context window could overflow with a
content-only budget).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from ask_llm.core.constants import APPROX_TOKEN_SAFETY_FACTOR
from ask_llm.core.markdown_structure import MarkdownStructure
from ask_llm.core.text_splitter import TextChunk
from ask_llm.utils.token_counter import TokenCounter


class BudgetPolicy(Protocol):
    """Decides whether a text fits the budget and measures it."""

    def fits(self, text: str) -> bool: ...

    def count(self, text: str) -> int: ...

    def hard_split(self, text: str) -> list[str]: ...


@dataclass(frozen=True)
class TokenBudget:
    """Token-count budget backed by ``TokenCounter``.

    The single owner of chunk-sizing correctness. Three concerns converge here:

    - ``prompt_overhead`` reserves tokens for the prompt template accompanying
      each chunk, so the budget covers ``prompt + content`` (review §4.4.4).
    - ``APPROX_TOKEN_SAFETY_FACTOR`` is applied for models whose real BPE is
      approximated by cl100k_base (DeepSeek/Qwen undercount CJK). Applying it
      in ``content_max_tokens`` — not just in the hard-split path — closes the
      fast-path overflow: a chunk that "fits" by raw cl100k count no longer
      admits content that overflows the provider's real context window
      (review V2 D1 / V1 B2).

    Attributes:
        model: Model name used for the tokenizer.
        max_tokens: Maximum total tokens per chunk, including prompt overhead.
        prompt_overhead: Tokens reserved for the prompt template that will
            accompany each chunk.
    """

    model: str
    max_tokens: int
    prompt_overhead: int = 0

    @property
    def _raw_content_cap(self) -> int:
        """Content cap before the approximate-model safety reduction."""
        return max(1, self.max_tokens - self.prompt_overhead)

    @property
    def content_max_tokens(self) -> int:
        """Effective per-chunk content cap (prompt overhead + safety factor)."""
        cap = self._raw_content_cap
        if TokenCounter.is_approximate_model(self.model):
            return max(1, int(cap * APPROX_TOKEN_SAFETY_FACTOR))
        return cap

    def count(self, text: str) -> int:
        return TokenCounter.count_tokens(text, self.model)

    def fits(self, text: str) -> bool:
        if not text.strip():
            return True
        return self.count(text) <= self.content_max_tokens

    def hard_split(self, text: str) -> list[str]:
        # Pass the raw cap; ``split_hard_by_max_tokens`` applies the safety
        # factor internally for approximate models, so the effective budget
        # matches ``content_max_tokens`` without double-applying it.
        return TokenCounter.split_hard_by_max_tokens(text, self._raw_content_cap, self.model)


def locate_pieces(source: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Map each piece to ``(start, length)`` within *source* (M2/2.25).

    Uses a monotonic find-cursor; a piece that can't be located verbatim (the
    splitter strips or synthesizes content) falls back to the whole source
    span rather than reporting a made-up offset. Moved here from
    ``utils.chunk_balance`` (which now imports it) so the splitter itself can
    use the same exact piece-location for hard-split spans.
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


def _stripped_span_chunk(raw: str, chunk_id: int, raw_start: int, type_name: str) -> TextChunk:
    """Build a chunk whose span covers exactly its stripped content.

    M2/2.25: the sentence-group path strips content before emitting it but
    used to span the unstripped raw region (``end_pos`` off by the stripped
    whitespace), so persisted spans drifted from the real text. The strip
    offset is now subtracted from the raw start; leftover whitespace between
    chunks appears as inter-span gaps, which position-aware reassembly
    understands.
    """
    content = raw.strip()
    lead = len(raw) - len(raw.lstrip())
    start = raw_start + lead
    return TextChunk(
        content=content,
        chunk_id=chunk_id,
        start_pos=start,
        end_pos=start + len(content),
        metadata={"type": type_name},
    )


class BinarySplitter:
    """Split Markdown with the heading/paragraph binary strategy under a budget.

    Structure facts (fences, frontmatter, headings) come from a single
    :class:`MarkdownStructure` parse; the budget decision is delegated to the
    injected :class:`BudgetPolicy`.
    """

    # Audit 4.1: hard recursion cap for paragraph splitting. The find-based
    # split-point search can make no progress on adversarial text; the offset
    # fallback guarantees strictly smaller halves, and this cap is the last
    # line of defense degrading to a forced token split.
    _MAX_PARAGRAPH_DEPTH: int = 64

    def __init__(self, budget: BudgetPolicy):
        self.budget = budget

    def split(self, text: str) -> list[TextChunk]:
        if not text.strip():
            return []

        if self.budget.fits(text):
            logger.debug("Entire text fits in budget, returning as single chunk")
            return [
                TextChunk(
                    content=text,
                    chunk_id=0,
                    start_pos=0,
                    end_pos=len(text),
                    metadata={"type": "full_document"},
                )
            ]

        structure = MarkdownStructure.parse(text)
        headings = [(h.level, h.title, h.start_pos) for h in structure.headings]

        if not headings:
            logger.debug("No headings found in Markdown, using binary paragraph splitting")
            return self._split_by_paragraphs_binary(text, 0, 0)

        for level in range(1, 7):
            chunks = self._split_by_headings_binary(text, headings, level, 0, 0)
            if chunks and len(chunks) > 1:
                logger.debug(
                    f"Successfully split using level {level} headings with binary strategy"
                )
                return chunks
            if chunks and len(chunks) == 1 and self.budget.fits(chunks[0].content):
                logger.debug("Entire document fits as single chunk")
                return chunks

        logger.debug("All heading levels failed, using binary paragraph splitting")
        return self._split_by_paragraphs_binary(text, 0, 0)

    def _split_by_headings_binary(
        self,
        text: str,
        headings: list[tuple[int, str, int]],
        target_level: int,
        start_pos: int,
        start_chunk_id: int,
    ) -> list[TextChunk]:
        text_start = start_pos
        text_end = start_pos + len(text)

        target_headings = [
            (level, title, pos)
            for level, title, pos in headings
            if level == target_level and text_start <= pos < text_end
        ]

        if not target_headings:
            return []

        if self.budget.fits(text):
            return [
                TextChunk(
                    content=text,
                    chunk_id=start_chunk_id,
                    start_pos=start_pos,
                    end_pos=start_pos + len(text),
                    metadata={"heading_level": target_level, "type": "heading_section"},
                )
            ]

        if len(target_headings) < 2:
            return []

        mid_idx = len(target_headings) // 2
        split_pos = target_headings[mid_idx][2]

        left_text = text[: split_pos - start_pos]
        right_text = text[split_pos - start_pos :]

        chunks: list[TextChunk] = []

        if left_text.strip():
            left_chunks = self._split_by_headings_binary(
                left_text, headings, target_level, start_pos, start_chunk_id
            )
            if left_chunks:
                chunks.extend(left_chunks)
                start_chunk_id += len(left_chunks)
            else:
                left_para_chunks = self._split_by_paragraphs_binary(
                    left_text, start_pos, start_chunk_id
                )
                chunks.extend(left_para_chunks)
                start_chunk_id += len(left_para_chunks)

        if right_text.strip():
            right_chunks = self._split_by_headings_binary(
                right_text, headings, target_level, split_pos, start_chunk_id
            )
            if right_chunks:
                chunks.extend(right_chunks)
            else:
                right_para_chunks = self._split_by_paragraphs_binary(
                    right_text, split_pos, start_chunk_id
                )
                chunks.extend(right_para_chunks)

        return chunks

    def _split_by_paragraphs_binary(
        self, text: str, start_pos: int, start_chunk_id: int, depth: int = 0
    ) -> list[TextChunk]:
        if self.budget.fits(text):
            return [
                TextChunk(
                    content=text,
                    chunk_id=start_chunk_id,
                    start_pos=start_pos,
                    end_pos=start_pos + len(text),
                    metadata={"type": "paragraph_section"},
                )
            ]

        # Audit 4.1: depth cap — degrade to a forced token split rather than
        # recursing further. Unreachable when the offset fallback below keeps
        # making progress, but guarantees termination on any input.
        if depth > self._MAX_PARAGRAPH_DEPTH:
            logger.warning(
                f"Paragraph split depth cap ({self._MAX_PARAGRAPH_DEPTH}) hit; "
                "forcing a token split for this section."
            )
            chunks: list[TextChunk] = []
            chunk_id = start_chunk_id
            current_pos = start_pos
            for piece in self.budget.hard_split(text):
                chunks.append(
                    TextChunk(
                        content=piece,
                        chunk_id=chunk_id,
                        start_pos=current_pos,
                        end_pos=current_pos + len(piece),
                        metadata={"type": "hard_token_split"},
                    )
                )
                chunk_id += 1
                current_pos += len(piece)
            return chunks

        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return self._split_long_paragraph(text, start_pos, start_chunk_id)

        # Merge display-math blocks ($$...$$) with their preceding paragraph so
        # that a $$ block never starts a chunk on its own (which causes LLMs to
        # drop or garble the equation).
        merged_paragraphs: list[str] = []
        for p in paragraphs:
            if p.startswith("$$") and merged_paragraphs:
                merged_paragraphs[-1] = merged_paragraphs[-1] + "\n\n" + p
            else:
                merged_paragraphs.append(p)
        paragraphs = merged_paragraphs

        mid_idx = len(paragraphs) // 2
        if mid_idx == 0:
            return self._split_long_paragraph(text, start_pos, start_chunk_id)

        split_text_pos = 0
        for i in range(mid_idx):
            para_start = text.find(paragraphs[i], split_text_pos)
            if para_start != -1:
                split_text_pos = para_start + len(paragraphs[i])
                split_text_pos = text.find("\n\n", split_text_pos)
                if split_text_pos != -1:
                    split_text_pos += 2
                else:
                    split_text_pos = para_start + len(paragraphs[i])

        if not 0 < split_text_pos < len(text):
            # Audit 4.1: merged $$ blocks (and stripped paragraphs) need not
            # appear verbatim in the source — with custom "\n \n"-style
            # separators every find() used to miss, leaving split_text_pos == 0
            # and recursing on the IDENTICAL text until RecursionError (or the
            # symmetric full-length case recursing on the whole left half).
            # Fall back to a character-offset split: both halves are strictly
            # smaller, so recursion terminates. Prefer a whitespace boundary
            # near the midpoint so chunks don't start/end mid-word.
            mid = max(1, len(text) // 2)
            lo, hi = max(1, mid - 200), min(len(text), mid + 200)
            candidates = [i for i in range(lo, hi) if text[i].isspace()]
            split_text_pos = min(candidates, key=lambda i: abs(i - mid)) if candidates else mid
            logger.debug("Paragraph find-miss; falling back to character-offset split.")

        left_text = text[:split_text_pos].rstrip()
        right_text = text[split_text_pos:].lstrip()

        chunks = []
        current_chunk_id = start_chunk_id

        if left_text.strip():
            left_chunks = self._split_by_paragraphs_binary(
                left_text, start_pos, current_chunk_id, depth + 1
            )
            chunks.extend(left_chunks)
            current_chunk_id += len(left_chunks)

        if right_text.strip():
            right_start_pos = start_pos + len(text) - len(right_text)
            right_chunks = self._split_by_paragraphs_binary(
                right_text, right_start_pos, current_chunk_id, depth + 1
            )
            chunks.extend(right_chunks)

        return chunks

    def _split_long_paragraph(
        self, paragraph: str, start_pos: int, start_chunk_id: int
    ) -> list[TextChunk]:
        # A paragraph containing a code fence must never be cut mid-fence: the
        # sentence regex can match code punctuation and the hard split ignores
        # structure entirely. Split on fence boundaries first, keeping each
        # fenced block as one atomic segment. See B4.
        structure = MarkdownStructure.parse(paragraph)
        if structure.fence_ranges:
            return self._split_paragraph_with_fences(paragraph, start_pos, start_chunk_id)

        if not re.search(r"[.!?]+\s+", paragraph):
            chunks: list[TextChunk] = []
            chunk_id = start_chunk_id
            current_pos = start_pos
            for piece in self.budget.hard_split(paragraph):
                chunks.append(
                    TextChunk(
                        content=piece,
                        chunk_id=chunk_id,
                        start_pos=current_pos,
                        end_pos=current_pos + len(piece),
                        metadata={"type": "character_split"},
                    )
                )
                chunk_id += 1
                current_pos += len(piece)
            return chunks

        sentences = re.split(r"([.!?]+\s+)", paragraph)
        combined_sentences: list[str] = []
        # re.split with a capture group returns [text, sep, text, ..., text]: the
        # final element is the tail after the last separator. Stepping by 2 over
        # the full length pairs each text with its following separator and lets
        # the else branch pick up that tail. The former ``len(sentences) - 1``
        # bound dropped it, silently losing the last sentence of every
        # budget-split paragraph.
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            else:
                combined_sentences.append(sentences[i])

        chunks = []
        chunk_id = start_chunk_id
        current_chunk = ""
        current_pos = start_pos

        for sentence in combined_sentences:
            cand = current_chunk + sentence if current_chunk else sentence
            if current_chunk and not self.budget.fits(cand):
                chunks.append(
                    _stripped_span_chunk(current_chunk, chunk_id, current_pos, "sentence_group")
                )
                chunk_id += 1
                current_pos += len(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = cand

        if current_chunk.strip():
            chunks.append(
                _stripped_span_chunk(current_chunk, chunk_id, current_pos, "sentence_group")
            )

        return self._enforce_budget_on_chunks(chunks, start_chunk_id)

    def _split_paragraph_with_fences(
        self, paragraph: str, start_pos: int, start_chunk_id: int
    ) -> list[TextChunk]:
        """Split a paragraph containing code fences without breaking any fence.

        Segments the paragraph into ``[text, fence_block, text, ...]`` keeping each
        fenced block intact, then greedily packs segments under the budget.
        A single fenced block larger than the budget is the only case that can still
        be hard-split (unavoidable); ordinary text around fences is never cut
        mid-fence.
        """
        fence_ranges = MarkdownStructure.parse(paragraph).fence_ranges
        segments: list[str] = []
        pos = 0
        for s, e in fence_ranges:
            if pos < s:
                segments.append(paragraph[pos:s])
            segments.append(paragraph[s:e])
            pos = e
        if pos < len(paragraph):
            segments.append(paragraph[pos:])
        segments = [s for s in segments if s]

        chunks: list[TextChunk] = []
        chunk_id = start_chunk_id
        current = ""
        current_pos = start_pos
        for seg in segments:
            cand = current + seg if current else seg
            if current and not self.budget.fits(cand):
                chunks.append(
                    TextChunk(
                        content=current,
                        chunk_id=chunk_id,
                        start_pos=current_pos,
                        end_pos=current_pos + len(current),
                        metadata={"type": "fence_aware_group"},
                    )
                )
                chunk_id += 1
                current_pos += len(current)
                current = seg
            else:
                current = cand
        if current.strip():
            chunks.append(
                TextChunk(
                    content=current,
                    chunk_id=chunk_id,
                    start_pos=current_pos,
                    end_pos=current_pos + len(current),
                    metadata={"type": "fence_aware_group"},
                )
            )

        return self._enforce_budget_on_chunks(chunks, start_chunk_id)

    def _enforce_budget_on_chunks(
        self, chunks: list[TextChunk], start_chunk_id: int
    ) -> list[TextChunk]:
        """Hard-split any chunk that still exceeds the budget (e.g. one very long sentence).

        M2/2.25: spans survive this pass. Chunks that already fit keep their
        original ``start_pos``/``end_pos`` (they may be narrower than the raw
        source region the caller measured, e.g. stripped sentence groups), and
        hard-split pieces are located verbatim inside the parent content so
        each piece's span points at its true text in the original document —
        instead of being rebuilt cumulatively from content lengths, which
        silently drifted every subsequent span.
        """
        if not chunks:
            return []
        out: list[TextChunk] = []
        nid = start_chunk_id
        for ch in chunks:
            if self.budget.fits(ch.content):
                out.append(
                    TextChunk(
                        content=ch.content,
                        chunk_id=nid,
                        start_pos=ch.start_pos,
                        end_pos=ch.end_pos,
                        metadata=ch.metadata,
                    )
                )
                nid += 1
                continue
            pieces = self.budget.hard_split(ch.content)
            for piece, (rel_start, rel_len) in zip(
                pieces, locate_pieces(ch.content, pieces), strict=False
            ):
                out.append(
                    TextChunk(
                        content=piece,
                        chunk_id=nid,
                        start_pos=ch.start_pos + rel_start,
                        end_pos=ch.start_pos + rel_start + rel_len,
                        metadata={**ch.metadata, "type": "hard_token_split"},
                    )
                )
                nid += 1
        return out
