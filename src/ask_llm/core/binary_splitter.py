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

    Attributes:
        model: Model name used for the tokenizer.
        max_tokens: Maximum total tokens per chunk, including prompt overhead.
        prompt_overhead: Tokens reserved for the prompt template that will
            accompany each chunk. Budget per chunk content is
            ``max_tokens - prompt_overhead``.
    """

    model: str
    max_tokens: int
    prompt_overhead: int = 0

    @property
    def content_max_tokens(self) -> int:
        """Token cap for chunk content after reserving prompt overhead."""
        return max(1, self.max_tokens - self.prompt_overhead)

    def count(self, text: str) -> int:
        return TokenCounter.count_tokens(text, self.model)

    def fits(self, text: str) -> bool:
        if not text.strip():
            return True
        return self.count(text) <= self.content_max_tokens

    def hard_split(self, text: str) -> list[str]:
        return TokenCounter.split_hard_by_max_tokens(text, self.content_max_tokens, self.model)


# Matches a display-math block: $$ ... $$ potentially spanning multiple lines.
DISPLAY_MATH_PATTERN = re.compile(r"^\$\$[\s\S]*?\$\$", re.MULTILINE)


class BinarySplitter:
    """Split Markdown with the heading/paragraph binary strategy under a budget.

    Structure facts (fences, frontmatter, headings) come from a single
    :class:`MarkdownStructure` parse; the budget decision is delegated to the
    injected :class:`BudgetPolicy`.
    """

    # Kept for backward compatibility (tests / external references).
    DISPLAY_MATH_PATTERN = DISPLAY_MATH_PATTERN

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
        self, text: str, start_pos: int, start_chunk_id: int
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

        left_text = text[:split_text_pos].rstrip()
        right_text = text[split_text_pos:].lstrip()

        chunks: list[TextChunk] = []
        current_chunk_id = start_chunk_id

        if left_text.strip():
            left_chunks = self._split_by_paragraphs_binary(left_text, start_pos, current_chunk_id)
            chunks.extend(left_chunks)
            current_chunk_id += len(left_chunks)

        if right_text.strip():
            right_start_pos = (
                start_pos + len(left_text) + (len(text) - len(left_text) - len(right_text))
            )
            right_chunks = self._split_by_paragraphs_binary(
                right_text, right_start_pos, current_chunk_id
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
        for i in range(0, len(sentences) - 1, 2):
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
                    TextChunk(
                        content=current_chunk.strip(),
                        chunk_id=chunk_id,
                        start_pos=current_pos,
                        end_pos=current_pos + len(current_chunk),
                        metadata={"type": "sentence_group"},
                    )
                )
                chunk_id += 1
                current_pos += len(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = cand

        if current_chunk.strip():
            chunks.append(
                TextChunk(
                    content=current_chunk.strip(),
                    chunk_id=chunk_id,
                    start_pos=current_pos,
                    end_pos=current_pos + len(current_chunk),
                    metadata={"type": "sentence_group"},
                )
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
        """Hard-split any chunk that still exceeds the budget (e.g. one very long sentence)."""
        if not chunks:
            return []
        out: list[TextChunk] = []
        nid = start_chunk_id
        pos = chunks[0].start_pos
        for ch in chunks:
            if self.budget.fits(ch.content):
                out.append(
                    TextChunk(
                        content=ch.content,
                        chunk_id=nid,
                        start_pos=pos,
                        end_pos=pos + len(ch.content),
                        metadata=ch.metadata,
                    )
                )
                nid += 1
                pos += len(ch.content)
                continue
            for piece in self.budget.hard_split(ch.content):
                out.append(
                    TextChunk(
                        content=piece,
                        chunk_id=nid,
                        start_pos=pos,
                        end_pos=pos + len(piece),
                        metadata={**ch.metadata, "type": "hard_token_split"},
                    )
                )
                nid += 1
                pos += len(piece)
        return out
