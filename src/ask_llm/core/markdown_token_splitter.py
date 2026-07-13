"""Markdown splitting by tiktoken budget (translation); mirrors MarkdownSplitter structure."""

from __future__ import annotations

import re

from loguru import logger

from ask_llm.core.text_splitter import TextChunk, TextSplitter
from ask_llm.utils.token_counter import TokenCounter


class MarkdownTokenSplitter(TextSplitter):
    """Split Markdown using heading/paragraph binary strategy with a token cap."""

    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    # Matches a markdown code fence: ``` or ~~~ (with optional language / trailing text).
    CODE_FENCE_PATTERN = re.compile(r"^(```|~~~).*$", re.MULTILINE)
    # Matches a display-math block: $$ ... $$ potentially spanning multiple lines.
    DISPLAY_MATH_PATTERN = re.compile(r"^\$\$[\s\S]*?\$\$", re.MULTILINE)

    def __init__(self, model: str, max_chunk_tokens: int):
        super().__init__(max_chunk_size=max_chunk_tokens)
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens

    def _tok(self, s: str) -> int:
        return TokenCounter.count_tokens(s, self.model)

    def _fits(self, s: str) -> bool:
        if not s.strip():
            return True
        return self._tok(s) <= self.max_chunk_tokens

    @classmethod
    def _find_code_fence_ranges(cls, text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` char ranges of fenced code blocks (inclusive).

        An unclosed fence extends to end-of-text. See ARCHITECTURE_REVIEW.md bug B4:
        without this, a ``#`` inside a code fence is treated as a heading and used
        as a split point, and a long fenced block is cut mid-fence.
        """
        ranges: list[tuple[int, int]] = []
        in_code = False
        block_start = 0
        for match in cls.CODE_FENCE_PATTERN.finditer(text):
            if not in_code:
                block_start = match.start()
                in_code = True
            else:
                ranges.append((block_start, match.end()))
                in_code = False
        if in_code:
            ranges.append((block_start, len(text)))
        return ranges

    @classmethod
    def _pos_in_code_fence(cls, ranges: list[tuple[int, int]], pos: int) -> bool:
        return any(start <= pos < end for start, end in ranges)

    def split(self, text: str) -> list[TextChunk]:
        if not text.strip():
            return []

        if self._fits(text):
            logger.debug("Entire text fits in max_chunk_tokens, returning as single chunk")
            return [
                TextChunk(
                    content=text,
                    chunk_id=0,
                    start_pos=0,
                    end_pos=len(text),
                    metadata={"type": "full_document"},
                )
            ]

        fence_ranges = self._find_code_fence_ranges(text)
        headings = []
        for match in self.HEADING_PATTERN.finditer(text):
            pos = match.start()
            # Skip headings that are actually ``#`` lines inside a fenced code block.
            if self._pos_in_code_fence(fence_ranges, pos):
                continue
            level = len(match.group(1))
            title = match.group(2)
            headings.append((level, title, pos))

        if not headings:
            logger.debug("No headings found in Markdown, using binary paragraph splitting (tokens)")
            return self._split_by_paragraphs_binary(text, 0, 0)

        for level in range(1, 7):
            chunks = self._split_by_headings_binary(text, headings, level, 0, 0)
            if chunks and len(chunks) > 1:
                logger.debug(
                    f"Successfully split using level {level} headings with binary strategy (tokens)"
                )
                return chunks
            if chunks and len(chunks) == 1 and self._fits(chunks[0].content):
                logger.debug("Entire document fits as single chunk")
                return chunks

        logger.debug("All heading levels failed, using binary paragraph splitting (tokens)")
        return self._split_by_paragraphs_binary(text, 0, 0)

    def _split_by_headings_binary(
        self,
        text: str,
        headings: list[tuple],
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

        if self._fits(text):
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
        if self._fits(text):
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
        # sentence regex can match code punctuation and the token hard-split
        # ignores structure entirely. Split on fence boundaries first, keeping
        # each fenced block as one atomic segment. See B4.
        if self.CODE_FENCE_PATTERN.search(paragraph):
            return self._split_paragraph_with_fences(paragraph, start_pos, start_chunk_id)

        if not re.search(r"[.!?]+\s+", paragraph):
            chunks: list[TextChunk] = []
            chunk_id = start_chunk_id
            current_pos = start_pos
            for piece in TokenCounter.split_hard_by_max_tokens(
                paragraph, self.max_chunk_tokens, self.model
            ):
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
            if current_chunk and self._tok(cand) > self.max_chunk_tokens:
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

        return self._enforce_max_tokens_on_chunks(chunks, start_chunk_id)

    def _split_paragraph_with_fences(
        self, paragraph: str, start_pos: int, start_chunk_id: int
    ) -> list[TextChunk]:
        """Split a paragraph containing code fences without breaking any fence.

        Segments the paragraph into ``[text, fence_block, text, ...]`` keeping each
        fenced block intact, then greedily packs segments under the token budget.
        A single fenced block larger than the budget is the only case that can still
        be hard-split (unavoidable); ordinary text around fences is never cut
        mid-fence.
        """
        fence_ranges = self._find_code_fence_ranges(paragraph)
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
            if current and self._tok(cand) > self.max_chunk_tokens:
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

        return self._enforce_max_tokens_on_chunks(chunks, start_chunk_id)

    def _enforce_max_tokens_on_chunks(
        self, chunks: list[TextChunk], start_chunk_id: int
    ) -> list[TextChunk]:
        """Hard-split any chunk that still exceeds the token budget (e.g. one very long sentence)."""
        if not chunks:
            return []
        out: list[TextChunk] = []
        nid = start_chunk_id
        pos = chunks[0].start_pos
        for ch in chunks:
            if self._tok(ch.content) <= self.max_chunk_tokens:
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
            for piece in TokenCounter.split_hard_by_max_tokens(
                ch.content, self.max_chunk_tokens, self.model
            ):
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
