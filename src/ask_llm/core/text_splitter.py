"""Text splitting utilities for translation."""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from loguru import logger
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """A chunk of text with metadata."""

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
            max_chunk_size: Maximum chunk size in characters
        """
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def split(self, text: str) -> List[TextChunk]:
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
            File type ('markdown' or 'text')
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext in (".md", ".markdown"):
            return "markdown"
        return "text"

    @staticmethod
    def create_splitter(file_path: str, max_chunk_size: int = 2000) -> "TextSplitter":
        """
        Create appropriate splitter based on file type.

        Args:
            file_path: Path to file
            max_chunk_size: Maximum chunk size

        Returns:
            TextSplitter instance
        """
        file_type = TextSplitter.detect_file_type(file_path)
        if file_type == "markdown":
            return MarkdownSplitter(max_chunk_size=max_chunk_size)
        return PlainTextSplitter(max_chunk_size=max_chunk_size)


class MarkdownSplitter(TextSplitter):
    """Splitter for Markdown files using heading hierarchy."""

    # Regex to match markdown headings: # Title, ## Title, etc.
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def split(self, text: str) -> List[TextChunk]:
        """
        Split Markdown text using binary splitting strategy.

        Strategy:
        1. If entire text fits in max_chunk_size, return as single chunk
        2. If exceeds, use binary splitting:
           - Prefer splitting by headings (starting from level 1)
           - If no headings or binary split still too large, split by paragraphs
           - Always split in half (binary), not into many small pieces

        Args:
            text: Markdown text to split

        Returns:
            List of text chunks
        """
        # Handle empty text
        if not text.strip():
            return []

        # If entire text fits, return as single chunk
        if len(text) <= self.max_chunk_size:
            logger.debug("Entire text fits in max_chunk_size, returning as single chunk")
            return [
                TextChunk(
                    content=text,
                    chunk_id=0,
                    start_pos=0,
                    end_pos=len(text),
                    metadata={"type": "full_document"},
                )
            ]

        # Find all headings with their positions
        headings = []
        for match in self.HEADING_PATTERN.finditer(text):
            level = len(match.group(1))  # Number of # characters
            title = match.group(2)
            pos = match.start()
            headings.append((level, title, pos))

        if not headings:
            # No headings found, use binary paragraph splitting
            logger.debug("No headings found in Markdown, using binary paragraph splitting")
            return self._split_by_paragraphs_binary(text, 0, 0)

        # Try binary splitting by headings, starting from level 1
        # We want to split even if individual sections are small, as long as total exceeds max_chunk_size
        for level in range(1, 7):
            chunks = self._split_by_headings_binary(text, headings, level, 0, 0)
            if chunks and len(chunks) > 1:
                # Only return if we actually split into multiple chunks
                logger.debug(
                    f"Successfully split using level {level} headings with binary strategy"
                )
                return chunks
            elif chunks and len(chunks) == 1 and len(chunks[0].content) <= self.max_chunk_size:
                # If we got one chunk that fits, that's fine (entire document fits)
                logger.debug("Entire document fits as single chunk")
                return chunks

        # If all heading levels failed, fall back to binary paragraph splitting
        logger.debug("All heading levels failed, using binary paragraph splitting")
        return self._split_by_paragraphs_binary(text, 0, 0)

    def _split_by_headings_binary(
        self,
        text: str,
        headings: List[tuple],
        target_level: int,
        start_pos: int,
        start_chunk_id: int,
    ) -> List[TextChunk]:
        """
        Split text by headings using binary strategy (split in half).

        Args:
            text: Text to split
            headings: List of (level, title, position) tuples
            target_level: Target heading level (1-6)
            start_pos: Starting position in original text
            start_chunk_id: Starting chunk ID

        Returns:
            List of text chunks
        """
        # Filter headings of target level within the current range
        text_start = start_pos
        text_end = start_pos + len(text)
        target_headings = [
            (level, title, pos)
            for level, title, pos in headings
            if level == target_level and text_start <= pos < text_end
        ]

        if not target_headings:
            return []

        # If current text segment fits, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [
                TextChunk(
                    content=text,
                    chunk_id=start_chunk_id,
                    start_pos=start_pos,
                    end_pos=start_pos + len(text),
                    metadata={"heading_level": target_level, "type": "heading_section"},
                )
            ]

        # Binary split: find middle heading
        # Need at least 2 headings to split
        if len(target_headings) < 2:
            # Only one or zero headings, can't split further by headings
            # Fall back to paragraph splitting
            return []

        mid_idx = len(target_headings) // 2

        # Split at middle heading
        split_pos = target_headings[mid_idx][2]

        # Left part: from start to split_pos
        left_text = text[: split_pos - start_pos]
        # Right part: from split_pos to end
        right_text = text[split_pos - start_pos :]

        chunks = []

        # Recursively split left part
        if left_text.strip():
            left_chunks = self._split_by_headings_binary(
                left_text, headings, target_level, start_pos, start_chunk_id
            )
            if left_chunks:
                chunks.extend(left_chunks)
                start_chunk_id += len(left_chunks)
            else:
                # If binary split by headings failed, try paragraphs
                left_para_chunks = self._split_by_paragraphs_binary(
                    left_text, start_pos, start_chunk_id
                )
                chunks.extend(left_para_chunks)
                start_chunk_id += len(left_para_chunks)

        # Recursively split right part
        if right_text.strip():
            right_chunks = self._split_by_headings_binary(
                right_text, headings, target_level, split_pos, start_chunk_id
            )
            if right_chunks:
                chunks.extend(right_chunks)
            else:
                # If binary split by headings failed, try paragraphs
                right_para_chunks = self._split_by_paragraphs_binary(
                    right_text, split_pos, start_chunk_id
                )
                chunks.extend(right_para_chunks)

        return chunks

    def _split_by_paragraphs_binary(
        self, text: str, start_pos: int, start_chunk_id: int
    ) -> List[TextChunk]:
        """
        Split text by paragraphs using binary strategy (split in half).

        Args:
            text: Text to split
            start_pos: Starting position in original text
            start_chunk_id: Starting chunk ID

        Returns:
            List of text chunks
        """
        # If current text segment fits, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [
                TextChunk(
                    content=text,
                    chunk_id=start_chunk_id,
                    start_pos=start_pos,
                    end_pos=start_pos + len(text),
                    metadata={"type": "paragraph_section"},
                )
            ]

        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            # No paragraphs found, fall back to sentence splitting
            return self._split_long_paragraph(text, start_pos, start_chunk_id)

        # Binary split: find middle paragraph
        mid_idx = len(paragraphs) // 2
        if mid_idx == 0:
            # Only one paragraph, split it by sentences if too long
            return self._split_long_paragraph(text, start_pos, start_chunk_id)

        # Calculate split position
        # Find the position of the middle paragraph in the original text
        split_text_pos = 0
        for i in range(mid_idx):
            # Find paragraph in original text
            para_start = text.find(paragraphs[i], split_text_pos)
            if para_start != -1:
                split_text_pos = para_start + len(paragraphs[i])
                # Skip the double newline
                split_text_pos = text.find("\n\n", split_text_pos)
                if split_text_pos != -1:
                    split_text_pos += 2
                else:
                    split_text_pos = para_start + len(paragraphs[i])

        # Left part: paragraphs before middle
        left_text = text[:split_text_pos].rstrip()
        # Right part: paragraphs from middle onwards
        right_text = text[split_text_pos:].lstrip()

        chunks = []
        current_chunk_id = start_chunk_id

        # Recursively split left part
        if left_text.strip():
            left_chunks = self._split_by_paragraphs_binary(left_text, start_pos, current_chunk_id)
            chunks.extend(left_chunks)
            current_chunk_id += len(left_chunks)

        # Recursively split right part
        if right_text.strip():
            right_start_pos = (
                start_pos + len(left_text) + (len(text) - len(left_text) - len(right_text))
            )  # Account for whitespace
            right_chunks = self._split_by_paragraphs_binary(
                right_text, right_start_pos, current_chunk_id
            )
            chunks.extend(right_chunks)

        return chunks

    def _split_long_paragraph(
        self, paragraph: str, start_pos: int, start_chunk_id: int
    ) -> List[TextChunk]:
        """
        Split a long paragraph by sentence boundaries.

        Args:
            paragraph: Paragraph to split
            start_pos: Starting position in original text
            start_chunk_id: Starting chunk ID

        Returns:
            List of text chunks
        """
        # If paragraph has no sentence boundaries, split by character count
        if not re.search(r"[.!?]+\s+", paragraph):
            # Split by character count
            chunks = []
            chunk_id = start_chunk_id
            current_pos = start_pos
            remaining = paragraph

            while len(remaining) > self.max_chunk_size:
                chunk_content = remaining[: self.max_chunk_size]
                chunks.append(
                    TextChunk(
                        content=chunk_content,
                        chunk_id=chunk_id,
                        start_pos=current_pos,
                        end_pos=current_pos + len(chunk_content),
                        metadata={"type": "character_split"},
                    )
                )
                chunk_id += 1
                current_pos += len(chunk_content)
                remaining = remaining[self.max_chunk_size :]

            # Add remaining chunk
            if remaining.strip():
                chunks.append(
                    TextChunk(
                        content=remaining.strip(),
                        chunk_id=chunk_id,
                        start_pos=current_pos,
                        end_pos=current_pos + len(remaining),
                        metadata={"type": "character_split"},
                    )
                )
            return chunks

        # Split by sentence boundaries (., !, ? followed by space or newline)
        sentences = re.split(r"([.!?]+\s+)", paragraph)
        # Recombine sentences with their punctuation
        combined_sentences = []
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
            # If adding this sentence would exceed max size, start new chunk
            if current_chunk and len(current_chunk) + len(sentence) > self.max_chunk_size:
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
                current_chunk += sentence

        # Add remaining chunk
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

        return chunks


class PlainTextSplitter(TextSplitter):
    """Splitter for plain text files using paragraphs and sentences."""

    def split(self, text: str) -> List[TextChunk]:
        """
        Split plain text using binary splitting strategy.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Handle empty text
        if not text.strip():
            return []

        # If entire text fits, return as single chunk
        if len(text) <= self.max_chunk_size:
            logger.debug("Entire text fits in max_chunk_size, returning as single chunk")
            return [
                TextChunk(
                    content=text,
                    chunk_id=0,
                    start_pos=0,
                    end_pos=len(text),
                    metadata={"type": "full_document"},
                )
            ]

        # Use binary paragraph splitting
        return self._split_by_paragraphs_binary(text, 0, 0)

    def _split_by_paragraphs_binary(
        self, text: str, start_pos: int, start_chunk_id: int
    ) -> List[TextChunk]:
        """
        Split text by paragraphs using binary strategy (split in half).

        Args:
            text: Text to split
            start_pos: Starting position in original text
            start_chunk_id: Starting chunk ID

        Returns:
            List of text chunks
        """
        # If current text segment fits, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [
                TextChunk(
                    content=text,
                    chunk_id=start_chunk_id,
                    start_pos=start_pos,
                    end_pos=start_pos + len(text),
                    metadata={"type": "paragraph_section"},
                )
            ]

        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            # No paragraphs found, fall back to sentence splitting
            return self._split_by_sentences(text, start_pos, start_chunk_id)

        # Binary split: find middle paragraph
        mid_idx = len(paragraphs) // 2
        if mid_idx == 0:
            # Only one paragraph, split it by sentences if too long
            return self._split_by_sentences(text, start_pos, start_chunk_id)

        # Calculate split position
        # Find the position of the middle paragraph in the original text
        split_text_pos = 0
        for i in range(mid_idx):
            # Find paragraph in original text
            para_start = text.find(paragraphs[i], split_text_pos)
            if para_start != -1:
                split_text_pos = para_start + len(paragraphs[i])
                # Skip the double newline
                split_text_pos = text.find("\n\n", split_text_pos)
                if split_text_pos != -1:
                    split_text_pos += 2
                else:
                    split_text_pos = para_start + len(paragraphs[i])

        # Left part: paragraphs before middle
        left_text = text[:split_text_pos].rstrip()
        # Right part: paragraphs from middle onwards
        right_text = text[split_text_pos:].lstrip()

        chunks = []
        current_chunk_id = start_chunk_id

        # Recursively split left part
        if left_text.strip():
            left_chunks = self._split_by_paragraphs_binary(left_text, start_pos, current_chunk_id)
            chunks.extend(left_chunks)
            current_chunk_id += len(left_chunks)

        # Recursively split right part
        if right_text.strip():
            right_start_pos = (
                start_pos + len(left_text) + (len(text) - len(left_text) - len(right_text))
            )  # Account for whitespace
            right_chunks = self._split_by_paragraphs_binary(
                right_text, right_start_pos, current_chunk_id
            )
            chunks.extend(right_chunks)

        return chunks

    def _split_by_sentences(
        self, text: str, start_pos: int, start_chunk_id: int
    ) -> List[TextChunk]:
        """
        Split text by sentence boundaries.

        Args:
            text: Text to split
            start_pos: Starting position in original text
            start_chunk_id: Starting chunk ID

        Returns:
            List of text chunks
        """
        # Split by sentence boundaries (., !, ? followed by space or newline)
        # This regex preserves the punctuation
        sentences = re.split(r"([.!?]+\s+)", text)
        # Recombine sentences with their punctuation
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            else:
                combined_sentences.append(sentences[i])

        chunks = []
        chunk_id = start_chunk_id
        current_chunk = ""
        current_chunk_start = start_pos

        for sentence in combined_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed max size, start new chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                chunks.append(
                    TextChunk(
                        content=current_chunk.strip(),
                        chunk_id=chunk_id,
                        start_pos=current_chunk_start,
                        end_pos=current_chunk_start + len(current_chunk),
                        metadata={"type": "sentence_group"},
                    )
                )
                chunk_id += 1
                current_chunk_start += len(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(
                TextChunk(
                    content=current_chunk.strip(),
                    chunk_id=chunk_id,
                    start_pos=current_chunk_start,
                    end_pos=current_chunk_start + len(current_chunk),
                    metadata={"type": "sentence_group"},
                )
            )

        return chunks
