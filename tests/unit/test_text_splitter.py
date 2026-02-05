"""Tests for text splitter module."""

import pytest

from ask_llm.core.text_splitter import (
    MarkdownSplitter,
    PlainTextSplitter,
    TextChunk,
    TextSplitter,
)


class TestTextSplitter:
    """Test TextSplitter base class."""

    def test_detect_file_type_markdown(self):
        """Test Markdown file type detection."""
        assert TextSplitter.detect_file_type("test.md") == "markdown"
        assert TextSplitter.detect_file_type("test.markdown") == "markdown"
        assert TextSplitter.detect_file_type("/path/to/file.md") == "markdown"

    def test_detect_file_type_text(self):
        """Test plain text file type detection."""
        assert TextSplitter.detect_file_type("test.txt") == "text"
        assert TextSplitter.detect_file_type("test") == "text"

    def test_create_splitter_markdown(self):
        """Test creating Markdown splitter."""
        splitter = TextSplitter.create_splitter("test.md")
        assert isinstance(splitter, MarkdownSplitter)

    def test_create_splitter_text(self):
        """Test creating plain text splitter."""
        splitter = TextSplitter.create_splitter("test.txt")
        assert isinstance(splitter, PlainTextSplitter)


class TestPlainTextSplitter:
    """Test PlainTextSplitter."""

    def test_split_simple_paragraphs(self):
        """Test splitting simple paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        # With large max_chunk_size, should return as single chunk
        splitter = PlainTextSplitter(max_chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert text in chunks[0].content

        # With small max_chunk_size, should split
        splitter = PlainTextSplitter(max_chunk_size=30)
        chunks = splitter.split(text)
        assert len(chunks) >= 2  # Should be split into multiple chunks

    def test_split_long_paragraph(self):
        """Test splitting long paragraph by sentences."""
        # Create a long paragraph
        sentences = [f"Sentence {i}. " for i in range(50)]
        text = "".join(sentences)
        splitter = PlainTextSplitter(max_chunk_size=100)
        chunks = splitter.split(text)

        assert len(chunks) > 1
        # Each chunk should be within max size
        for chunk in chunks:
            assert len(chunk.content) <= splitter.max_chunk_size

    def test_split_no_paragraphs(self):
        """Test splitting text without paragraph breaks."""
        text = "This is a single line of text without paragraph breaks."
        splitter = PlainTextSplitter(max_chunk_size=1000)
        chunks = splitter.split(text)

        assert len(chunks) >= 1
        assert chunks[0].content == text.strip()

    def test_split_empty_text(self):
        """Test splitting empty text."""
        splitter = PlainTextSplitter(max_chunk_size=1000)
        chunks = splitter.split("")
        assert len(chunks) == 0

    def test_chunk_metadata(self):
        """Test chunk metadata."""
        text = "First paragraph.\n\nSecond paragraph."
        splitter = PlainTextSplitter(max_chunk_size=1000)
        chunks = splitter.split(text)

        assert chunks[0].chunk_id == 0
        assert "type" in chunks[0].metadata


class TestMarkdownSplitter:
    """Test MarkdownSplitter."""

    def test_split_by_level1_headings(self):
        """Test splitting by level 1 headings."""
        text = """# Title 1

Content for title 1.

## Subtitle

More content.

# Title 2

Content for title 2.
"""
        # With large max_chunk_size, should return as single chunk
        splitter = MarkdownSplitter(max_chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert "# Title 1" in chunks[0].content
        assert "# Title 2" in chunks[0].content

        # With small max_chunk_size, should split by headings
        splitter = MarkdownSplitter(max_chunk_size=50)
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        # Check that chunks contain headings
        assert any("# Title 1" in chunk.content for chunk in chunks)
        assert any("# Title 2" in chunk.content for chunk in chunks)

    def test_split_by_level2_headings(self):
        """Test splitting by level 2 headings when level 1 produces long chunks."""
        text = """# Main Title

## Section 1

Very long content here. """ + "x" * 500 + """

## Section 2

More content.

## Section 3

Even more content.
"""
        splitter = MarkdownSplitter(max_chunk_size=200)
        chunks = splitter.split(text)

        # Should split by level 2 headings
        assert len(chunks) >= 2

    def test_split_no_headings(self):
        """Test splitting Markdown without headings (fallback to paragraphs)."""
        text = """This is a paragraph.

This is another paragraph.

And one more paragraph.
"""
        # With large max_chunk_size, should return as single chunk
        splitter = MarkdownSplitter(max_chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1

        # With small max_chunk_size, should split by paragraphs
        splitter = MarkdownSplitter(max_chunk_size=30)
        chunks = splitter.split(text)
        assert len(chunks) >= 2

    def test_split_preserves_heading_structure(self):
        """Test that heading structure is preserved."""
        text = """# Main Title

Content here.

## Subsection

More content.

### Sub-subsection

Even more.
"""
        splitter = MarkdownSplitter(max_chunk_size=1000)
        chunks = splitter.split(text)

        # Should have at least one chunk with main title
        assert len(chunks) >= 1
        assert "# Main Title" in chunks[0].content

    def test_split_long_chunk_fallback(self):
        """Test fallback when all heading levels produce long chunks."""
        # Create a very long single paragraph
        text = "# Title\n\n" + "x" * 5000
        splitter = MarkdownSplitter(max_chunk_size=100)
        chunks = splitter.split(text)

        # Should fall back to paragraph/sentence splitting
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= splitter.max_chunk_size

    def test_chunk_heading_metadata(self):
        """Test that chunks have heading metadata."""
        text = """# Title 1

Content.

## Subtitle

More content.
"""
        # With large max_chunk_size, returns as full_document
        splitter = MarkdownSplitter(max_chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0].metadata.get("type") == "full_document"

        # With small max_chunk_size, should split and have heading metadata
        splitter = MarkdownSplitter(max_chunk_size=50)
        chunks = splitter.split(text)
        # May have heading metadata if split by headings, or paragraph metadata if split by paragraphs
        assert len(chunks) >= 1
        assert any("type" in chunk.metadata for chunk in chunks)
