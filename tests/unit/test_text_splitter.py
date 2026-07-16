"""Tests for text splitter primitives.

The char-based ``MarkdownSplitter`` / ``PlainTextSplitter`` / ``create_splitter``
were removed in v2.17 (P3.2) as dead production code; the live algorithm is
``BinarySplitter`` with ``TokenBudget`` (covered by
``test_markdown_token_splitter.py`` and ``test_binary_splitter.py``).
"""

from ask_llm.core.text_splitter import TextChunk, TextSplitter


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

    def test_detect_file_type_notebook(self):
        """Test notebook file type detection."""
        assert TextSplitter.detect_file_type("nb.ipynb") == "notebook"


class TestTextChunk:
    """Test the shared TextChunk model."""

    def test_defaults(self):
        chunk = TextChunk(content="x", chunk_id=0)
        assert chunk.start_pos == 0
        assert chunk.end_pos == 0
        assert chunk.metadata == {}

    def test_positions(self):
        chunk = TextChunk(content="x", chunk_id=3, start_pos=10, end_pos=20)
        assert chunk.chunk_id == 3
        assert (chunk.start_pos, chunk.end_pos) == (10, 20)
