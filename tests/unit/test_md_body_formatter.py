"""Unit tests for markdown body formatter module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.core.md_body_formatter import BodyFormatter
from ask_llm.core.models import ProcessingResult, RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.core.text_splitter import TextChunk

_TEST_PROMPT_TEMPLATE = "Format body:\n\n{content}"


class TestBodyFormatter:
    """Test BodyFormatter."""

    @pytest.fixture(autouse=True)
    def _init_app_config(self, sample_config_file: Path) -> None:
        """BodyFormatter reads format_body from unified config via get_config()."""
        load_result = ConfigLoader.load(sample_config_file)
        set_config(load_result)

    def _create_mock_processor(self, response: str | None = None) -> RequestProcessor:
        """Create a mock processor with predefined or echo response."""
        mock_provider = MagicMock(spec=LLMProviderProtocol)
        mock_provider.name = "mock"
        mock_provider.default_model = "mock-model"
        mock_provider.config = MagicMock()
        mock_provider.config.api_temperature = 0.7

        processor = RequestProcessor(mock_provider)

        def mock_process_with_metadata(*args, **kwargs):
            content = kwargs.get("content", args[0] if args else "")
            # Echo back with a prefix to verify formatting happened
            resp = response if response is not None else f"FORMATTED:\n{content}"
            return ProcessingResult(
                content=resp,
                metadata=RequestMetadata(
                    provider="mock",
                    model="mock-model",
                    temperature=0.7,
                    input_words=10,
                    input_tokens=20,
                    output_words=10,
                    output_tokens=20,
                    latency=0.1,
                ),
            )

        processor.process_with_metadata = mock_process_with_metadata
        return processor

    def test_format_body_single_chunk(self):
        """Test formatting short text that fits in a single chunk."""
        text = "# Title\n\nSome content here.\n\n## Section\nMore content.\n"

        processor = self._create_mock_processor()
        formatter = BodyFormatter(
            processor=processor,
            model="gpt-4",
            prompt_template=_TEST_PROMPT_TEMPLATE,
            max_chunk_tokens=10000,  # Large enough for single chunk
        )

        result, stats = formatter.format_body(text)

        assert "FORMATTED:" in result
        assert "# Title" in result
        assert "Some content here" in result
        assert stats.chunks_processed == 1

    def test_format_body_empty(self):
        """Test formatting empty text."""
        processor = self._create_mock_processor()
        formatter = BodyFormatter(
            processor=processor,
            model="gpt-4",
            prompt_template=_TEST_PROMPT_TEMPLATE,
        )

        result, stats = formatter.format_body("")
        assert result == ""
        assert stats.chunks_processed == 0

        result, stats = formatter.format_body("   \n\n  ")
        assert result == ""
        assert stats.chunks_processed == 0

    def test_format_body_multi_chunk_merge_order(self):
        """Test that multi-chunk results are merged in correct order."""
        # Mock the splitter to return controlled chunks
        chunks = [
            TextChunk(
                content="# Part 1\n\nContent A.\n", chunk_id=0, start_pos=0, end_pos=20, metadata={}
            ),
            TextChunk(
                content="# Part 2\n\nContent B.\n",
                chunk_id=1,
                start_pos=20,
                end_pos=40,
                metadata={},
            ),
            TextChunk(
                content="# Part 3\n\nContent C.\n",
                chunk_id=2,
                start_pos=40,
                end_pos=60,
                metadata={},
            ),
        ]

        call_order = []

        def mock_process(*args, **kwargs):
            content = kwargs.get("content", "")
            call_order.append(content)
            return ProcessingResult(
                content=f"FORMATTED:\n{content}",
                metadata=RequestMetadata(
                    provider="mock",
                    model="mock-model",
                    temperature=0.7,
                    input_words=1,
                    input_tokens=1,
                    output_words=1,
                    output_tokens=1,
                    latency=0.1,
                ),
            )

        processor = self._create_mock_processor()
        processor.process_with_metadata = mock_process

        formatter = BodyFormatter(
            processor=processor,
            model="gpt-4",
            prompt_template=_TEST_PROMPT_TEMPLATE,
        )

        with patch(
            "ask_llm.core.markdown_token_splitter.MarkdownTokenSplitter.split",
            return_value=chunks,
        ):
            result, stats = formatter.format_body("dummy")

        # Each chunk should be present in order
        assert "# Part 1" in result
        assert "# Part 2" in result
        assert "# Part 3" in result
        # Verify order: Part 1 before Part 2 before Part 3
        assert result.index("# Part 1") < result.index("# Part 2")
        assert result.index("# Part 2") < result.index("# Part 3")
        assert stats.chunks_processed == 3

    def test_format_body_concurrency_single_worker(self):
        """Test that concurrency=1 processes chunks sequentially."""
        chunks = [
            TextChunk(content="A", chunk_id=0, start_pos=0, end_pos=1, metadata={}),
            TextChunk(content="B", chunk_id=1, start_pos=1, end_pos=2, metadata={}),
        ]

        call_order = []

        def mock_process(*args, **kwargs):
            content = kwargs.get("content", "")
            call_order.append(content)
            return ProcessingResult(
                content=f"[{content}]",
                metadata=RequestMetadata(
                    provider="mock",
                    model="mock-model",
                    temperature=0.7,
                    input_words=1,
                    input_tokens=1,
                    output_words=1,
                    output_tokens=1,
                    latency=0.1,
                ),
            )

        processor = self._create_mock_processor()
        processor.process_with_metadata = mock_process

        formatter = BodyFormatter(
            processor=processor,
            model="gpt-4",
            prompt_template=_TEST_PROMPT_TEMPLATE,
            concurrency=1,
        )

        with patch(
            "ask_llm.core.markdown_token_splitter.MarkdownTokenSplitter.split",
            return_value=chunks,
        ):
            result, stats = formatter.format_body("dummy")

        # Sequential processing should call in chunk order
        assert call_order == ["A", "B"]
        assert result == "[A]\n\n[B]"
        assert stats.chunks_processed == 2

    def test_load_prompt_from_file(self):
        """Test loading prompt template from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Custom body prompt\n\n{content}")
            temp_path = f.name

        try:
            processor = self._create_mock_processor()
            formatter = BodyFormatter(
                processor=processor,
                model="gpt-4",
                prompt_file=temp_path,
            )

            assert formatter.prompt_template == "Custom body prompt\n\n{content}"
        finally:
            Path(temp_path).unlink()

    def test_missing_prompt_raises(self):
        """Test that missing prompt template raises ValueError."""
        processor = self._create_mock_processor()
        formatter = BodyFormatter(
            processor=processor,
            model="gpt-4",
            prompt_template=None,
            prompt_file=None,
        )

        with pytest.raises(ValueError, match="Prompt template required"):
            formatter.format_body("# Title\n\nContent\n")

    def test_format_body_preserves_structure(self):
        """Test that body formatting preserves markdown structure."""
        text = """# Main Title

This is a paragraph with some text.

## Section One

- Item 1
- Item 2
- Item 3

### Subsection

Here is `inline code` and a block:

```python
def hello():
    print("world")
```

## Section Two

Some math: $E = mc^2$

$$
\\int_a^b f(x) dx
$$

| Col1 | Col2 |
|------|------|
| A    | B    |
"""

        processor = self._create_mock_processor()
        formatter = BodyFormatter(
            processor=processor,
            model="gpt-4",
            prompt_template=_TEST_PROMPT_TEMPLATE,
            max_chunk_tokens=10000,
        )

        result, stats = formatter.format_body(text)

        # Verify structure is preserved (echo response includes original)
        assert "# Main Title" in result
        assert "## Section One" in result
        assert "### Subsection" in result
        assert "```python" in result
        assert "$E = mc^2$" in result
        assert "| Col1 | Col2 |" in result
        assert stats.chunks_processed == 1

    def test_join_chunks_prevents_markdown_gluing(self):
        """Test that _join_chunks inserts blank lines between chunks.

        When paragraphs are split and processed separately, the original
        blank-line separators can be lost. This verifies that headings,
        images, and blockquotes are not glued to preceding content.
        """
        # Simulate two chunks that would otherwise glue together
        chunks = [
            "![image](path.png)",
            "## 13.1 Heading",
        ]
        result = BodyFormatter._join_chunks(chunks)
        assert "![image](path.png)\n\n## 13.1 Heading" in result
        assert "![image](path.png)## 13.1 Heading" not in result

        # Blockquote followed by heading
        chunks = [
            "> Figure caption",
            "## Next Section",
        ]
        result = BodyFormatter._join_chunks(chunks)
        assert "> Figure caption\n\n## Next Section" in result

        # Chunks that already end/start with newlines should normalize
        chunks = [
            "Text\n",
            "\n## Heading",
        ]
        result = BodyFormatter._join_chunks(chunks)
        assert "Text\n\n## Heading" in result
        assert "Text\n\n\n## Heading" not in result

    def test_load_prompt_with_at_prefix(self, sample_config_file):
        """Test loading prompt with @ prefix (project-relative)."""
        load_result = ConfigLoader.load(str(sample_config_file))
        set_config(load_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            prompts_dir = project_root / "prompts"
            prompts_dir.mkdir()
            prompt_file = prompts_dir / "test-body.md"
            prompt_file.write_text("Test body prompt\n\n{content}")

            (project_root / "pyproject.toml").write_text("")

            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(project_root)

                processor = self._create_mock_processor()
                formatter = BodyFormatter(
                    processor=processor,
                    model="gpt-4",
                    prompt_file="@prompts/test-body.md",
                )

                assert "Test body prompt" in formatter.prompt_template
            finally:
                os.chdir(old_cwd)
