"""Unit tests for markdown heading formatter module."""

import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ask_llm.core.md_heading_formatter import (
    HeadingApplier,
    HeadingExtractor,
    HeadingFormatter,
    HeadingMatch,
)
from ask_llm.core.models import ProcessingResult, RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol


class TestHeadingExtractor:
    """Test HeadingExtractor."""

    def test_extract_headings_basic(self):
        """Test extracting basic headings."""
        text = """# Title
Content here.

## Section 1
More content.

### Subsection
Even more.
"""
        headings = HeadingExtractor.extract(text)

        assert len(headings) == 3
        assert headings[0].level == 1
        assert headings[0].title == "Title"
        assert headings[1].level == 2
        assert headings[1].title == "Section 1"
        assert headings[2].level == 3
        assert headings[2].title == "Subsection"

    def test_extract_headings_with_numbering(self):
        """Test extracting headings with numbering."""
        text = """# Title
# 1 First Section
# 1.1 Subsection
# 1.1.1 Sub-subsection
"""
        headings = HeadingExtractor.extract(text)

        assert len(headings) == 4
        assert headings[0].title == "Title"
        assert headings[1].title == "1 First Section"
        assert headings[2].title == "1.1 Subsection"
        assert headings[3].title == "1.1.1 Sub-subsection"

    def test_extract_headings_empty(self):
        """Test extracting headings from text with no headings."""
        text = """This is just regular text.
No headings here.
"""
        headings = HeadingExtractor.extract(text)

        assert len(headings) == 0

    def test_extract_headings_positions(self):
        """Test that heading positions are correctly recorded."""
        text = """# First
Content.

## Second
More content.
"""
        headings = HeadingExtractor.extract(text)

        assert headings[0].start_pos < headings[1].start_pos
        assert headings[0].end_pos < headings[1].start_pos
        assert headings[1].start_pos < headings[1].end_pos

    def test_extract_headings_all_levels(self):
        """Test extracting headings at all levels."""
        text = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6
"""
        headings = HeadingExtractor.extract(text)

        assert len(headings) == 6
        for i, heading in enumerate(headings, 1):
            assert heading.level == i


class TestHeadingFormatter:
    """Test HeadingFormatter."""

    def _create_mock_processor(self, response: str) -> RequestProcessor:
        """Create a mock processor with predefined response."""
        mock_provider = MagicMock(spec=LLMProviderProtocol)
        mock_provider.name = "mock"
        mock_provider.default_model = "mock-model"
        mock_provider.config = MagicMock()
        mock_provider.config.api_temperature = 0.7

        processor = RequestProcessor(mock_provider)

        def mock_process_with_metadata(*args, **kwargs):
            return ProcessingResult(
                content=response,
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

    def test_format_headings_basic(self):
        """Test formatting headings with basic response."""
        headings = [
            HeadingMatch("# Title", 0, 7, 1, "Title"),
            HeadingMatch("# 1 Section", 10, 22, 1, "1 Section"),
        ]

        mock_response = """# Title
## 1 Section"""

        processor = self._create_mock_processor(mock_response)
        formatter = HeadingFormatter(processor=processor)

        formatted = formatter.format_headings(headings)

        assert len(formatted) == 2
        assert formatted[0] == "# Title"
        assert formatted[1] == "## 1 Section"

    def test_format_headings_with_numbering(self):
        """Test formatting headings with numbering."""
        headings = [
            HeadingMatch("# Title", 0, 7, 1, "Title"),
            HeadingMatch("# 1 Section", 10, 22, 1, "1 Section"),
            HeadingMatch("# 1.1 Subsection", 25, 42, 1, "1.1 Subsection"),
            HeadingMatch("# 1.1.1 Sub-subsection", 45, 69, 1, "1.1.1 Sub-subsection"),
        ]

        mock_response = """# Title
## 1 Section
### 1.1 Subsection
#### 1.1.1 Sub-subsection"""

        processor = self._create_mock_processor(mock_response)
        formatter = HeadingFormatter(processor=processor)

        formatted = formatter.format_headings(headings)

        assert len(formatted) == 4
        assert formatted[0] == "# Title"
        assert formatted[1] == "## 1 Section"
        assert formatted[2] == "### 1.1 Subsection"
        assert formatted[3] == "#### 1.1.1 Sub-subsection"

    def test_format_headings_empty(self):
        """Test formatting empty heading list."""
        processor = self._create_mock_processor("")
        formatter = HeadingFormatter(processor=processor)

        formatted = formatter.format_headings([])

        assert len(formatted) == 0

    def test_parse_formatted_headings_with_extra_text(self):
        """Test parsing formatted headings when LLM adds extra text."""
        headings = [
            HeadingMatch("# Title", 0, 7, 1, "Title"),
        ]

        # LLM response with extra text
        mock_response = """Here are the formatted headings:

# Title

That's all!"""

        processor = self._create_mock_processor(mock_response)
        formatter = HeadingFormatter(processor=processor)

        formatted = formatter.format_headings(headings)

        assert len(formatted) == 1
        assert formatted[0] == "# Title"

    def test_parse_formatted_headings_count_mismatch(self):
        """Test handling count mismatch in LLM response."""
        headings = [
            HeadingMatch("# Title", 0, 7, 1, "Title"),
            HeadingMatch("# Section", 10, 20, 1, "Section"),
        ]

        # LLM response with only one heading
        mock_response = """# Title"""

        processor = self._create_mock_processor(mock_response)
        formatter = HeadingFormatter(processor=processor)

        with pytest.raises(RuntimeError, match="LLM API call failed"):
            formatter.format_headings(headings)

    def test_load_prompt_from_file(self):
        """Test loading prompt template from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Custom prompt template\n\n{content}")
            temp_path = f.name

        try:
            processor = self._create_mock_processor("")
            formatter = HeadingFormatter(processor=processor, prompt_file=temp_path)

            assert formatter.prompt_template == "Custom prompt template\n\n{content}"
        finally:
            Path(temp_path).unlink()

    def test_format_headings_batched(self):
        """Test that headings exceeding batch_size are processed in batches with context."""
        # Create 100 headings (exceeds default batch_size of 80)
        headings = [
            HeadingMatch(f"# Heading {i}", i * 15, i * 15 + 12, 1, f"Heading {i}")
            for i in range(100)
        ]

        call_count = 0

        def mock_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            content = kwargs.get("content", args[0] if args else "")
            # Extract heading lines (handles context-aware format with ---)
            heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
            lines = [
                line.strip()
                for line in content.strip().split("\n")
                if line.strip() and heading_pattern.match(line.strip())
            ]
            formatted = [f"## {line.split(' ', 1)[1]}" for line in lines]
            return ProcessingResult(
                content="\n".join(formatted),
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

        mock_provider = MagicMock(spec=LLMProviderProtocol)
        mock_provider.name = "mock"
        mock_provider.default_model = "mock-model"
        mock_provider.config = MagicMock()
        mock_provider.config.api_temperature = 0.7
        processor = RequestProcessor(mock_provider)
        processor.process_with_metadata = mock_process

        formatter = HeadingFormatter(processor=processor, batch_size=50)
        formatted = formatter.format_headings(headings)

        assert len(formatted) == 100
        assert call_count == 2  # 100 headings / 50 per batch = 2 batches

    def test_load_prompt_with_at_prefix(self):
        """Test loading prompt with @ prefix (project-relative)."""
        # Create a temporary project structure
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            prompts_dir = project_root / "prompts"
            prompts_dir.mkdir()
            prompt_file = prompts_dir / "test-format.md"
            prompt_file.write_text("Test prompt\n\n{content}")

            # Create markers to identify project root
            (project_root / "pyproject.toml").write_text("")

            # Change to project root
            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(project_root)

                processor = self._create_mock_processor("")
                formatter = HeadingFormatter(
                    processor=processor, prompt_file="@prompts/test-format.md"
                )

                assert "Test prompt" in formatter.prompt_template
            finally:
                os.chdir(old_cwd)


class TestHeadingApplier:
    """Test HeadingApplier."""

    def test_apply_formatted_headings(self):
        """Test applying formatted headings to text."""
        text = """# Title
Content here.

# 1 Section
More content.
"""

        headings = [
            HeadingMatch("# Title", 0, 7, 1, "Title"),
            HeadingMatch("# 1 Section", 20, 32, 1, "1 Section"),
        ]

        formatted_headings = ["# Title", "## 1 Section"]

        applier = HeadingApplier()
        result = applier.apply(text, headings, formatted_headings)

        assert "## 1 Section" in result
        assert "# Title" in result
        assert "Content here" in result

    def test_apply_preserves_content(self):
        """Test that applying headings preserves non-heading content."""
        text = """# Title
Some content here.

# Section
More content.
"""

        headings = [
            HeadingMatch("# Title", 0, 7, 1, "Title"),
            HeadingMatch("# Section", 25, 35, 1, "Section"),
        ]

        formatted_headings = ["# Title", "## Section"]

        applier = HeadingApplier()
        result = applier.apply(text, headings, formatted_headings)

        assert "Some content here" in result
        assert "More content" in result

    def test_apply_count_mismatch(self):
        """Test that count mismatch raises error."""
        text = "# Title\n"

        headings = [
            HeadingMatch("# Title", 0, 7, 1, "Title"),
        ]

        formatted_headings = ["# Title", "## Section"]  # Mismatch

        applier = HeadingApplier()

        with pytest.raises(ValueError, match="count mismatch"):
            applier.apply(text, headings, formatted_headings)

    def test_apply_reverse_order(self):
        """Test that headings are replaced in reverse order."""
        text = """# First
Content 1.

# Second
Content 2.

# Third
Content 3.
"""

        headings = [
            HeadingMatch("# First", 0, 7, 1, "First"),
            HeadingMatch("# Second", 20, 28, 1, "Second"),
            HeadingMatch("# Third", 40, 48, 1, "Third"),
        ]

        formatted_headings = ["# First", "## Second", "### Third"]

        applier = HeadingApplier()
        result = applier.apply(text, headings, formatted_headings)

        # Verify all headings are replaced correctly
        assert result.count("# First") == 1
        assert result.count("## Second") == 1
        assert result.count("### Third") == 1
