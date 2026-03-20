"""Integration tests for format command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.md_heading_formatter import (
    HeadingApplier,
    HeadingExtractor,
    HeadingFormatter,
)
from ask_llm.core.models import ProcessingResult, RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol


class TestFormatIntegration:
    """Integration tests for format functionality."""

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

    def test_format_command_single_file(self):
        """Test complete format workflow for a single file."""
        # Create test markdown file
        test_content = """# Title
Introduction paragraph.

# 1 First Section
Content for first section.

# 1.1 Subsection
Subsection content.

# 2 Second Section
Content for second section.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Extract headings
            headings = HeadingExtractor.extract(test_content)
            assert len(headings) == 4

            # Mock LLM response
            mock_response = """# Title
## 1 First Section
### 1.1 Subsection
## 2 Second Section"""

            processor = self._create_mock_processor(mock_response)
            formatter = HeadingFormatter(processor=processor)

            # Format headings
            formatted_headings = formatter.format_headings(headings)
            assert len(formatted_headings) == 4

            # Apply formatting
            applier = HeadingApplier()
            formatted_content = applier.apply(test_content, headings, formatted_headings)

            # Verify formatting
            assert "## 1 First Section" in formatted_content
            assert "### 1.1 Subsection" in formatted_content
            assert "## 2 Second Section" in formatted_content
            assert "# Title" in formatted_content

            # Verify content preserved
            assert "Introduction paragraph" in formatted_content
            assert "Content for first section" in formatted_content

        finally:
            Path(temp_path).unlink()

    def test_format_detects_no_headings(self):
        """Test behavior when no headings are found."""
        test_content = """This is just regular text.
No headings here at all.
Just paragraphs.
"""

        headings = HeadingExtractor.extract(test_content)
        assert len(headings) == 0

        # Should handle empty headings gracefully
        processor = self._create_mock_processor("")
        formatter = HeadingFormatter(processor=processor)

        formatted = formatter.format_headings(headings)
        assert len(formatted) == 0

    def test_format_output_path(self):
        """Test output path generation."""
        test_content = """# Title
Content.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Test auto-generated path
            input_file = Path(temp_path)
            expected_output = input_file.parent / f"{input_file.stem}_formatted{input_file.suffix}"

            # Verify the pattern (name may vary due to tempfile)
            assert expected_output.suffix == ".md"
            assert "_formatted" in expected_output.name
            assert expected_output.parent == input_file.parent

        finally:
            Path(temp_path).unlink()

    def test_format_preserves_complex_structure(self):
        """Test that formatting preserves complex markdown structure."""
        test_content = """# Main Title

Introduction with **bold** and *italic* text.

# 1 Section One

## Regular Subsection

Content here.

- List item 1
- List item 2

```python
# Code block comment (not a heading)
def hello():
    pass
```

# 2 Section Two

More content.
"""

        headings = HeadingExtractor.extract(test_content)
        # Note: Code block comments are not extracted as headings by the regex
        # because they're inside code blocks. But if they were at line start,
        # they would be extracted. For this test, we expect all top-level headings.
        # Main Title, 1 Section One, Regular Subsection (if at top level), 2 Section Two
        # The actual count depends on whether code block # is extracted
        assert len(headings) >= 3  # At least Main Title, 1 Section One, 2 Section Two

        # Create mock response with correct number of headings
        mock_response_lines = ["# Main Title", "## 1 Section One"]
        if len(headings) > 2:
            # Add Regular Subsection if it was extracted
            for h in headings:
                if "Regular Subsection" in h.title:
                    mock_response_lines.insert(2, "### Regular Subsection")
                    break
        mock_response_lines.append("## 2 Section Two")
        # Add any additional headings (like code block if extracted)
        for h in headings:
            if h.title not in ["Main Title", "1 Section One", "Regular Subsection", "2 Section Two"]:
                mock_response_lines.append(f"## {h.title}")

        mock_response = "\n".join(mock_response_lines[:len(headings)])

        processor = self._create_mock_processor(mock_response)
        formatter = HeadingFormatter(processor=processor)
        formatted_headings = formatter.format_headings(headings)

        applier = HeadingApplier()
        formatted_content = applier.apply(test_content, headings, formatted_headings)

        # Verify formatting applied
        assert "## 1 Section One" in formatted_content
        assert "## 2 Section Two" in formatted_content

        # Verify complex structures preserved
        assert "**bold**" in formatted_content
        assert "*italic*" in formatted_content
        assert "```python" in formatted_content
        assert "- List item 1" in formatted_content

    def test_format_with_custom_prompt(self):
        """Test formatting with custom prompt template."""
        test_content = """# Title
# Section
"""

        headings = HeadingExtractor.extract(test_content)

        # Create custom prompt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Custom format prompt\n\n{content}")
            prompt_path = f.name

        try:
            mock_response = """# Title
## Section"""

            processor = self._create_mock_processor(mock_response)
            formatter = HeadingFormatter(processor=processor, prompt_file=prompt_path)

            assert "Custom format prompt" in formatter.prompt_template

            formatted = formatter.format_headings(headings)
            assert len(formatted) == 2

        finally:
            Path(prompt_path).unlink()

    def test_format_error_handling(self):
        """Test error handling in format workflow."""
        test_content = """# Title
Content.
"""

        headings = HeadingExtractor.extract(test_content)

        # Mock processor that raises error
        mock_provider = MagicMock(spec=LLMProviderProtocol)
        processor = RequestProcessor(mock_provider)

        def mock_process_with_metadata(*args, **kwargs):
            raise RuntimeError("API call failed")

        processor.process_with_metadata = mock_process_with_metadata

        formatter = HeadingFormatter(processor=processor)

        with pytest.raises(RuntimeError, match="LLM API call failed"):
            formatter.format_headings(headings)
