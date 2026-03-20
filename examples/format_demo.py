#!/usr/bin/env python3
"""
Demo script for format command.

Demonstrates markdown heading formatting functionality.
Can use mock LLM responses or real API calls.
"""

import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

from ask_llm.core.md_heading_formatter import (
    HeadingApplier,
    HeadingExtractor,
    HeadingFormatter,
)
from ask_llm.core.models import ProcessingResult, RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import LLMProviderProtocol


def create_mock_provider() -> LLMProviderProtocol:
    """Create a mock LLM provider for testing."""
    mock_provider = MagicMock(spec=LLMProviderProtocol)
    mock_provider.name = "mock"
    mock_provider.default_model = "mock-model"
    mock_provider.config = MagicMock()
    mock_provider.config.api_temperature = 0.7
    return mock_provider


def create_mock_processor(mock_response: str) -> RequestProcessor:
    """Create a mock RequestProcessor with predefined response."""
    mock_provider = create_mock_provider()

    def mock_process_with_metadata(*args, **kwargs):
        return ProcessingResult(
            content=mock_response,
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

    processor = RequestProcessor(mock_provider)
    processor.process_with_metadata = mock_process_with_metadata
    return processor


def demo_extract_headings():
    """Demo 1: Extract headings from markdown."""
    print("=" * 60)
    print("Demo 1: Extract Headings")
    print("=" * 60)

    markdown_content = """# Title
Some content here.

# 1 title1
Content for section 1.

# 1.1 title1.1
Subsection content.

# 1.1.1 title1.1.1
Sub-subsection content.

## Another Section
More content.
"""

    headings = HeadingExtractor.extract(markdown_content)

    print(f"\nExtracted {len(headings)} headings:\n")
    for i, heading in enumerate(headings, 1):
        print(f"{i}. Level {heading.level}: {heading.title}")
        print(f"   Position: {heading.start_pos}-{heading.end_pos}")
        print(f"   Raw: {heading.raw_text[:50]}...")
        print()

    return headings, markdown_content


def demo_format_headings_mock(headings, use_real_api: bool = False):
    """Demo 2: Format headings using LLM (mock or real)."""
    print("=" * 60)
    print("Demo 2: Format Headings with LLM")
    print("=" * 60)

    # Mock response
    mock_formatted = """# Title
## 1 title1
### 1.1 title1.1
#### 1.1.1 title1.1.1
## Another Section"""

    if use_real_api:
        print("\nUsing real API (requires configuration)...")
        # This would require actual config and API setup
        # For demo purposes, we'll use mock
        print("Note: Real API demo requires valid configuration.")
        print("Using mock response for demonstration.\n")
        use_real_api = False

    if not use_real_api:
        print("\nUsing mock LLM response:\n")
        processor = create_mock_processor(mock_formatted)
    else:
        # Real API would be initialized here
        processor = None  # Placeholder

    formatter = HeadingFormatter(processor=processor)

    print("Original headings:")
    for h in headings:
        print(f"  {h.raw_text}")

    try:
        formatted_headings = formatter.format_headings(headings)
        print("\nFormatted headings:")
        for fh in formatted_headings:
            print(f"  {fh}")
        return formatted_headings
    except Exception as e:
        print(f"\nError: {e}")
        return None


def demo_apply_formatted_headings(original_content: str, headings, formatted_headings):
    """Demo 3: Apply formatted headings back to content."""
    print("\n" + "=" * 60)
    print("Demo 3: Apply Formatted Headings")
    print("=" * 60)

    applier = HeadingApplier()
    formatted_content = applier.apply(original_content, headings, formatted_headings)

    print("\nOriginal content (first 300 chars):")
    print("-" * 60)
    print(original_content[:300])
    print("...")

    print("\nFormatted content (first 300 chars):")
    print("-" * 60)
    print(formatted_content[:300])
    print("...")

    return formatted_content


def demo_full_workflow():
    """Demo 4: Full workflow with file I/O."""
    print("\n" + "=" * 60)
    print("Demo 4: Full Workflow")
    print("=" * 60)

    # Create test markdown file
    test_content = """# Main Title

Introduction paragraph.

# 1 First Section
Content for first section.

# 1.1 Subsection One
Subsection content.

# 1.2 Subsection Two
More subsection content.

# 2 Second Section
Content for second section.

# 2.1 Subsection of Second
Nested content.
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        print(f"\nCreated test file: {temp_file}")
        print("\nOriginal content:")
        print("-" * 60)
        print(test_content)

        # Extract headings
        headings = HeadingExtractor.extract(test_content)
        print(f"\nExtracted {len(headings)} headings")

        # Format headings (mock)
        mock_formatted = """# Main Title
## 1 First Section
### 1.1 Subsection One
### 1.2 Subsection Two
## 2 Second Section
### 2.1 Subsection of Second"""

        processor = create_mock_processor(mock_formatted)
        formatter = HeadingFormatter(processor=processor)
        formatted_headings = formatter.format_headings(headings)

        # Apply formatting
        applier = HeadingApplier()
        formatted_content = applier.apply(test_content, headings, formatted_headings)

        # Write output
        output_file = temp_file.replace(".md", "_formatted.md")
        Path(output_file).write_text(formatted_content, encoding="utf-8")

        print("\nFormatted content:")
        print("-" * 60)
        print(formatted_content)

        print(f"\n✓ Output saved to: {output_file}")
        print("\nComparison:")
        print("-" * 60)
        for orig, fmt in zip(headings, formatted_headings):
            print(f"  {orig.raw_text.strip():30} → {fmt.strip()}")

    finally:
        # Cleanup
        Path(temp_file).unlink()
        if Path(output_file).exists():
            Path(output_file).unlink()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Format Command Demo")
    print("=" * 60)
    print("\nThis demo shows how the format command works:")
    print("1. Extract headings from markdown")
    print("2. Format headings using LLM")
    print("3. Apply formatted headings back")
    print("4. Full workflow example")
    print("\nNote: Using mock LLM responses for demonstration.")
    print("=" * 60)

    try:
        # Demo 1: Extract headings
        headings, original_content = demo_extract_headings()

        # Demo 2: Format headings
        formatted_headings = demo_format_headings_mock(headings, use_real_api=False)

        if formatted_headings:
            # Demo 3: Apply formatting
            demo_apply_formatted_headings(
                original_content, headings, formatted_headings
            )

        # Demo 4: Full workflow
        demo_full_workflow()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
