"""Tests for the token-aware Markdown splitter (translation body path).

Covers code-fence awareness (B4): headings inside a fenced block must not
become split points, and fenced code blocks (that fit the budget) must not be
cut mid-fence.
"""

from __future__ import annotations

from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter

MODEL = "gpt-4"


def _all_fences_balanced(chunks) -> bool:
    """Every chunk has an even number of ``` markers (no fence straddles a cut)."""
    return all(c.content.count("```") % 2 == 0 for c in chunks)


def test_heading_inside_code_fence_not_a_split_point():
    """A ``#`` line inside a fenced block is not treated as a heading."""
    text = (
        "# Real Heading\n\n"
        "Intro paragraph here with enough words to stand alone.\n\n"
        "```python\n"
        "# this is a comment, not a heading\n"
        "x = 1\n"
        "```\n\n"
        "Tail paragraph after the code block ends the document nicely.\n"
    )
    chunks = MarkdownTokenSplitter(MODEL, max_chunk_tokens=30).split(text)
    assert _all_fences_balanced(chunks)


def test_fenced_code_block_not_broken_across_chunks():
    """A fenced block that fits the budget survives paragraph splitting intact."""
    code = "\n".join(f"x{i} = {i}" for i in range(6))  # small enough to fit one chunk
    intro = "Lead paragraph that introduces the snippet below. " * 6
    outro = "Closing paragraph that follows the code block. " * 6
    text = f"{intro}\n\n```python\n{code}\n```\n\n{outro}\n"
    chunks = MarkdownTokenSplitter(MODEL, max_chunk_tokens=80).split(text)
    assert len(chunks) >= 2, "expected the document to be split"
    assert _all_fences_balanced(chunks)
    # The fenced block must appear whole in exactly one chunk.
    assert any("x0 = 0" in c.content and "x5 = 5" in c.content for c in chunks)


def test_find_code_fence_ranges_handles_unclosed_fence():
    """An unclosed fence extends to end of text."""
    ranges = MarkdownTokenSplitter._find_code_fence_ranges("```\ncode without end")
    assert ranges == [(0, len("```\ncode without end"))]


def test_find_code_fence_ranges_paired():
    ranges = MarkdownTokenSplitter._find_code_fence_ranges("a\n```\nx\n```\nb")
    assert len(ranges) == 1  # exactly one closed range
    start, _ = ranges[0]
    # Range starts at the opening fence, not before it.
    assert "a\n```"[start : start + 3] == "```"


def test_two_fenced_blocks_each_kept_whole():
    """Multiple fenced blocks each stay within a single chunk."""
    text = (
        f"{'intro line. ' * 10}\n\n"
        "```python\na = 1\nb = 2\n```\n\n"
        f"{'middle line. ' * 10}\n\n"
        "```js\nconst x = 0;\n```\n\n"
        f"{'outro line. ' * 10}\n"
    )
    chunks = MarkdownTokenSplitter(MODEL, max_chunk_tokens=60).split(text)
    assert _all_fences_balanced(chunks)
