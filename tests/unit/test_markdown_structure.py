"""Unit tests for the single MarkdownStructure parser (P3.1)."""

from ask_llm.core.markdown_structure import MarkdownStructure


class TestFenceRanges:
    def test_paired_fences(self):
        text = "intro\n```python\nprint('hi')\n```\noutro\n"
        s = MarkdownStructure.parse(text)
        assert len(s.fence_ranges) == 1
        start, end = s.fence_ranges[0]
        assert text[start:end].startswith("```python")
        assert "print" in text[start:end]

    def test_unclosed_fence_extends_to_eof(self):
        text = "intro\n```\ncode\n# not a heading\n"
        s = MarkdownStructure.parse(text)
        assert s.fence_ranges == [(6, len(text))]

    def test_tilde_fences(self):
        text = "~~~js\ncode\n~~~\n"
        s = MarkdownStructure.parse(text)
        assert len(s.fence_ranges) == 1


class TestFrontmatter:
    def test_frontmatter_detected(self):
        text = "---\ntitle: foo\ntags: [a, b]\n---\n# Real Heading\nbody\n"
        s = MarkdownStructure.parse(text)
        assert s.frontmatter_range is not None
        start, end = s.frontmatter_range
        assert start == 0
        assert text[start:end].startswith("---")
        assert "tags" in text[start:end]

    def test_no_frontmatter_when_not_at_start(self):
        text = "# Heading\n\n---\n\nbody\n"
        s = MarkdownStructure.parse(text)
        assert s.frontmatter_range is None

    def test_frontmatter_headings_excluded(self):
        """A '# foo' inside frontmatter must not be treated as a heading."""
        text = "---\ndescription: |\n  # not a heading\n---\n# Real\n"
        s = MarkdownStructure.parse(text)
        titles = [h.title for h in s.headings]
        assert titles == ["Real"]

    def test_unclosed_frontmatter_not_detected(self):
        text = "---\ntitle: foo\n# Heading\n"
        s = MarkdownStructure.parse(text)
        assert s.frontmatter_range is None


class TestHeadings:
    def test_levels_and_positions(self):
        text = "# One\n\ntext\n\n## Two\n### Three\n"
        s = MarkdownStructure.parse(text)
        assert [(h.level, h.title) for h in s.headings] == [
            (1, "One"),
            (2, "Two"),
            (3, "Three"),
        ]
        first = s.headings[0]
        assert text[first.start_pos : first.end_pos] == "# One"

    def test_headings_inside_fences_excluded(self):
        text = "```\n# fake\n```\n# real\n"
        s = MarkdownStructure.parse(text)
        assert [h.title for h in s.headings] == ["real"]

    def test_headings_at_level(self):
        text = "# A\n## B\n## C\n# D\n"
        s = MarkdownStructure.parse(text)
        assert [h.title for h in s.headings_at_level(2)] == ["B", "C"]

    def test_is_protected(self):
        text = "---\nx: 1\n---\n# H\n```\ncode\n```\ntail\n"
        s = MarkdownStructure.parse(text)
        fm_start, _fm_end = s.frontmatter_range
        assert s.is_protected(fm_start + 2)
        fence_start, _ = s.fence_ranges[0]
        assert s.is_protected(fence_start + 1)
        h = s.headings[0]
        assert not s.is_protected(h.start_pos)


class TestConsumerEquivalence:
    """Legacy classmethods delegate to MarkdownStructure identically."""

    def test_heading_extractor_matches_parser(self):
        from ask_llm.core.md_heading_formatter import HeadingExtractor

        text = "---\nt: x\n---\n# A\n```\n# fake\n```\n## B\n"
        headings = HeadingExtractor.extract(text)
        assert [(h.level, h.title) for h in headings] == [(1, "A"), (2, "B")]

    def test_token_splitter_fence_ranges_match_parser(self):
        from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter

        text = "a\n```\ncode\n```\nb\n"
        assert MarkdownTokenSplitter._find_code_fence_ranges(text) == (
            MarkdownStructure.parse(text).fence_ranges
        )
