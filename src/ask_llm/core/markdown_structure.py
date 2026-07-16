"""Single-pass Markdown structure parser (P3.1).

One parse produces every structural fact the splitting/formatting pipelines
need — code-fence ranges, YAML frontmatter range, and heading spans with
levels — so protection rules live in exactly one place:

- Headings inside fenced code blocks are never real headings.
- Headings inside YAML frontmatter are never real headings.
- An unclosed fence extends to end-of-text (fail-safe: protect, don't split).

Consumers: ``HeadingExtractor``, ``MarkdownTokenSplitter`` (and later the
``chunk_balance`` rebalance path via the BinarySplitter stage).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Regex to match markdown headings: # Title, ## Title, etc.
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Matches a markdown code fence: ``` or ~~~ (with optional language / trailing text).
CODE_FENCE_PATTERN = re.compile(r"^(```|~~~).*$", re.MULTILINE)

# YAML frontmatter: a ``---`` line at document start, closed by the next
# ``---`` (or ``...``) line. Only recognized at offset 0.
_FRONTMATTER_PATTERN = re.compile(r"\A---[ \t]*\r?\n[\s\S]*?(?:\r?\n)(?:---|\.\.\.)[ \t]*(?:\r?\n|$)")


@dataclass(frozen=True)
class HeadingSpan:
    """A real (non-code, non-frontmatter) markdown heading."""

    level: int  # 1-6
    title: str  # heading text without the leading #s
    start_pos: int  # char offset of the heading line start
    end_pos: int  # char offset of the heading line end (regex match end)


@dataclass
class MarkdownStructure:
    """Structural index of a markdown document."""

    text: str
    fence_ranges: list[tuple[int, int]] = field(default_factory=list)
    frontmatter_range: tuple[int, int] | None = None
    headings: list[HeadingSpan] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> MarkdownStructure:
        """Parse *text* once, extracting fences, frontmatter, and headings."""
        frontmatter = cls._find_frontmatter_range(text)
        fence_ranges = cls._find_code_fence_ranges(text)

        protected = list(fence_ranges)
        if frontmatter is not None:
            protected.append(frontmatter)

        headings: list[HeadingSpan] = []
        for match in HEADING_PATTERN.finditer(text):
            pos = match.start()
            if cls._pos_in_ranges(protected, pos):
                continue
            headings.append(
                HeadingSpan(
                    level=len(match.group(1)),
                    title=match.group(2),
                    start_pos=pos,
                    end_pos=match.end(),
                )
            )

        return cls(
            text=text,
            fence_ranges=fence_ranges,
            frontmatter_range=frontmatter,
            headings=headings,
        )

    @staticmethod
    def _find_frontmatter_range(text: str) -> tuple[int, int] | None:
        """Return the ``(start, end)`` range of YAML frontmatter, or None.

        Only a ``---`` block at offset 0 counts; a thematic break mid-document
        is not frontmatter.
        """
        match = _FRONTMATTER_PATTERN.match(text)
        if not match:
            return None
        return (match.start(), match.end())

    @staticmethod
    def _find_code_fence_ranges(text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` char ranges of fenced code blocks.

        An unclosed fence extends to end-of-text. See ARCHITECTURE_REVIEW.md
        bug B4: without this, a ``#`` inside a code fence is treated as a
        heading and used as a split point, and a long fenced block is cut
        mid-fence.
        """
        ranges: list[tuple[int, int]] = []
        in_code = False
        block_start = 0
        for match in CODE_FENCE_PATTERN.finditer(text):
            if not in_code:
                block_start = match.start()
                in_code = True
            else:
                ranges.append((block_start, match.end()))
                in_code = False
        if in_code:
            ranges.append((block_start, len(text)))
        return ranges

    @staticmethod
    def _pos_in_ranges(ranges: list[tuple[int, int]], pos: int) -> bool:
        return any(start <= pos < end for start, end in ranges)

    def is_protected(self, pos: int) -> bool:
        """True if *pos* lies inside a fence or the frontmatter."""
        if self._pos_in_ranges(self.fence_ranges, pos):
            return True
        if self.frontmatter_range is not None:
            start, end = self.frontmatter_range
            return start <= pos < end
        return False

    def headings_at_level(self, level: int) -> list[HeadingSpan]:
        """Return headings of exactly *level* in document order."""
        return [h for h in self.headings if h.level == level]
