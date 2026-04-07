"""Discover Markdown files from CLI arguments (paths, globs, directories)."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Sequence

from loguru import logger

# Normalized lowercase suffixes for Markdown sources
MD_SUFFIXES: tuple[str, ...] = (".md", ".markdown")


def _normalize_suffix(path: Path) -> bool:
    return path.suffix.lower() in MD_SUFFIXES


def _iter_markdown_in_directory(directory: Path, *, recursive: bool) -> Iterable[Path]:
    """Yield Markdown files under ``directory``."""
    if recursive:
        for p in sorted(directory.rglob("*")):
            if p.is_file() and _normalize_suffix(p):
                yield p
    else:
        for pattern in ("*.md", "*.markdown"):
            for p in sorted(directory.glob(pattern)):
                if p.is_file():
                    yield p


def _expand_glob_or_literal(raw: str) -> list[Path]:
    """
    Expand a single user argument to existing paths (glob, file, or directory).

    Mirrors legacy ``format`` behaviour: try glob first; if empty, treat as literal path.
    """
    matches = glob.glob(raw)
    if matches:
        return [Path(m) for m in matches]

    p = Path(raw)
    if p.exists():
        return [p]

    logger.debug("Pattern matched nothing and path does not exist: {}", raw)
    return []


def discover_markdown_files(
    patterns: Sequence[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    """
    Resolve CLI patterns to a sorted, deduplicated list of Markdown file paths.

    Supports:

    - Glob patterns (e.g. ``*.md``, ``docs/**/*.md`` when enabled by glob)
    - Explicit files
    - Directories: all ``.md`` / ``.markdown`` files inside (optionally recursive)

    Args:
        patterns: Raw arguments from the CLI (file paths, dirs, globs)
        recursive: When the argument is a directory, search subdirectories

    Returns:
        Sorted unique paths to Markdown files
    """
    seen: set[str] = set()
    out: list[Path] = []

    def add(path: Path) -> None:
        key = os.path.normcase(str(path.resolve()))
        if key not in seen:
            seen.add(key)
            out.append(path)

    for raw in patterns:
        expanded = _expand_glob_or_literal(raw)
        if not expanded:
            logger.warning("No such file or glob match: {}", raw)
            continue

        for path in expanded:
            path = path.resolve()
            if path.is_dir():
                for md in _iter_markdown_in_directory(path, recursive=recursive):
                    add(md)
            elif path.is_file():
                if _normalize_suffix(path):
                    add(path)
                else:
                    logger.warning(
                        "Skipping non-Markdown file (use .md or .markdown): {}",
                        path,
                    )
            else:
                logger.warning("Not a file or directory: {}", path)

    out.sort(key=lambda p: str(p))
    return out
