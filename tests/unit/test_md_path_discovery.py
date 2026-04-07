"""Unit tests for Markdown path discovery."""

from pathlib import Path

import pytest

from ask_llm.utils.md_path_discovery import discover_markdown_files


class TestDiscoverMarkdownFiles:
    """Tests for discover_markdown_files."""

    def test_file_only(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("# T\n", encoding="utf-8")
        out = discover_markdown_files([str(f)])
        assert [p.resolve() for p in out] == [f.resolve()]

    def test_directory_non_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "root.md").write_text("# A\n", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("# B\n", encoding="utf-8")
        out = discover_markdown_files([str(tmp_path)], recursive=False)
        names = {p.name for p in out}
        assert "root.md" in names
        assert "nested.md" not in names

    def test_directory_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "root.md").write_text("# A\n", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("# B\n", encoding="utf-8")
        out = discover_markdown_files([str(tmp_path)], recursive=True)
        names = {p.name for p in out}
        assert names == {"root.md", "nested.md"}

    def test_skip_non_md(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("x", encoding="utf-8")
        out = discover_markdown_files([str(tmp_path / "x.txt")])
        assert out == []

    def test_dedup(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("# T\n", encoding="utf-8")
        out = discover_markdown_files([str(f), str(f)])
        assert len(out) == 1

    def test_glob(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("# B\n", encoding="utf-8")
        import os

        old = os.getcwd()
        try:
            os.chdir(tmp_path)
            out = discover_markdown_files(["*.md"])
        finally:
            os.chdir(old)
        assert len(out) == 2
