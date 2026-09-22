"""Unit tests for FileHandler chunked I/O callbacks (P4.10)."""

from pathlib import Path

import pytest

from ask_llm.utils.file_handler import FileHandler


class TestChunkedIO:
    def test_read_chunked_reports_bytes(self, tmp_path):
        f = tmp_path / "in.txt"
        text = "你好世界\n" * 4000  # multibyte; forces several chunks
        f.write_text(text, encoding="utf-8")

        seen: list[int] = []
        content = FileHandler.read_chunked(f, on_chunk=seen.append)
        assert content == text
        assert seen, "on_chunk should fire"
        assert sum(seen) == len(text.encode("utf-8"))

    def test_read_chunked_without_callback(self, tmp_path):
        f = tmp_path / "in.txt"
        f.write_text("abc", encoding="utf-8")
        assert FileHandler.read_chunked(f) == "abc"

    def test_write_chunked_reports_bytes(self, tmp_path):
        f = tmp_path / "out.txt"
        text = "你好世界\n" * 4000
        seen: list[int] = []
        FileHandler.write_chunked(f, text, on_chunk=seen.append)
        assert f.read_text(encoding="utf-8") == text
        assert seen, "on_chunk should fire"
        assert sum(seen) == len(text.encode("utf-8"))
        # no chunk exceeds the configured chunk size in bytes... characters,
        # actually: slices are character-based, byte counts vary for CJK.
        assert all(n > 0 for n in seen)

    def test_write_chunked_without_callback(self, tmp_path):
        f = tmp_path / "out.txt"
        FileHandler.write_chunked(f, "xyz")
        assert f.read_text(encoding="utf-8") == "xyz"


class TestAtomicWrite:
    """H10: writes must be atomic (tmp + os.replace) so a crash mid-write
    never truncates the target — for --inplace the target is the user's
    source document."""

    def test_write_atomic_failure_preserves_original(self, tmp_path, monkeypatch):
        target = tmp_path / "out.md"
        target.write_text("original content", encoding="utf-8")

        def boom(path, payload):
            raise OSError("disk gone")

        monkeypatch.setattr("ask_llm.utils.file_handler.atomic_write_text", boom)
        with pytest.raises(OSError):
            FileHandler.write(target, "new content", force=True)
        assert target.read_text(encoding="utf-8") == "original content"
        assert not (tmp_path / "out.md.tmp").exists()

    def test_write_chunked_atomic_failure_preserves_original(self, tmp_path, monkeypatch):
        target = tmp_path / "out.md"
        target.write_text("original content", encoding="utf-8")

        real_replace = Path.replace

        def exploding_replace(self, target_path):
            if self.suffix == ".tmp":
                raise OSError("replace failed")
            return real_replace(self, target_path)

        monkeypatch.setattr(Path, "replace", exploding_replace)
        with pytest.raises(OSError):
            FileHandler.write_chunked(target, "new content")
        assert target.read_text(encoding="utf-8") == "original content"
        assert not (tmp_path / "out.md.tmp").exists()

    def test_write_chunked_roundtrip_and_no_tmp_leftover(self, tmp_path):
        target = tmp_path / "out.md"
        content = "x" * 10_000
        FileHandler.write_chunked(target, content)
        assert target.read_text(encoding="utf-8") == content
        assert not (tmp_path / "out.md.tmp").exists()

    def test_write_roundtrip_content(self, tmp_path):
        target = tmp_path / "nested" / "out.md"
        FileHandler.write(target, "hello", force=True)
        assert target.read_text(encoding="utf-8") == "hello"


def test_generate_output_path_expands_tilde(tmp_path, monkeypatch):
    """M12/2.25: `~/` in either the input-derived or custom output path must
    expand to the home directory, not create a literal './~' dir in cwd."""
    from ask_llm.utils.file_handler import FileHandler

    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
    out = FileHandler.generate_output_path("~/notes/input.md", "~/out/result.md")
    assert not out.startswith("~")
    assert Path(out).is_absolute() and str(Path.home()) in out

    out2 = FileHandler.generate_output_path("~/notes/input.md", None)
    assert "~" not in out2
    assert Path(out2).is_absolute()
