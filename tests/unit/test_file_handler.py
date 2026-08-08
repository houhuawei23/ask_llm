"""Unit tests for FileHandler chunked I/O callbacks (P4.10)."""

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
