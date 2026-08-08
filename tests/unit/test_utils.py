"""Unit tests for utility modules."""

import pytest

from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.utils.token_counter import TokenCounter
from ask_llm.utils.file_handler import FileHandler


class TestTokenCounter:
    """Test TokenCounter."""

    def test_count_words(self):
        """Test word counting."""
        assert TokenCounter.count_words("") == 0
        assert TokenCounter.count_words("Hello") == 1
        assert TokenCounter.count_words("Hello world") == 2
        assert TokenCounter.count_words("  Multiple   spaces  ") == 2

    def test_get_encoding_falls_back_when_no_config(self):
        """P2.6: no loaded config must not crash the hot path (embedded use)."""
        from unittest.mock import patch

        with patch("ask_llm.utils.token_counter.get_config_or_none", return_value=None):
            # Empty/unknown model falls back to the default encoding, not RuntimeError.
            assert TokenCounter._get_encoding("") == "cl100k_base"
            assert TokenCounter._get_encoding("totally-unknown-model") == "cl100k_base"


    def test_count_characters(self):
        """Test character counting."""
        assert TokenCounter.count_characters("") == 0
        assert TokenCounter.count_characters("Hello") == 5
        assert TokenCounter.count_characters("Hello world") == 11

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello world"
        stats = TokenCounter.estimate_tokens(text)

        assert "word_count" in stats
        assert "token_count" in stats
        assert "char_count" in stats
        assert stats["word_count"] == 2
        assert stats["char_count"] == 11

    def test_get_encoding(self, sample_config_file):
        """Test encoding selection."""
        load_result = ConfigLoader.load(str(sample_config_file))
        set_config(load_result)
        # When model is None, uses default from config (cl100k_base)
        assert TokenCounter._get_encoding(None) == "cl100k_base"
        assert TokenCounter._get_encoding("gpt-4") == "cl100k_base"
        assert TokenCounter._get_encoding("deepseek-chat") == "cl100k_base"

    def test_format_stats(self):
        """Test stats formatting."""
        text = "Hello world"
        formatted = TokenCounter.format_stats(text)

        assert "Words:" in formatted
        assert "Tokens:" in formatted
        assert "Chars:" in formatted

    def test_count_tokens_cached_returns_same_result(self):
        """Caching must not change the token count for repeated inputs."""
        TokenCounter.clear_cache()
        text = "The quick brown fox jumps over the lazy dog."
        first = TokenCounter.count_tokens(text, "gpt-4")
        second = TokenCounter.count_tokens(text, "gpt-4")
        assert first == second
        assert first > 0

    def test_clear_cache(self):
        """clear_cache() must empty the LRU cache without error."""
        TokenCounter.count_tokens("some text to cache", "gpt-4")
        TokenCounter.clear_cache()
        info = TokenCounter._count_tokens_cached.cache_info()
        assert info.currsize == 0

    def test_is_approximate_model(self):
        """DeepSeek/Qwen are flagged as using an approximate tokenizer."""
        assert TokenCounter.is_approximate_model("deepseek-chat") is True
        assert TokenCounter.is_approximate_model("deepseek-reasoner") is True
        assert TokenCounter.is_approximate_model("qwen-max") is True
        assert TokenCounter.is_approximate_model("Qwen-Plus") is True  # case-insensitive
        assert TokenCounter.is_approximate_model("gpt-4") is False
        assert TokenCounter.is_approximate_model(None) is False

    def test_approximate_warn_fires_once(self):
        """The approximation warning fires exactly once per model."""
        from unittest.mock import patch

        TokenCounter._warned_approximate.discard("deepseek-chat")
        with patch("ask_llm.utils.token_counter.logger") as mock_logger:
            TokenCounter.count_tokens("hello world", "deepseek-chat")
            TokenCounter.count_tokens("another sentence", "deepseek-chat")
        # Exactly one warning despite two calls
        warning_calls = list(mock_logger.warning.call_args_list)
        assert len(warning_calls) == 1
        assert "approximate" in warning_calls[0][0][0].lower()

    def test_split_applies_safety_margin_for_approximate_model(self):
        """Chunks for approximate models are smaller than the raw budget."""
        TokenCounter.clear_cache()
        # Build text large enough to require splitting under both budgets.
        text = "\n".join(f"Paragraph number {i}." for i in range(400))
        gpt_chunks = TokenCounter.split_hard_by_max_tokens(text, 100, "gpt-4")
        deepseek_chunks = TokenCounter.split_hard_by_max_tokens(text, 100, "deepseek-chat")
        # Same text, but deepseek budget is shrunk by the safety factor -> more,
        # smaller chunks.
        assert len(deepseek_chunks) >= len(gpt_chunks)
        # No chunk exceeds the (cl100k) 100-token budget.
        assert all(TokenCounter.count_tokens(c, "deepseek-chat") <= 100 for c in deepseek_chunks)

    def test_truncate_fallback_is_word_based(self):
        """Without a real encoding, truncation follows word count, not chars."""
        from unittest.mock import patch

        text = "alpha bravo charlie delta echo foxtrot"
        with patch.object(TokenCounter, "get_encoding", return_value=None):
            truncated = TokenCounter.truncate_to_tokens(text, 3)
        assert truncated.split() == ["alpha", "bravo", "charlie"]


class TestFileHandler:
    """Test FileHandler."""

    def test_read_file(self, temp_dir):
        """Test reading file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello world")

        content = FileHandler.read(test_file)
        assert content == "Hello world"

    def test_read_nonexistent_file(self, temp_dir):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            FileHandler.read(temp_dir / "nonexistent.txt")

    def test_write_progress_total_is_bytes_for_multibyte(self, temp_dir):
        """B10: write-progress total must be byte length, not char count.

        For multibyte (CJK) text bytes > chars; a char-count total made the
        progress bar overshoot 100%. The bar total now equals the UTF-8 byte
        length so it matches the byte-based increments.
        """
        from unittest.mock import patch

        captured: dict = {}

        class _FakeTqdm:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def update(self, _n):
                pass

        content = "中文" * 50  # 100 chars, but 300 UTF-8 bytes
        with (
            patch("ask_llm.utils.file_handler.tqdm", _FakeTqdm),
            patch.object(FileHandler, "_get_chunk_size", return_value=10),
            patch.object(FileHandler, "_get_tqdm_ncols", return_value=80),
        ):
            FileHandler._write_with_progress(temp_dir / "out.txt", content)

        assert captured["total"] == len(content.encode("utf-8"))
        assert captured["total"] == 300  # bytes, not 100 chars

    def test_write_file(self, temp_dir):
        """Test writing file."""
        test_file = temp_dir / "output.txt"

        FileHandler.write(test_file, "Test content")

        assert test_file.exists()
        assert test_file.read_text() == "Test content"

    def test_write_file_exists(self, temp_dir):
        """Test writing to existing file without force raises error."""
        test_file = temp_dir / "exists.txt"
        test_file.write_text("Original")

        with pytest.raises(FileExistsError):
            FileHandler.write(test_file, "New content", force=False)

    def test_write_file_force(self, temp_dir):
        """Test writing to existing file with force."""
        test_file = temp_dir / "exists.txt"
        test_file.write_text("Original")

        FileHandler.write(test_file, "New content", force=True)

        assert test_file.read_text() == "New content"

    def test_generate_output_path(self, temp_dir, sample_config_file):
        """Test output path generation."""
        load_result = ConfigLoader.load(str(sample_config_file))
        set_config(load_result)
        input_path = temp_dir / "input.txt"

        output = FileHandler.generate_output_path(input_path)
        assert output.endswith("input_output.txt")

    def test_generate_output_path_custom(self, temp_dir):
        """Test custom output path."""
        input_path = temp_dir / "input.txt"
        custom = temp_dir / "custom.md"

        output = FileHandler.generate_output_path(input_path, custom)
        assert output == str(custom)

    def test_detect_type(self):
        """Test file type detection."""
        assert FileHandler.detect_type("file.txt") == ".txt"
        assert FileHandler.detect_type("file.MD") == ".md"
        assert FileHandler.detect_type("/path/to/file.py") == ".py"

    def test_is_text_file(self):
        """Test text file detection."""
        assert FileHandler.is_text_file("file.txt") is True
        assert FileHandler.is_text_file("file.py") is True
        assert FileHandler.is_text_file("file.md") is True
        assert FileHandler.is_text_file("file.bin") is False
