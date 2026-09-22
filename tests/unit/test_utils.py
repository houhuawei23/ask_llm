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
        assert not TokenCounter._token_cache
        assert TokenCounter._token_cache_bytes == 0

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


class TestEncodingSelection:
    """Encoding-map selection must prefer the longest matching key."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4", "cl100k_base"),
            ("gpt-4o", "o200k_base"),
            ("gpt-4o-2024-08-06", "o200k_base"),
            ("gpt-4o-mini", "o200k_base"),
            ("openai/gpt-4o-mini", "o200k_base"),
            ("gpt-3.5-turbo", "cl100k_base"),
            ("deepseek-chat", "cl100k_base"),
        ],
    )
    def test_get_encoding_matches_longest_key(self, model, expected):
        assert TokenCounter._get_encoding(model) == expected

    @pytest.mark.parametrize("model", ["kimi-k2.6", "glm-5.1", "MiniMax-M2.5"])
    def test_new_cjk_providers_are_approximate(self, model):
        """Kimi/GLM/MiniMax must get the approximate-model safety factor."""
        assert TokenCounter.is_approximate_model(model)


class TestAudit44WordFallbackFloor:
    """Audit 4.4: the tiktoken-free fallback must not collapse CJK to 1 'word'."""

    def test_cjk_text_gets_char_proportional_floor(self):
        from ask_llm.utils.token_counter import TokenCounter

        text = "这是一段没有空格的中文文本"  # 13 han chars; whitespace count == 1
        estimate = TokenCounter._word_fallback_estimate(text)
        assert estimate >= len(text)

    def test_latin_text_keeps_word_count(self):
        from ask_llm.utils.token_counter import TokenCounter

        text = "hello world this is a plain sentence"
        estimate = TokenCounter._word_fallback_estimate(text)
        assert estimate >= len(text.split())
        assert estimate <= len(text)  # sanity: floor never exceeds char count

    def test_empty_text_returns_one(self):
        from ask_llm.utils.token_counter import TokenCounter

        assert TokenCounter._word_fallback_estimate("") >= 1


class TestAudit51HardSplitParity:
    """Audit 5.1: the windowed hard split must be byte-identical to the
    full-range binary search over a varied corpus."""

    @staticmethod
    def _reference_split(TokenCounter, text, max_tokens, model):  # noqa: N803
        """The pre-optimization algorithm (full-range binary search)."""
        import tiktoken

        enc = TokenCounter.get_encoding(model)
        budget = max_tokens

        def count(s):
            return len(enc.encode(s))

        text = text.strip()
        if not text:
            return []
        if count(text) <= budget:
            return [text]

        out = []
        remaining = text
        while remaining:
            if count(remaining) <= budget:
                out.append(remaining)
                break
            lo, hi = 1, len(remaining)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if count(remaining[:mid]) <= budget:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            cut = remaining.rfind("\n", 0, best)
            if cut <= 0 or cut < best // 4:
                cut = best
            piece = remaining[:cut].strip()
            if not piece:
                piece = remaining[:best].strip()
                cut = best
            out.append(piece)
            remaining = remaining[cut:].lstrip()
        return out

    def test_hard_split_matches_reference_on_corpus(self):
        import random

        from ask_llm.utils.token_counter import TokenCounter

        rng = random.Random(20260922)
        words_en = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "a",
            "lazy",
            "dog",
            "while",
            "streams",
            "of",
            "tokens",
            "flow",
            "through",
            "binary",
            "searches",
            "in",
            "the",
            "splitter",
            "engine",
        ]
        corpus = [
            "word " * 2000,
            "这是一段很长的中文文本，没有任何空格，用来验证 CJK 分词的窗口搜索与全量搜索完全一致。"
            * 40,
            "```python\ndef f():\n    return 'x'\n```\n" * 60,
            "Mixed 中文 and English words with\nnewlines\n" * 100,
            "=SUM(A1) @cmd $VAR `tick` |pipe| ;semi &and" * 80,
        ]
        for _ in range(8):
            n = rng.randint(200, 1500)
            corpus.append(" ".join(rng.choice(words_en) for _ in range(n)))
            corpus.append("".join(rng.choice("汉 字 与 单 词abc, 0123.") for _ in range(n)))

        for text in corpus:
            for budget in (32, 128, 900):
                got = TokenCounter.split_hard_by_max_tokens(text, budget, "gpt-4")
                want = self._reference_split(TokenCounter, text, budget, "gpt-4")
                assert got == want, f"parity broken (budget={budget}) on: {text[:60]!r}"

    def test_token_cache_respects_byte_budget(self):
        from ask_llm.utils.token_counter import TokenCounter

        TokenCounter.clear_cache()
        original = TokenCounter._TOKEN_CACHE_MAX_BYTES
        TokenCounter._TOKEN_CACHE_MAX_BYTES = 1000
        try:
            for i in range(50):
                TokenCounter.count_tokens(f"document number {i} " + "x" * 100, "gpt-4")
            total = sum(nbytes for _, nbytes in TokenCounter._token_cache.values())
            assert total <= 1000 + max(
                (nbytes for _, nbytes in TokenCounter._token_cache.values()), default=0
            )
            assert len(TokenCounter._token_cache) < 50
        finally:
            TokenCounter._TOKEN_CACHE_MAX_BYTES = original
            TokenCounter.clear_cache()
