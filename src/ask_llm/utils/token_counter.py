"""Token counting utilities."""

from functools import lru_cache
from typing import Any, ClassVar

from loguru import logger

from ask_llm.config.context import get_config_or_none
from ask_llm.core.constants import APPROX_TOKEN_SAFETY_FACTOR, TOKEN_COUNT_CACHE_SIZE

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using word count approximation")


# Fallback encoding when no config is loaded (e.g. library / embedded use
# without set_config). cl100k_base is the project's common default across
# ENCODING_MAP. See ARCHITECTURE_REVIEW.md 4.2.3.
_DEFAULT_ENCODING_FALLBACK = "cl100k_base"


def _default_encoding() -> str:
    """Configured default encoding, falling back when no config is loaded."""
    lr = get_config_or_none()
    if lr is not None:
        return lr.unified_config.token.default_encoding
    return _DEFAULT_ENCODING_FALLBACK


class TokenCounter:
    """Count tokens and words in text."""

    # 缓存 encoding 对象，避免每次调用 tiktoken.get_encoding()
    _encoding_cache: ClassVar[dict[str, Any]] = {}

    # Providers whose real BPE tokenizer differs from the cl100k_base fallback.
    # cl100k_base materially undercounts CJK text, so counts for these models are
    # approximate; chunk sizing applies APPROX_TOKEN_SAFETY_FACTOR to compensate.
    _APPROXIMATE_PREFIXES: ClassVar[tuple[str, ...]] = ("deepseek", "qwen")
    _warned_approximate: ClassVar[set[str]] = set()

    # Model to encoding mapping (kept in code as models evolve frequently)

    # Model to encoding mapping
    ENCODING_MAP: ClassVar[dict[str, str]] = {
        # GPT-4 models
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        # GPT-3.5 models
        "gpt-3.5": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        # DeepSeek models
        "deepseek": "cl100k_base",
        "deepseek-chat": "cl100k_base",
        "deepseek-reasoner": "cl100k_base",
        # Qwen models
        "qwen": "cl100k_base",
        "qwen-turbo": "cl100k_base",
        "qwen-plus": "cl100k_base",
        "qwen-max": "cl100k_base",
    }

    @classmethod
    def count_words(cls, text: str) -> int:
        """
        Count words in text.

        Args:
            text: Input text

        Returns:
            Number of words
        """
        if not text:
            return 0
        # Split by whitespace and filter empty strings
        words = text.split()
        return len(words)

    @classmethod
    def is_approximate_model(cls, model: str | None) -> bool:
        """True if token counts for ``model`` rely on a non-native BPE approximation.

        DeepSeek and Qwen ship their own tokenizers; we fall back to cl100k_base,
        which undercounts CJK text. Callers that size against a provider context
        window should apply :data:`APPROX_TOKEN_SAFETY_FACTOR`.
        """
        if not model:
            return False
        m = model.lower()
        return any(m.startswith(p) or p in m for p in cls._APPROXIMATE_PREFIXES)

    @classmethod
    def _warn_approximate_once(cls, model: str | None) -> None:
        """Emit a single WARNING per model when its tokenizer is approximated."""
        if not cls.is_approximate_model(model):
            return
        key = (model or "").lower()
        if key in cls._warned_approximate:
            return
        cls._warned_approximate.add(key)
        logger.warning(
            f"Token counts for '{model}' are approximate (using cl100k_base; "
            f"DeepSeek/Qwen use their own BPE, which undercounts CJK). Chunk "
            f"sizing applies a {int(APPROX_TOKEN_SAFETY_FACTOR * 100)}% safety "
            f"margin to avoid context-window overflow."
        )

    @classmethod
    def count_characters(cls, text: str) -> int:
        """
        Count characters in text.

        Args:
            text: Input text

        Returns:
            Number of characters
        """
        return len(text)

    @classmethod
    def get_encoding(cls, model: str | None = None) -> Any | None:
        """
        Get tiktoken encoding object for a model.

        Args:
            model: Model name for encoding selection

        Returns:
            tiktoken.Encoding instance, or None if tiktoken unavailable / fails
        """
        if not TIKTOKEN_AVAILABLE:
            return None
        try:
            encoding_name = cls._get_encoding(model)
            cache_key = encoding_name
            enc = cls._encoding_cache.get(cache_key)
            if enc is None:
                enc = tiktoken.get_encoding(encoding_name)
                cls._encoding_cache[cache_key] = enc
            return enc
        except Exception as e:
            logger.debug(f"Token encoding retrieval failed: {e}")
            return None

    @classmethod
    def count_tokens(cls, text: str, model: str | None = None) -> int:
        """
        Count tokens in text using tiktoken.

        Results are cached (LRU) keyed by (text, model) to avoid re-encoding the
        same substrings during repeated binary-search splitting.

        Args:
            text: Input text
            model: Model name for encoding selection

        Returns:
            Number of tokens, or word count if tiktoken unavailable
        """
        if not text:
            return 0
        cls._warn_approximate_once(model)
        return cls._count_tokens_cached(text, model)

    @staticmethod
    @lru_cache(maxsize=TOKEN_COUNT_CACHE_SIZE)
    def _count_tokens_cached(text: str, model: str | None) -> int:
        """Cache-backed token count implementation. See :meth:`count_tokens`."""
        if not TIKTOKEN_AVAILABLE:
            return TokenCounter.count_words(text)

        try:
            encoding = TokenCounter.get_encoding(model)
            if encoding is None:
                return TokenCounter.count_words(text)
            return len(encoding.encode(text))
        except Exception as e:
            logger.debug(f"Token counting failed: {e}, falling back to word count")
            return TokenCounter.count_words(text)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the token-count LRU cache. Useful in tests or long-running processes."""
        cls._count_tokens_cached.cache_clear()

    @classmethod
    def estimate_tokens(cls, text: str, model: str | None = None) -> dict:
        """
        Estimate various text metrics.

        Args:
            text: Input text
            model: Model name

        Returns:
            Dictionary with word_count, token_count, char_count
        """
        return {
            "word_count": cls.count_words(text),
            "token_count": cls.count_tokens(text, model),
            "char_count": cls.count_characters(text),
        }

    @classmethod
    def _get_encoding(cls, model: str | None) -> str:
        """
        Get encoding name for a model.

        Args:
            model: Model name

        Returns:
            Encoding name
        """
        if not model:
            return _default_encoding()

        model_lower = model.lower()

        # Check for exact match first
        if model_lower in cls.ENCODING_MAP:
            return cls.ENCODING_MAP[model_lower]

        # Check for partial match
        for key, encoding in cls.ENCODING_MAP.items():
            if key in model_lower:
                return encoding

        return _default_encoding()

    @classmethod
    def split_hard_by_max_tokens(
        cls, text: str, max_tokens: int, model: str | None = None
    ) -> list[str]:
        """
        Greedy split: each returned segment has at most max_tokens (tiktoken), snapping at newlines when possible.

        For providers whose tokenizer is approximated (DeepSeek/Qwen), the budget
        is reduced by :data:`APPROX_TOKEN_SAFETY_FACTOR` because cl100k_base
        undercounts CJK and a "fitting" chunk could overflow the real context
        window. See ARCHITECTURE_REVIEW.md bug B2.
        """
        text = text.strip()
        if not text:
            return []
        budget = max_tokens
        if cls.is_approximate_model(model):
            budget = max(1, int(max_tokens * APPROX_TOKEN_SAFETY_FACTOR))
        if cls.count_tokens(text, model) <= budget:
            return [text]

        out: list[str] = []
        remaining = text
        while remaining:
            if cls.count_tokens(remaining, model) <= budget:
                out.append(remaining)
                break

            lo, hi = 1, len(remaining)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if cls.count_tokens(remaining[:mid], model) <= budget:
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

    @classmethod
    def truncate_to_tokens(cls, text: str, max_tokens: int, model: str | None = None) -> str:
        """
        Truncate text to maximum token count.

        When tiktoken is unavailable or the encoding cannot be resolved, falls back
        to word-based truncation — consistent with :meth:`count_tokens`' word-count
        fallback — rather than the previous ``max_tokens * 4`` char heuristic that
        used a different approximation than :meth:`count_tokens`.

        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            model: Model name

        Returns:
            Truncated text
        """
        if not text:
            return text
        if cls.count_tokens(text, model) <= max_tokens:
            return text

        encoding = cls.get_encoding(model) if TIKTOKEN_AVAILABLE else None
        if encoding is not None:
            try:
                tokens = encoding.encode(text)
                return str(encoding.decode(tokens[:max_tokens]))
            except Exception as e:
                logger.warning(f"Token truncation failed: {e}")

        # Fallback: word-based, consistent with count_tokens' word-count fallback.
        return " ".join(text.split()[:max_tokens])

    @classmethod
    def format_stats(cls, text: str, model: str | None = None) -> str:
        """
        Format text statistics as a string.

        Args:
            text: Input text
            model: Model name

        Returns:
            Formatted statistics string
        """
        stats = cls.estimate_tokens(text, model)
        return (
            f"Words: {stats['word_count']}, "
            f"Tokens: {stats['token_count']}, "
            f"Chars: {stats['char_count']}"
        )
