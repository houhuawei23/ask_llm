"""Token counting utilities."""

import re
import threading
from collections import OrderedDict
from typing import Any, ClassVar

from loguru import logger

from ask_llm.config.context import get_config_or_none
from ask_llm.core.constants import APPROX_TOKEN_SAFETY_FACTOR

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
    _APPROXIMATE_PREFIXES: ClassVar[tuple[str, ...]] = (
        "deepseek",
        "qwen",
        "kimi",
        "glm",
        "minimax",
    )
    _warned_approximate: ClassVar[set[str]] = set()
    _warned_word_fallback: ClassVar[bool] = False

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

        DeepSeek, Qwen, Kimi, GLM and MiniMax ship their own tokenizers; we fall
        back to cl100k_base, which undercounts CJK text. Callers that size
        against a provider context window should apply
        :data:`APPROX_TOKEN_SAFETY_FACTOR`.
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

    @classmethod
    def _warn_word_fallback_once(cls) -> None:
        """One-shot WARNING: whitespace word counts are a bad token proxy for CJK."""
        if not cls._warned_word_fallback:
            cls._warned_word_fallback = True
            logger.warning(
                "tiktoken is unavailable or failing; token estimation fell back to "
                "whitespace word counts, which badly undercount CJK text and can "
                "oversize chunks until the provider rejects them. Install tiktoken "
                "for accurate counts."
            )

    # Audit 4.4: word-count fallback floor. Whitespace word counts collapse
    # CJK text (no spaces → whole paragraphs = 1 "word"), oversizing chunks
    # until the provider rejects them. cl100k encodes a han character at
    # ~1-2 tokens and ~4 latin characters at ~1 token, so a char-class mix is
    # a far safer floor; for han-only text it lands at roughly
    # 1/APPROX_TOKEN_SAFETY_FACTOR of the budget math's assumption — i.e.
    # deliberately conservative.
    _FALLBACK_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿가-힯]")
    _FALLBACK_CJK_TOKENS_PER_CHAR = 1.0
    _FALLBACK_OTHER_CHARS_PER_TOKEN = 4.0

    @classmethod
    def _word_fallback_estimate(cls, text: str) -> int:
        """Word count lifted by a CJK-aware character floor (tiktoken-free path)."""
        try:
            words = cls.count_words(text)
        except Exception:
            return 1
        try:
            cjk = len(cls._FALLBACK_CJK_RE.findall(text))
            other = len(text) - cjk
            char_estimate = int(
                cjk * cls._FALLBACK_CJK_TOKENS_PER_CHAR
                + other / cls._FALLBACK_OTHER_CHARS_PER_TOKEN
            )
        except TypeError:
            # Non-str input (e.g. a test double): the word count is all we have.
            return max(words, 1)
        return max(words, char_estimate, 1)

    # Audit 5.1: the token-count cache holds *whole documents* as keys; a
    # fixed entry count (the old lru_cache(1024)) could pin hundreds of MB.
    # Bound total cached text bytes instead, evicting least-recently-used.
    _TOKEN_CACHE_MAX_BYTES: ClassVar[int] = 64 * 1024 * 1024
    _token_cache: ClassVar["OrderedDict[tuple[str, str | None], tuple[int, int]]"] = OrderedDict()
    _token_cache_bytes: ClassVar[int] = 0
    _token_cache_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _count_tokens_cached(cls, text: str, model: str | None) -> int:
        """Byte-bounded LRU memo for :meth:`count_tokens` (audit 5.1)."""
        key = (text, model)
        lock = cls._token_cache_lock
        with lock:
            entry = cls._token_cache.get(key)
            if entry is not None:
                value, _entry_bytes = entry
                cls._token_cache.move_to_end(key)
                return value

        value = cls._compute_token_count(text, model)

        # char count as a cheap byte proxy (UTF-8 is 1-4 bytes/char, so this
        # over-admits slightly; the bound is a guardrail, not an exact quota)
        entry_bytes = max(1, len(text))
        with lock:
            if key not in cls._token_cache:
                cls._token_cache[key] = (value, entry_bytes)
                cls._token_cache_bytes += entry_bytes
                while cls._token_cache_bytes > cls._TOKEN_CACHE_MAX_BYTES and cls._token_cache:
                    _, (_, evicted_bytes) = cls._token_cache.popitem(last=False)
                    cls._token_cache_bytes -= evicted_bytes
            return value

    @classmethod
    def _compute_token_count(cls, text: str, model: str | None) -> int:
        """Cache-miss token count. See :meth:`count_tokens`."""
        if not TIKTOKEN_AVAILABLE:
            cls._warn_word_fallback_once()
            return cls._word_fallback_estimate(text)

        try:
            encoding = cls.get_encoding(model)
            if encoding is None:
                cls._warn_word_fallback_once()
                return cls._word_fallback_estimate(text)
            return len(encoding.encode(text))
        except Exception as e:
            logger.debug(f"Token counting failed: {e}, falling back to word count")
            cls._warn_word_fallback_once()
            return cls._word_fallback_estimate(text)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the token-count LRU cache. Useful in tests or long-running processes."""
        with cls._token_cache_lock:
            cls._token_cache.clear()
            cls._token_cache_bytes = 0

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

        # Partial match, longest key first: insertion order made
        # "gpt-4o-2024-08-06" hit "gpt-4" (cl100k_base) instead of "gpt-4o"
        # (o200k_base) — a 10-20% CJK miscount fed straight into chunk budgets.
        for key, encoding in sorted(
            cls.ENCODING_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
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

            # Audit 5.1 note: this bisection is deliberately kept full-range
            # and exact. Prefix token counts are NON-monotone (BPE merges at a
            # cut boundary), so a windowed search seeded from token offsets
            # converges to different cuts and breaks the byte-identical
            # regression contract; the 5.1 optimization here is the byte-
            # bounded token cache instead.
            best = cls._bisect_best_cut(remaining, 1, len(remaining), budget, model) or 1

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
    def _bisect_best_cut(
        cls, remaining: str, lo: int, hi: int, budget: int, model: str | None
    ) -> int | None:
        """Largest cut in [lo, hi] whose prefix encodes to ≤ budget tokens.

        Mirrors the historical inline bisection exactly (same mid sequence), so
        results are byte-identical to previous releases.
        """
        best: int | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if cls.count_tokens(remaining[:mid], model) <= budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best
