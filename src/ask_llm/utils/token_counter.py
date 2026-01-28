"""Token counting utilities."""

from typing import ClassVar, Optional

from loguru import logger

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using word count approximation")


class TokenCounter:
    """Count tokens and words in text."""

    # Default encoding for unknown models
    DEFAULT_ENCODING: ClassVar[str] = "cl100k_base"

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
    def count_tokens(cls, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Input text
            model: Model name for encoding selection

        Returns:
            Number of tokens, or word count if tiktoken unavailable
        """
        if not text:
            return 0

        if not TIKTOKEN_AVAILABLE:
            return cls.count_words(text)

        try:
            encoding_name = cls._get_encoding(model)
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.debug(f"Token counting failed: {e}, falling back to word count")
            return cls.count_words(text)

    @classmethod
    def estimate_tokens(cls, text: str, model: Optional[str] = None) -> dict:
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
    def _get_encoding(cls, model: Optional[str]) -> str:
        """
        Get encoding name for a model.

        Args:
            model: Model name

        Returns:
            Encoding name
        """
        if not model:
            return cls.DEFAULT_ENCODING

        model_lower = model.lower()

        # Check for exact match first
        if model_lower in cls.ENCODING_MAP:
            return cls.ENCODING_MAP[model_lower]

        # Check for partial match
        for key, encoding in cls.ENCODING_MAP.items():
            if key in model_lower:
                return encoding

        return cls.DEFAULT_ENCODING

    @classmethod
    def truncate_to_tokens(cls, text: str, max_tokens: int, model: Optional[str] = None) -> str:
        """
        Truncate text to maximum token count.

        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            model: Model name

        Returns:
            Truncated text
        """
        if not TIKTOKEN_AVAILABLE:
            # Rough approximation: ~4 characters per token
            max_chars = max_tokens * 4
            return text[:max_chars]

        try:
            encoding_name = cls._get_encoding(model)
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(text)

            if len(tokens) <= max_tokens:
                return text

            truncated = encoding.decode(tokens[:max_tokens])
            return truncated
        except Exception as e:
            logger.warning(f"Token truncation failed: {e}")
            return text

    @classmethod
    def format_stats(cls, text: str, model: Optional[str] = None) -> str:
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
