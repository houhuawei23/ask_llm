"""Token counting utilities using tiktoken."""

import re
from typing import Optional


try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of words
    """
    # Split by whitespace and filter out empty strings
    words = text.split()
    return len(words)


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        model: Model name (optional, for selecting appropriate encoding)
        
    Returns:
        Number of tokens, or word count if tiktoken is not available
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback to word count approximation
        return count_words(text)
    
    try:
        # Try to get encoding based on model
        encoding_name = _get_encoding_for_model(model)
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to cl100k_base (most common for OpenAI-compatible APIs)
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Final fallback to word count
            return count_words(text)


def _get_encoding_for_model(model: Optional[str]) -> str:
    """
    Get appropriate encoding name for a model.
    
    Args:
        model: Model name
        
    Returns:
        Encoding name
    """
    if not model:
        return "cl100k_base"
    
    model_lower = model.lower()
    
    # Map models to encodings
    # Most OpenAI-compatible models use cl100k_base
    if any(x in model_lower for x in ['gpt-4', 'gpt-3.5', 'deepseek', 'qwen']):
        return "cl100k_base"
    elif 'gpt-3' in model_lower or 'davinci' in model_lower:
        return "p50k_base"
    else:
        # Default to cl100k_base
        return "cl100k_base"

