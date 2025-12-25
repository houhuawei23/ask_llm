"""LLM API Provider modules."""

from .base import BaseProvider
from .openai_compatible import OpenAICompatibleProvider

# For backward compatibility, provide aliases
DeepSeekProvider = OpenAICompatibleProvider
QwenProvider = OpenAICompatibleProvider

__all__ = ['BaseProvider', 'OpenAICompatibleProvider', 'DeepSeekProvider', 'QwenProvider']

