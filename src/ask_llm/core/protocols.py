"""Protocol definitions for type checking."""

from collections.abc import Generator
from typing import Any, NamedTuple, Protocol

from ask_llm.core.models import ProviderConfig


class ReasoningChunk(NamedTuple):
    """A streaming chunk that carries both content and reasoning tokens."""

    content: str
    reasoning: str


class LLMProviderProtocol(Protocol):
    """Protocol for LLM providers compatible with ask_llm."""

    config: ProviderConfig
    name: str
    default_model: str
    available_models: list[str]

    def call(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        temperature: float | None = None,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | ReasoningChunk | Generator[str | ReasoningChunk, None, None]:
        """Call the LLM API."""
        ...
