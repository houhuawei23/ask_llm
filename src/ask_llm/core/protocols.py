"""Protocol definitions for type checking."""

from collections.abc import Generator
from typing import Any, NamedTuple, Protocol

from ask_llm.core.models import ProviderConfig


class TokenUsage(NamedTuple):
    """Provider-reported token accounting (absent on streaming paths)."""

    input_tokens: int
    output_tokens: int
    cache_hit_tokens: int | None = None


class ReasoningChunk(NamedTuple):
    """A streaming chunk that carries both content and reasoning tokens.

    ``usage`` carries the provider-reported accounting for non-streaming
    responses (``None`` for stream deltas, where usage is not reported);
    consumers must treat it as optional and fall back to local estimates.
    """

    content: str
    reasoning: str
    usage: TokenUsage | None = None


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

    def test_connection(self) -> tuple[bool, str, float]:
        """Probe the provider API: (success, message, latency_seconds)."""
        ...
