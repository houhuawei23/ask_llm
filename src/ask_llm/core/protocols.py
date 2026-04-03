"""Protocol definitions for type checking."""

from typing import Dict, Generator, List, NamedTuple, Optional, Protocol, Union


class ReasoningChunk(NamedTuple):
    """A streaming chunk that carries both content and reasoning tokens."""

    content: str
    reasoning: str


class LLMProviderProtocol(Protocol):
    """Protocol for LLM providers compatible with ask_llm."""

    config: object
    name: str
    default_model: str
    available_models: List[str]

    def call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Generator[Union[str, "ReasoningChunk"], None, None]]:
        """Call the LLM API."""
        ...
