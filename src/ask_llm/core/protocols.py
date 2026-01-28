"""Protocol definitions for type checking."""

from typing import Dict, Generator, List, Optional, Protocol, Union


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
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Call the LLM API."""
        ...
