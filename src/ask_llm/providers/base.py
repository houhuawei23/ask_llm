"""Base provider class for LLM API providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

from ask_llm.core.models import ProviderConfig


class BaseProvider(ABC):
    """Abstract base class for LLM API providers."""
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the provider configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Call the LLM API.
        
        Args:
            prompt: Single prompt text (alternative to messages)
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (overrides config)
            model: Model name (overrides config)
            stream: If True, return a generator for streaming responses
            **kwargs: Additional API parameters
            
        Returns:
            Response string, or generator if streaming
            
        Raises:
            ValueError: If neither prompt nor messages provided
            RuntimeError: If API call fails
        """
        pass
    
    def _get_temperature(self, override: Optional[float] = None) -> float:
        """
        Get temperature value with override.
        
        Args:
            override: Override value
            
        Returns:
            Temperature value
        """
        return override if override is not None else self.config.api_temperature
    
    def _get_model(self, override: Optional[str] = None) -> str:
        """
        Get model name with override.
        
        Args:
            override: Override value
            
        Returns:
            Model name
        """
        return override if override is not None else self.config.api_model
    
    @property
    def name(self) -> str:
        """Get provider name."""
        return self.config.api_provider
    
    @property
    def default_model(self) -> str:
        """Get default model name."""
        return self.config.api_model
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available models."""
        return self.config.models or [self.config.api_model]
