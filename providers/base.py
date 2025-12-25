"""Base provider class for LLM API providers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseProvider(ABC):
    """Abstract base class for LLM API providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Configuration dictionary containing api_key, api_base, etc.
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate the provider configuration."""
        pass
    
    @abstractmethod
    def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: The input prompt text
            temperature: Temperature parameter (overrides config if provided)
            model: Model name (overrides config if provided)
            **kwargs: Additional parameters
            
        Returns:
            The response text from the API
        """
        pass
    
    def _get_temperature(self, temperature: Optional[float] = None) -> float:
        """Get temperature value, using override if provided."""
        return temperature if temperature is not None else self.config.get('api_temperature', 0.7)
    
    def _get_model(self, model: Optional[str] = None) -> str:
        """Get model name, using override if provided."""
        return model if model is not None else self.config.get('api_model', '')

