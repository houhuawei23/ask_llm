"""OpenAI Compatible API provider implementation."""

from typing import Optional
from openai import OpenAI
from .base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    """Provider for OpenAI-compatible APIs (DeepSeek, Qwen, etc.)."""
    
    def validate_config(self) -> None:
        """Validate OpenAI-compatible configuration."""
        required_keys = ['api_key', 'api_base', 'api_model']
        for key in required_keys:
            if key not in self.config or not self.config[key]:
                raise ValueError(f"Missing required config key: {key}")
    
    def call(
        self,
        prompt: str = None,
        messages: Optional[list] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Call OpenAI-compatible API.
        
        Args:
            prompt: The input prompt text (used if messages is not provided)
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Temperature parameter
            model: Model name
            **kwargs: Additional parameters (top_p, max_tokens, etc.)
            
        Returns:
            The response text from the API
        """
        client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['api_base']
        )
        
        model_name = self._get_model(model)
        temp = self._get_temperature(temperature)
        
        # Get provider name for error messages
        provider_name = self.config.get('api_provider', 'OpenAI-compatible')
        
        # Prepare messages
        if messages:
            api_messages = messages
        elif prompt:
            api_messages = [{'role': 'user', 'content': prompt}]
        else:
            raise ValueError("Either prompt or messages must be provided")
        
        # Prepare parameters
        params = {
            'model': model_name,
            'messages': api_messages,
            'temperature': temp,
        }
        
        # Add optional parameters
        if 'api_top_p' in self.config:
            params['top_p'] = self.config['api_top_p']
        if 'top_p' in kwargs:
            params['top_p'] = kwargs['top_p']
        if 'max_tokens' in kwargs:
            params['max_tokens'] = kwargs['max_tokens']
        
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"{provider_name} API call failed: {str(e)}") from e

