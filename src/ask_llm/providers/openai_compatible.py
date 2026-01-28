"""OpenAI Compatible API provider implementation."""

from typing import Any, Dict, Generator, List, Optional, Union

from loguru import logger
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

from ask_llm.providers.base import BaseProvider
from ask_llm.core.models import ProviderConfig


class OpenAICompatibleProvider(BaseProvider):
    """
    Provider for OpenAI-compatible APIs.
    
    Supports DeepSeek, Qwen, and other OpenAI-compatible endpoints.
    """
    
    def __init__(self, config: ProviderConfig, default_model: Optional[str] = None):
        """
        Initialize the provider.
        
        Args:
            config: Provider configuration
            default_model: Default model name (if None, uses first model from models list)
        """
        super().__init__(config, default_model=default_model)
        self._client: Optional[OpenAI] = None
    
    def validate_config(self) -> None:
        """
        Validate OpenAI-compatible configuration.
        
        Raises:
            ValueError: If required fields are missing
        """
        required = ["api_key", "api_base"]
        missing = []
        
        for field in required:
            value = getattr(self.config, field, None)
            if not value or value == "your-api-key-here":
                missing.append(field)
        
        if missing:
            raise ValueError(
                f"Missing required configuration fields: {', '.join(missing)}"
            )
        
        if not self.config.models:
            raise ValueError(
                f"Provider '{self.config.api_provider}' must have at least one model in 'models' list"
            )
    
    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
            )
        return self._client
    
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
        Call OpenAI-compatible API.
        
        Args:
            prompt: Single prompt text
            messages: List of message dictionaries
            temperature: Sampling temperature
            model: Model name override
            stream: Whether to stream response
            **kwargs: Additional parameters (max_tokens, top_p, etc.)
            
        Returns:
            Response string or streaming generator
            
        Raises:
            ValueError: If neither prompt nor messages provided
            RuntimeError: If API call fails
        """
        # Validate inputs
        if messages:
            api_messages = messages
        elif prompt:
            api_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        model_name = self._get_model(model)
        temp = self._get_temperature(temperature)
        
        # Build API parameters
        params: Dict[str, Any] = {
            "model": model_name,
            "messages": api_messages,
            "temperature": temp,
            "stream": stream,
        }
        
        # Add optional parameters
        if self.config.api_top_p is not None:
            params["top_p"] = self.config.api_top_p
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        
        # Apply any additional kwargs
        for key in ["max_tokens", "top_p", "presence_penalty", "frequency_penalty"]:
            if key in kwargs:
                params[key] = kwargs[key]
        
        logger.debug(
            f"Calling {self.name} API with model={model_name}, "
            f"temperature={temp}, stream={stream}"
        )
        
        try:
            if stream:
                return self._stream_response(params)
            else:
                return self._complete_response(params)
                
        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise RuntimeError(
                f"API authentication failed. Please check your API key."
            ) from e
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise RuntimeError(
                f"API rate limit exceeded. Please try again later."
            ) from e
        except APIError as e:
            logger.error(f"API error: {e}")
            raise RuntimeError(f"API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise RuntimeError(f"API call failed: {e}") from e
    
    def _complete_response(self, params: Dict[str, Any]) -> str:
        """
        Get complete (non-streaming) response.
        
        Args:
            params: API parameters
            
        Returns:
            Response text
        """
        # Remove stream parameter for non-streaming call
        params = {k: v for k, v in params.items() if k != "stream"}
        
        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""
        
        logger.debug(f"Received response: {len(content)} characters")
        return content.strip()
    
    def _stream_response(
        self,
        params: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        Stream response from API.
        
        Args:
            params: API parameters
            
        Yields:
            Text chunks from the streaming response
        """
        try:
            stream = self.client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except GeneratorExit:
            # Handle generator cleanup
            logger.debug("Stream generator closed")
            raise
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise RuntimeError(f"Stream failed: {e}") from e
    
    def test_connection(self, test_message: str = "Hello") -> tuple[bool, str, float]:
        """
        Test API connection.
        
        Args:
            test_message: Test message to send
            
        Returns:
            Tuple of (success, message, latency_seconds)
        """
        import time
        
        start = time.time()
        try:
            response = self.call(
                prompt=test_message,
                max_tokens=10,
                temperature=0.0
            )
            latency = time.time() - start
            return True, f"Response: {response[:50]}...", latency
        except Exception as e:
            latency = time.time() - start
            return False, str(e), latency
