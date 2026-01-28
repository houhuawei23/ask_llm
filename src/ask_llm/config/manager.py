"""Configuration management and CLI overrides."""

from typing import Any, Dict, List, Optional

from loguru import logger

from ask_llm.core.models import AppConfig, ProviderConfig


class ConfigManager:
    """Manage application configuration with CLI overrides."""

    def __init__(self, config: AppConfig):
        """
        Initialize with base configuration.

        Args:
            config: Base application configuration
        """
        self._base_config = config
        self._current_provider: str = config.default_provider
        self._overrides: Dict[str, Any] = {}

    @property
    def config(self) -> AppConfig:
        """Get base configuration."""
        return self._base_config

    @property
    def current_provider_name(self) -> str:
        """Get current provider name."""
        return self._current_provider

    def set_provider(self, name: str) -> None:
        """
        Set current provider.

        Args:
            name: Provider name

        Raises:
            ValueError: If provider not found
        """
        if name not in self._base_config.providers:
            available = ", ".join(self._base_config.providers.keys())
            raise ValueError(f"Provider '{name}' not found. Available: {available}")
        self._current_provider = name
        logger.debug(f"Switched to provider: {name}")

    def get_provider_config(self, provider_name: Optional[str] = None) -> ProviderConfig:
        """
        Get provider configuration with overrides applied.

        Args:
            provider_name: Provider name (uses current if None)

        Returns:
            Provider configuration with overrides
        """
        name = provider_name or self._current_provider
        base = self._base_config.get_provider_config(name)

        # Create a copy with overrides
        config_dict = base.model_dump()
        config_dict.update(self._overrides)

        return ProviderConfig.model_validate(config_dict)

    def apply_overrides(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Apply CLI argument overrides.

        Args:
            model: Override model name
            temperature: Override temperature
            api_key: Override API key
            api_base: Override API base URL
            **kwargs: Additional overrides
        """
        if model is not None:
            self._overrides["_model_override"] = model
            logger.debug(f"Override: model = {model}")

        if temperature is not None:
            self._overrides["api_temperature"] = temperature
            logger.debug(f"Override: temperature = {temperature}")

        if api_key is not None:
            self._overrides["api_key"] = api_key
            logger.debug("Override: api_key = ***")

        if api_base is not None:
            self._overrides["api_base"] = api_base
            logger.debug(f"Override: api_base = {api_base}")

        for key, value in kwargs.items():
            if value is not None:
                self._overrides[key] = value
                logger.debug(f"Override: {key} = {value}")

    def get_model_override(self) -> Optional[str]:
        """
        Get model override if set.

        Returns:
            Model name override or None
        """
        return self._overrides.get("_model_override")

    def get_default_model(self, provider_name: Optional[str] = None) -> str:
        """
        Get default model for a provider.

        Args:
            provider_name: Provider name (uses current if None)

        Returns:
            Default model name
        """
        # First check if there's a global default_model
        if self._base_config.default_model:
            return self._base_config.default_model

        # Otherwise use first model from provider's models list
        config = self.get_provider_config(provider_name)
        if config.models:
            return config.models[0]

        raise ValueError(
            f"No default model available for provider '{provider_name or self._current_provider}'"
        )

    def clear_overrides(self) -> None:
        """Clear all overrides."""
        self._overrides.clear()
        logger.debug("Cleared all configuration overrides")

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self._base_config.providers.keys())

    def get_available_models(self, provider_name: Optional[str] = None) -> List[str]:
        """
        Get list of available models for a provider.

        Args:
            provider_name: Provider name (uses current if None)

        Returns:
            List of model names
        """
        config = self.get_provider_config(provider_name)
        if config.models:
            return config.models
        # Fallback to default_model if no models list
        default_model = self.get_default_model(provider_name)
        return [default_model]
