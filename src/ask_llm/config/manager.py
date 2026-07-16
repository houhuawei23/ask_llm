"""Configuration management and CLI overrides."""

from typing import Any

from loguru import logger

from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.models import AppConfig, ProviderConfig


class ConfigManager:
    """Manage application configuration with CLI overrides."""

    def __init__(self, config: AppConfig, unified_config: UnifiedConfig | None = None):
        """
        Initialize with base configuration.

        Args:
            config: Base application configuration
            unified_config: Unified (non-provider) configuration sections. Optional for
                backward compatibility; callers that need rate limits or unified
                sections should always pass it (load_cli_session does).
        """
        self._base_config = config
        self._unified_config = unified_config
        self._current_provider: str = config.default_provider
        self._overrides: dict[str, Any] = {}
        # Track the source of each override for transparency/debugging.
        # Maps override key -> source label (e.g. "CLI", "ENV", "default_config.yml").
        self._override_sources: dict[str, str] = {}

    @property
    def config(self) -> AppConfig:
        """Get base configuration."""
        return self._base_config

    @property
    def unified_config(self) -> UnifiedConfig | None:
        """Get the unified (non-provider) configuration, if attached."""
        return self._unified_config

    @property
    def current_provider_name(self) -> str:
        """Get current provider name."""
        return self._current_provider

    def set_provider(self, name: str, *, source: str = "CLI") -> None:
        """
        Set current provider.

        Args:
            name: Provider name
            source: Origin of this change (for override tracing).

        Raises:
            ValueError: If provider not found
        """
        if name not in self._base_config.providers:
            available = ", ".join(self._base_config.providers.keys())
            raise ValueError(f"Provider '{name}' not found. Available: {available}")
        self._current_provider = name
        self._override_sources["provider"] = f"{source}: {name}"
        logger.debug(f"Switched to provider: {name} (source={source})")

    def get_provider_config(self, provider_name: str | None = None) -> ProviderConfig:
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
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        *,
        source: str = "CLI",
        **kwargs: Any,
    ) -> None:
        """
        Apply CLI argument overrides.

        Args:
            model: Override model name
            temperature: Override temperature
            api_key: Override API key
            api_base: Override API base URL
            source: Origin label for traceability (e.g. "CLI", "ENV").
            **kwargs: Additional overrides
        """
        if model is not None:
            self._overrides["_model_override"] = model
            self._override_sources["model"] = f"{source}: {model}"
            logger.debug(f"Override: model = {model} (source={source})")

        if temperature is not None:
            self._overrides["api_temperature"] = temperature
            self._override_sources["temperature"] = f"{source}: {temperature}"
            logger.debug(f"Override: temperature = {temperature} (source={source})")

        if api_key is not None:
            self._overrides["api_key"] = api_key
            self._override_sources["api_key"] = f"{source}: ***"
            logger.debug(f"Override: api_key = *** (source={source})")

        if api_base is not None:
            self._overrides["api_base"] = api_base
            self._override_sources["api_base"] = f"{source}: {api_base}"
            logger.debug(f"Override: api_base = {api_base} (source={source})")

        for key, value in kwargs.items():
            if value is not None:
                display_value = "***" if "key" in key.lower() else str(value)
                self._overrides[key] = value
                self._override_sources[key] = f"{source}: {display_value}"
                logger.debug(f"Override: {key} = {display_value} (source={source})")

    def get_model_override(self) -> str | None:
        """
        Get model override if set.

        Returns:
            Model name override or None
        """
        return self._overrides.get("_model_override")

    def get_default_model(self, provider_name: str | None = None) -> str:
        """
        Get default model for a provider.

        Priority:
            1. Provider's own models[0] (which is its default_model set during config load)
            2. Global default_model from AppConfig
            3. Raise error if neither is available

        Args:
            provider_name: Provider name (uses current if None)

        Returns:
            Default model name
        """
        config = self.get_provider_config(provider_name)
        # First use provider's own default_model (models[0] after config load reordering)
        if config.models:
            return config.models[0]

        # Fallback to global default_model
        if self._base_config.default_model:
            return self._base_config.default_model

        raise ValueError(
            f"No default model available for provider '{provider_name or self._current_provider}'"
        )

    def clear_overrides(self) -> None:
        """Clear all overrides."""
        self._overrides.clear()
        self._override_sources.clear()
        logger.debug("Cleared all configuration overrides")

    def get_override_sources(self) -> dict[str, str]:
        """Return a mapping of override key -> human-readable source label.

        Useful for debugging configuration provenance (e.g. ``config show --debug-config``).
        Keys without an override entry inherit from ``default_config.yml``.
        """
        return dict(self._override_sources)

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return list(self._base_config.providers.keys())

    def get_available_models(self, provider_name: str | None = None) -> list[str]:
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
