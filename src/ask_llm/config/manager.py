"""Configuration management and CLI overrides."""

from typing import Any

from loguru import logger

from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.models import ProviderConfig


class ConfigManager:
    """Manage application configuration with CLI overrides."""

    def __init__(self, unified_config: UnifiedConfig):
        """
        Initialize with the unified configuration (single config object).

        Args:
            unified_config: The validated UnifiedConfig (providers + behavior
                sections). There is no second config object: providers and
                rate limits / sections come from the same instance.
        """
        self._base_config = unified_config
        self._current_provider = unified_config.default_provider
        if not unified_config.providers:
            # M17/2.25: an empty provider table used to raise a bare
            # StopIteration from the fallback below — obscure and unactionable.
            raise ValueError(
                "No providers configured. Add at least one provider to "
                "providers.yml or default_config.yml, or run 'ask-llm config init'."
            )
        if self._current_provider not in unified_config.providers:
            # Fail fast with a clear fallback instead of surfacing a confusing
            # "Provider '...' not found" on the first get_provider_config() call.
            fallback = next(iter(unified_config.providers))
            logger.warning(
                f"Default provider '{self._current_provider}' not found in configured "
                f"providers; falling back to '{fallback}'"
            )
            self._current_provider = fallback
        # H1: overrides are stored **per provider**. A single global slot made
        # ``get_provider_config(other)`` silently inherit the current
        # provider's api_key / sampling overrides (e.g. a key pasted into the
        # interactive gate leaked into every fallback / batch provider).
        self._overrides: dict[str, dict[str, Any]] = {}
        # Model override is a run-level intent, not a provider field: it never
        # belonged inside the ProviderConfig payload (it was previously smuggled
        # in as a fake "_model_override" key that pydantic happened to ignore).
        self._model_override: str | None = None
        # Track the source of each override for transparency/debugging.
        # Maps override key -> source label (e.g. "CLI", "ENV", "default_config.yml").
        self._override_sources: dict[str, str] = {}

    @property
    def unified_config(self) -> UnifiedConfig:
        """Get the unified configuration (the single config object)."""
        return self._base_config

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
        Get provider configuration with that provider's own overrides applied.

        Overrides recorded via :meth:`apply_overrides` are attributed to the
        provider that was current when they were applied; requesting another
        provider's config never inherits them (H1).

        Args:
            provider_name: Provider name (uses current if None)

        Returns:
            Provider configuration with overrides
        """
        name = provider_name or self._current_provider
        base = self._base_config.get_provider_config(name)

        # No overrides (hot path): hand out the base config as-is — a
        # model_dump → model_validate round-trip would re-run field validators
        # and re-log an "empty API key" warning on every call for key-less
        # providers (e.g. ollama).
        overrides = self._overrides.get(name)
        if not overrides:
            return base

        # With overrides, rebuild through validation so plain strings (e.g. an
        # api_key override) are coerced to the declared field types (SecretStr).
        config_dict = base.model_dump()
        config_dict.update(overrides)

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
        Apply CLI argument overrides to the **current** provider.

        Overrides are attributed to the provider that is current at call time
        (every call site performs ``set_provider`` first). Field overrides never
        leak to other providers (H1); the model override is run-level and stays
        global.

        Args:
            model: Override model name (run-level, see :meth:`get_model_override`)
            temperature: Override temperature
            api_key: Override API key
            api_base: Override API base URL
            source: Origin label for traceability (e.g. "CLI", "ENV").
            **kwargs: Additional overrides
        """
        bucket = self._overrides.setdefault(self._current_provider, {})

        if model is not None:
            self._model_override = model
            self._override_sources["model"] = f"{source}: {model}"
            logger.debug(f"Override: model = {model} (source={source})")

        if temperature is not None:
            bucket["api_temperature"] = temperature
            self._override_sources["temperature"] = f"{source}: {temperature}"
            logger.debug(f"Override: temperature = {temperature} (source={source})")

        if api_key is not None:
            bucket["api_key"] = api_key
            self._override_sources["api_key"] = f"{source}: ***"
            logger.debug(f"Override: api_key = *** (source={source})")

        if api_base is not None:
            bucket["api_base"] = api_base
            self._override_sources["api_base"] = f"{source}: {api_base}"
            logger.debug(f"Override: api_base = {api_base} (source={source})")

        for key, value in kwargs.items():
            if value is not None:
                display_value = "***" if "key" in key.lower() else str(value)
                bucket[key] = value
                self._override_sources[key] = f"{source}: {display_value}"
                logger.debug(f"Override: {key} = {display_value} (source={source})")

    def get_model_override(self) -> str | None:
        """
        Get model override if set.

        Returns:
            Model name override or None
        """
        return self._model_override

    def get_default_model(self, provider_name: str | None = None) -> str:
        """
        Get default model for a provider.

        Priority:
            1. Provider's own models[0] (which is its default_model set during config load)
            2. Global default_model from UnifiedConfig
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
        """Clear all overrides (every provider and the model override)."""
        self._overrides.clear()
        self._model_override = None
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
