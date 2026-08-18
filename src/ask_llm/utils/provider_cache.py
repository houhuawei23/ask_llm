"""Global provider adapter cache for connection reuse across runs.

Creating a provider adapter (and the underlying HTTP client) is not free.
This module provides a process-wide LRU cache so that repeated calls to the
same provider/model reuse the same adapter instance, keeping HTTP connections
warm and reducing startup latency.

Engine access goes through ``ask_llm.utils.engine_facade`` (P4.6); the
``EngineConfigView`` compatibility import lives here too.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ask_llm.core.models import ProviderConfig
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.utils.engine_facade import EngineConfigView, create_engine_adapter

__all__ = ["EngineConfigView", "ProviderAdapterCache"]


def _to_provider_config(config: ProviderConfig) -> ProviderConfig:
    """Type guard: cache inputs must be real ``ProviderConfig`` objects.

    Accepting a ``dict`` here was the root cause of the v2.15.1 adapter
    dict-vs-object crash; the deprecated dict path was removed.
    """
    if not isinstance(config, ProviderConfig):
        raise TypeError(
            f"ProviderAdapterCache.get expects a ProviderConfig, got {type(config).__name__}"
        )
    return config


@lru_cache(maxsize=128)
def _create_cached_adapter(
    provider: str,
    api_base: str,
    api_key: str,
    models: tuple[str, ...],
    api_temperature: float,
    api_top_p: float | None,
    max_tokens: int | None,
    timeout: float,
    default_model: str,
) -> LLMProviderProtocol:
    """Create a provider adapter from primitive, hashable fields.

    The lru_cache wrapper guarantees that the same HTTP client is reused for
    identical provider configurations. The adapter is created from a real
    ``ProviderConfig`` object so that downstream code can access
    ``adapter.config.api_temperature`` and other attributes consistently.
    """
    provider_config = ProviderConfig(
        api_provider=provider,
        api_base=api_base,
        api_key=api_key,
        models=list(models),
        api_temperature=api_temperature,
        api_top_p=api_top_p,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return create_engine_adapter(provider_config, default_model=default_model or None)


class ProviderAdapterCache:
    """Process-wide cache for llm-engine provider adapters.

    Example:
        adapter = ProviderAdapterCache.get(provider_config, default_model="gpt-4")
        # Subsequent calls with the same config return the same adapter.
    """

    @classmethod
    def get(
        cls,
        config: ProviderConfig | dict[str, Any],
        *,
        default_model: str | None = None,
    ) -> LLMProviderProtocol:
        """Get or create a cached provider adapter.

        Args:
            config: Provider configuration. A ``ProviderConfig`` object is the
                supported input; a ``dict`` is accepted for backward
                compatibility (emits ``DeprecationWarning``).
            default_model: Default model name for the adapter.

        Returns:
            A cached or newly created provider adapter.

        Raises:
            TypeError: If ``config`` is neither a ``ProviderConfig`` nor a dict.
        """
        pc = _to_provider_config(config)
        return _create_cached_adapter(
            pc.api_provider,
            pc.api_base,
            pc.get_api_key(),
            tuple(pc.models),
            float(pc.api_temperature),
            float(pc.api_top_p) if pc.api_top_p is not None else None,
            int(pc.max_tokens) if pc.max_tokens is not None else None,
            float(pc.timeout),
            default_model or "",
        )

    @classmethod
    def clear(cls) -> None:
        """Clear the adapter cache.

        Call this when provider credentials change or before process shutdown to
        release underlying HTTP connections.
        """
        _create_cached_adapter.cache_clear()
