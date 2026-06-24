"""Global provider adapter cache for connection reuse across runs.

Creating a provider adapter (and the underlying HTTP client) is not free.
This module provides a process-wide LRU cache so that repeated calls to the
same provider/model reuse the same adapter instance, keeping HTTP connections
warm and reducing startup latency.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from llm_engine import create_provider_adapter

from ask_llm.core.models import ProviderConfig


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
) -> Any:
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
    return create_provider_adapter(provider_config, default_model=default_model or None)


class ProviderAdapterCache:
    """Process-wide cache for llm-engine provider adapters.

    Example:
        adapter = ProviderAdapterCache.get(adapter_config, default_model="gpt-4")
        # Subsequent calls with the same config return the same adapter.
    """

    @classmethod
    def get(cls, config: Any, *, default_model: str | None = None) -> Any:
        """Get or create a cached provider adapter.

        Args:
            config: Provider configuration object (ProviderConfigProtocol or dict).
            default_model: Default model name for the adapter.

        Returns:
            A cached or newly created provider adapter.
        """
        if isinstance(config, dict):
            provider = config.get("api_provider", "")
            api_base = config.get("api_base", "")
            api_key = config.get("api_key", "")
            models = tuple(config.get("models", []))
            api_temperature = config.get("api_temperature", 0.7)
            api_top_p = config.get("api_top_p")
            max_tokens = config.get("max_tokens")
            timeout = config.get("timeout", 60.0)
        else:
            provider = getattr(config, "api_provider", "")
            api_base = getattr(config, "api_base", "")
            api_key = getattr(config, "api_key", "")
            models = tuple(getattr(config, "models", []))
            api_temperature = getattr(config, "api_temperature", 0.7)
            api_top_p = getattr(config, "api_top_p", None)
            max_tokens = getattr(config, "max_tokens", None)
            timeout = getattr(config, "timeout", 60.0)

        return _create_cached_adapter(
            provider,
            api_base,
            api_key,
            models,
            float(api_temperature),
            float(api_top_p) if api_top_p is not None else None,
            int(max_tokens) if max_tokens is not None else None,
            float(timeout),
            default_model or "",
        )

    @classmethod
    def clear(cls) -> None:
        """Clear the adapter cache.

        Call this when provider credentials change or before process shutdown to
        release underlying HTTP connections.
        """
        _create_cached_adapter.cache_clear()

    @classmethod
    def info(cls) -> dict[str, Any]:
        """Return cache statistics."""
        cache_info = _create_cached_adapter.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
        }
