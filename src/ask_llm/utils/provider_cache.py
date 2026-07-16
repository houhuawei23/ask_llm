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
from ask_llm.core.protocols import LLMProviderProtocol

_DICT_DEPRECATION_MSG = (
    "Passing a dict to ProviderAdapterCache.get is deprecated; pass a "
    "ProviderConfig object instead. Support for dict inputs will be removed "
    "in a future release."
)


def _to_provider_config(config: ProviderConfig | dict[str, Any]) -> ProviderConfig:
    """Coerce a cache input into a real ``ProviderConfig`` object.

    Accepting a ``dict`` here is the root cause of the v2.15.1
    adapter dict-vs-object crash: a dict leaked through this seam and broke
    downstream ``adapter.config.api_temperature`` attribute access. Dicts are
    still accepted for backward compatibility but emit a ``DeprecationWarning``
    and are rebuilt as a ``ProviderConfig`` so the adapter always carries an
    object-typed config.
    """
    if isinstance(config, ProviderConfig):
        return config
    if isinstance(config, dict):
        import warnings

        warnings.warn(_DICT_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
        return ProviderConfig(**config)
    raise TypeError(
        f"ProviderAdapterCache.get expects a ProviderConfig (or dict), got {type(config).__name__}"
    )


class EngineConfigView:
    """Plain-attribute view of a ``ProviderConfig`` for the llm_engine boundary.

    llm_engine reads ``config.api_key`` as a plain string (``getattr``) and
    forwards it to the HTTP client. Our ``ProviderConfig`` stores the key as
    ``SecretStr``, so this view unwraps it exactly once at the boundary and
    masks it again in ``repr`` to avoid accidental log leaks.
    """

    __slots__ = (
        "api_base",
        "api_key",
        "api_provider",
        "api_temperature",
        "api_top_p",
        "max_tokens",
        "models",
        "timeout",
    )

    def __init__(self, pc: ProviderConfig):
        self.api_provider = pc.api_provider
        self.api_key = pc.get_api_key()
        self.api_base = pc.api_base
        self.models = list(pc.models)
        self.api_temperature = pc.api_temperature
        self.api_top_p = pc.api_top_p
        self.max_tokens = pc.max_tokens
        self.timeout = pc.timeout

    def __repr__(self) -> str:
        masked = "***" if self.api_key else ""
        return (
            f"EngineConfigView(api_provider={self.api_provider!r}, api_key={masked!r}, "
            f"api_base={self.api_base!r}, models={self.models!r})"
        )


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
    return create_provider_adapter(
        EngineConfigView(provider_config), default_model=default_model or None
    )


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
