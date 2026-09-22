"""Global provider adapter cache for connection reuse across runs.

Creating a provider adapter (and the underlying HTTP client) is not free.
This module provides a process-wide LRU cache so that repeated calls to the
same provider/model reuse the same adapter instance, keeping HTTP connections
warm and reducing startup latency.

Audit 3.4 lifecycle guarantees (previously an unbounded leak of up to 128
never-closed HTTP clients):

- The cache key carries a ``sha256`` digest of the API key, never the
  plaintext key (the previous ``lru_cache`` retained every raw key in its
  key tuples for the lifetime of the process).
- A generation counter is part of the key; :meth:`ProviderAdapterCache.clear`
  bumps it, so adapters created under previous credentials can never be
  silently reused after a credential rotation.
- Evicted and cleared adapters get a best-effort ``close()`` of their
  underlying HTTP client; an ``atexit`` hook releases whatever is still
  cached when the process exits.

Engine access goes through ``ask_llm.utils.engine_facade`` (P4.6); the
``EngineConfigView`` compatibility import lives here too.
"""

from __future__ import annotations

import atexit
import hashlib
import threading
from collections import OrderedDict
from typing import Any

from loguru import logger
from pydantic import SecretStr

from ask_llm.core.models import ProviderConfig
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.utils.engine_facade import EngineConfigView, create_engine_adapter

__all__ = ["EngineConfigView", "ProviderAdapterCache"]

_CACHE_MAXSIZE = 128

_lock = threading.Lock()
# Keyed by (generation, provider, api_base, key_digest, models, sampling
# params, timeout, default_model) — enough to uniquely identify a connection
# while keeping the plaintext API key out of memory-resident tuples.
_adapters: OrderedDict[tuple[Any, ...], LLMProviderProtocol] = OrderedDict()
_generation = 0


def _drain() -> list[LLMProviderProtocol]:
    """Atomically empty the cache and return its adapters (bumps generation)."""
    global _generation
    with _lock:
        _generation += 1
        stale = list(_adapters.values())
        _adapters.clear()
    return stale


def _close_adapter(adapter: LLMProviderProtocol) -> None:
    """Best-effort release of *adapter*'s underlying HTTP client.

    llm_engine's ``ProviderAdapter`` has no ``close()`` (yet); its
    OpenAI-compatible provider keeps the SDK client at ``_provider._client``.
    Only raw attributes are probed — the provider's ``client`` property
    lazily *creates* a client when missing, which at shutdown would build a
    connection nobody uses. Every failure is swallowed: closing is hygiene,
    never an error at eviction or interpreter exit.
    """
    candidates: list[Any] = [adapter]
    inner = getattr(adapter, "_provider", None)
    if inner is not None:
        candidates.append(inner)
    for obj in candidates:
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
                return
            except Exception as e:
                logger.debug(f"Provider adapter close() failed (ignored): {e}")
                return
        client = getattr(obj, "_client", None)
        client_close = getattr(client, "close", None)
        if callable(client_close):
            try:
                client_close()
                return
            except Exception as e:
                logger.debug(f"Provider HTTP client close() failed (ignored): {e}")
                return


def _close_all_adapters() -> None:
    """Release every still-cached adapter (``atexit`` hook)."""
    for adapter in _drain():
        _close_adapter(adapter)


atexit.register(_close_all_adapters)


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


def _cache_key(config: ProviderConfig, *, default_model: str, generation: int) -> tuple[Any, ...]:
    """Hashable identity of a connection — digest of the key, not the key."""
    return (
        generation,
        config.api_provider,
        config.api_base,
        hashlib.sha256(config.get_api_key().encode("utf-8")).hexdigest(),
        tuple(config.models),
        float(config.api_temperature),
        float(config.api_top_p) if config.api_top_p is not None else None,
        int(config.max_tokens) if config.max_tokens is not None else None,
        float(config.timeout),
        default_model,
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
        config: ProviderConfig,
        *,
        default_model: str | None = None,
    ) -> LLMProviderProtocol:
        """Get or create a cached provider adapter.

        Args:
            config: Provider configuration object (the historically accepted
                ``dict`` input was removed with the v2.15 crash path — the
                signature said otherwise, which was misleading).
            default_model: Default model name for the adapter.

        Returns:
            A cached or newly created provider adapter.

        Raises:
            TypeError: If ``config`` is not a ``ProviderConfig``.
        """
        pc = _to_provider_config(config)
        key = _cache_key(pc, default_model=default_model or "", generation=_generation)
        with _lock:
            adapter = _adapters.get(key)
            if adapter is not None:
                _adapters.move_to_end(key)
                return adapter

        fresh = create_engine_adapter(
            ProviderConfig(
                api_provider=pc.api_provider,
                api_base=pc.api_base,
                api_key=SecretStr(pc.get_api_key()),
                models=list(pc.models),
                api_temperature=pc.api_temperature,
                api_top_p=pc.api_top_p,
                max_tokens=pc.max_tokens,
                timeout=pc.timeout,
            ),
            default_model=default_model or None,
        )

        with _lock:
            winner = _adapters.get(key)
            if winner is not None:
                # Another thread created the same adapter while we built ours;
                # release the loser's (lazily unconnected) client and reuse his.
                _close_adapter(fresh)
                _adapters.move_to_end(key)
                return winner
            _adapters[key] = fresh
            while len(_adapters) > _CACHE_MAXSIZE:
                _, evicted = _adapters.popitem(last=False)
                _close_adapter(evicted)
            return fresh

    @classmethod
    def clear(cls) -> None:
        """Drop all cached adapters and close their HTTP clients.

        Call this when provider credentials change: the generation bump
        guarantees the next ``get`` for the same config builds a fresh adapter
        instead of silently reusing one created under the old credentials.
        """
        for adapter in _drain():
            _close_adapter(adapter)

    @classmethod
    def close(cls) -> None:
        """Release every cached adapter's HTTP client (process shutdown)."""
        cls.clear()
