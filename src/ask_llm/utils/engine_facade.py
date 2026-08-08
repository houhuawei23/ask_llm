"""Single seam for all ``llm_engine`` imports (P4.6).

Every ``llm_engine`` symbol used by ask_llm enters through this module —
the external engine is a module-private dependency everywhere else
(grep invariant: ``llm_engine`` appears only here and in comments).

Exports:
- :class:`EngineConfigView` — plain-attribute ``ProviderConfig`` view that
  unwraps the ``SecretStr`` key exactly once for the HTTP client boundary.
- :func:`create_engine_adapter` — adapter creation (fresh, uncached; for a
  cached adapter use ``ProviderAdapterCache``, which delegates here).
- :func:`load_engine_providers_config` — engine's providers.yml catalog
  (base_url fallback resolution), returning ``{}`` on any failure.
"""

from __future__ import annotations

from typing import Any

from llm_engine import create_provider_adapter as _create_provider_adapter

from ask_llm.core.models import ProviderConfig
from ask_llm.core.protocols import LLMProviderProtocol


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


def create_engine_adapter(
    config: ProviderConfig | EngineConfigView,
    *,
    default_model: str | None = None,
) -> LLMProviderProtocol:
    """Create a fresh (uncached) llm_engine provider adapter.

    Accepts a ``ProviderConfig`` (wrapped in :class:`EngineConfigView` to
    unwrap the ``SecretStr`` key) or a pre-built ``EngineConfigView``.
    For connection reuse, prefer ``ProviderAdapterCache.get``.
    """
    view = config if isinstance(config, EngineConfigView) else EngineConfigView(config)
    return _create_provider_adapter(view, default_model=default_model)


def load_engine_providers_config() -> dict[str, Any]:
    """Load the engine's own providers catalog (base_url fallback source).

    Returns ``{}`` when the engine loader is unavailable or fails; callers
    treat that as "no fallback data".
    """
    try:
        from llm_engine.config_loader import load_providers_config

        return load_providers_config() or {}
    except Exception:
        return {}
