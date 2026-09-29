"""Single seam for provider adapter creation (P4.6; litellm backend since 2.27).

Every provider adapter enters ask_llm through this module:

- :func:`create_engine_adapter` — adapter creation (fresh, uncached; for a
  cached adapter use ``ProviderAdapterCache``, which delegates here).
- :func:`load_engine_providers_config` — the ``providers.yml`` catalog used as
  the base_url fallback source, returning ``{}`` on any failure.

The adapter itself lives in :mod:`ask_llm.core.provider_adapter`;
``litellm`` is imported lazily and only there.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ask_llm.config.providers_catalog import (
    load_first_providers_yml,
    runtime_providers_yml_paths,
)
from ask_llm.core.models import ProviderConfig
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.core.provider_adapter import LiteLLMProviderAdapter

__all__ = ["create_engine_adapter", "load_engine_providers_config"]


def create_engine_adapter(
    config: ProviderConfig,
    *,
    default_model: str | None = None,
) -> LLMProviderProtocol:
    """Create a fresh (uncached) litellm-backed provider adapter.

    For connection reuse, prefer ``ProviderAdapterCache.get``, which delegates
    here.
    """
    return LiteLLMProviderAdapter(config, default_model=default_model)


def load_engine_providers_config() -> dict[str, Any]:
    """Load the providers.yml catalog used as the base_url fallback source.

    Reads *runtime* paths only (env override, user config dir, packaged copy)
    — never cwd — because the merged provider config sends the resolved API
    key to the ``base_url`` found here. Returns ``{}`` when no usable catalog
    is found; callers treat that as "no fallback data".
    """
    try:
        data, _source = load_first_providers_yml(paths=runtime_providers_yml_paths())
        return data or {}
    except Exception as e:
        logger.warning(f"providers.yml catalog unavailable ({e}); no base_url fallback data.")
        return {}
