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
- :func:`bind_engine_providers_yml` — points the engine's *own* providers.yml
  lookup at ask_llm's resolved runtime catalog instead of the process cwd.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from llm_engine import create_provider_adapter as _create_provider_adapter
from loguru import logger

from ask_llm.config.providers_catalog import resolved_runtime_providers_yml
from ask_llm.core.models import ProviderConfig
from ask_llm.core.protocols import LLMProviderProtocol

_BINDING_LOCK = threading.Lock()
# Unwrapped engine loader recorded by the first bind, restored by the reset helper.
_original_load_providers_config: Callable[..., dict[str, Any]] | None = None


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


def bind_engine_providers_yml() -> Path | None:
    """Make ``llm_engine`` read its per-model metadata from ask_llm's catalog.

    ``llm_engine`` resolves flags such as ``reasoning`` (the DeepSeek
    ``thinking: {"type": "disabled"}`` toggle), ``token_per_character`` and
    ``functions.json_output`` by calling ``load_providers_config()`` with no
    path, whose search order starts at ``Path("providers.yml")`` — i.e. **the
    process cwd**. Nothing else on that list exists in this deployment
    (``site-packages/providers.yml`` and ``~/.config/llm-engine/providers.yml``
    are not shipped), so the lookup silently degrades whenever ask-llm is run
    outside a directory containing ``providers.yml``:

    - ``reasoning: false`` is lost → ``thinking`` is left enabled → a thinking
      model (e.g. ``deepseek-v4-flash``) streams 30-170s+ of
      ``reasoning_content`` before any content. Translation/format pipelines
      ignore reasoning, so their progress bar sits at "Processing..." with no
      tokens and the run looks hung (and max_tokens can be burned into an
      empty body).
    - ``token_per_character`` / ``json_output`` fall back to defaults.

    Binding the lookup to :func:`resolved_runtime_providers_yml` (the same
    file ask_llm already resolved and validated, cwd excluded) makes behavior
    independent of the directory ask-llm was launched from. Idempotent; the
    patch only supplies the default path, explicit ``config_path`` callers are
    untouched.

    Returns:
        The bound catalog path, or ``None`` when no runtime catalog exists (the
        engine's legacy cwd search then stays in effect for that lookup).
    """
    global _original_load_providers_config

    try:
        from llm_engine import config_loader as engine_config_loader
    except Exception as e:  # pragma: no cover - import guard
        logger.warning(f"llm_engine config_loader unavailable ({e}); catalog not bound.")
        return None

    with _BINDING_LOCK:
        original = engine_config_loader.load_providers_config
        if not getattr(original, "__ask_llm_bound__", False):

            def load_providers_config(config_path: Path | None = None) -> dict[str, Any]:
                """Engine loader wrapper: default to ask_llm's runtime catalog."""
                if config_path is None:
                    config_path = resolved_runtime_providers_yml()
                return cast(dict[str, Any], original(config_path))

            # Mark the wrapper so repeated binding never stacks wrappers.
            load_providers_config.__ask_llm_bound__ = True  # type: ignore[attr-defined]
            load_providers_config.__doc__ = original.__doc__
            engine_config_loader.load_providers_config = load_providers_config
            if _original_load_providers_config is None:
                _original_load_providers_config = original
            first_bind = True
        else:
            first_bind = False

    resolved = resolved_runtime_providers_yml()
    if first_bind:
        if resolved is None:
            logger.warning(
                "No runtime providers.yml resolved; llm_engine per-model flags "
                "(reasoning/json_output/token_per_character) are unavailable and a "
                "providers.yml in the current directory may be used instead."
            )
        else:
            logger.debug(f"Bound llm_engine providers.yml lookup to: {resolved}")
    return resolved


def _reset_engine_providers_yml_binding() -> None:
    """Undo :func:`bind_engine_providers_yml` (tests / re-binding after config swap)."""
    global _original_load_providers_config

    with _BINDING_LOCK:
        if _original_load_providers_config is not None:
            try:
                from llm_engine import config_loader as engine_config_loader

                engine_config_loader.load_providers_config = _original_load_providers_config
            except Exception:  # pragma: no cover - defensive
                pass
        _original_load_providers_config = None


def create_engine_adapter(
    config: ProviderConfig | EngineConfigView,
    *,
    default_model: str | None = None,
) -> LLMProviderProtocol:
    """Create a fresh (uncached) llm_engine provider adapter.

    Accepts a ``ProviderConfig`` (wrapped in :class:`EngineConfigView` to
    unwrap the ``SecretStr`` key) or a pre-built ``EngineConfigView``.
    For connection reuse, prefer ``ProviderAdapterCache.get``.

    The engine catalog binding is installed here (before the provider object
    reads per-model flags in its constructor), so adapters behave the same no
    matter which directory ask-llm was started from.
    """
    bind_engine_providers_yml()
    view = config if isinstance(config, EngineConfigView) else EngineConfigView(config)
    # llm_engine is untyped; its adapter satisfies LLMProviderProtocol at runtime.
    return cast(LLMProviderProtocol, _create_provider_adapter(view, default_model=default_model))


def load_engine_providers_config() -> dict[str, Any]:
    """Load the engine's own providers catalog (base_url fallback source).

    Binding first keeps this lookup on ask_llm's runtime catalog rather than a
    ``providers.yml`` that happens to sit in the cwd. Returns ``{}`` when the
    engine loader is unavailable or fails; callers treat that as "no fallback
    data".
    """
    bind_engine_providers_yml()
    try:
        from llm_engine.config_loader import load_providers_config

        return load_providers_config() or {}
    except Exception as e:
        logger.warning(
            f"llm_engine providers catalog unavailable ({e}); no base_url fallback data."
        )
        return {}
