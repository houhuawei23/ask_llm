"""Provider fallback chain resolution.

Builds ordered lists of fallback :class:`ModelConfig` entries from the
application configuration so that batch/translation tasks can retry with
alternate providers or models when the primary fails.
"""

from __future__ import annotations

from ask_llm.core.batch_models import ModelConfig
from ask_llm.core.models import AppConfig


def build_fallback_chain(
    app_config: AppConfig,
    primary_config: ModelConfig,
) -> list[ModelConfig]:
    """Return fallback ``ModelConfig`` chain for a primary task config.

    The chain is sourced from the primary provider's ``fallback_to`` list in
    ``AppConfig``. If the provider is unknown or has no fallbacks, an empty
    list is returned.

    Args:
        app_config: Loaded application configuration.
        primary_config: The primary model configuration for the task.

    Returns:
        Ordered list of fallback model configurations.
    """
    provider_cfg = app_config.providers.get(primary_config.provider)
    if not provider_cfg or not provider_cfg.fallback_to:
        return []

    result: list[ModelConfig] = []
    for fb in provider_cfg.fallback_to:
        result.append(
            ModelConfig(
                provider=fb.provider,
                model=fb.model,
                temperature=fb.temperature
                if fb.temperature is not None
                else primary_config.temperature,
                top_p=fb.top_p if fb.top_p is not None else primary_config.top_p,
                max_tokens=fb.max_tokens
                if fb.max_tokens is not None
                else primary_config.max_tokens,
            )
        )
    return result
