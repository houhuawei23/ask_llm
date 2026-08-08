"""Provider adapter cache manager for global batch processing.

Extracted from :class:`GlobalBatchProcessor` to isolate provider/fallback-chain
construction from task scheduling. Builds a cache keyed by (provider, model[, timeout])
so worker threads reuse adapters instead of creating them per task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ask_llm.core.batch_models import BatchTask, ModelConfig
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.utils.provider_cache import ProviderAdapterCache

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager


class ProviderManager:
    """Build and cache LLM provider adapters for batch tasks."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager

    def build_provider_cache(
        self,
        tasks: list[BatchTask],
    ) -> dict[str, LLMProviderProtocol]:
        """Pre-build provider adapter cache for all unique (provider, model, timeout) combos.

        This eliminates per-task adapter creation overhead and avoids mutating the
        shared ConfigManager inside worker threads.
        """
        # Local import to avoid a circular dependency at module load time.
        from ask_llm.core.task_executor import paper_request_timeout_seconds

        cache: dict[str, LLMProviderProtocol] = {}
        seen: set[str] = set()

        def _add_config(mc: ModelConfig, task: BatchTask) -> None:
            key = f"{mc.provider}/{mc.model}"
            if task.task_kind == "paper_explain":
                timeout = paper_request_timeout_seconds()
                key += f" / timeout={timeout}"
            if key in seen:
                return
            seen.add(key)

            base_cfg = self.config_manager.config.get_provider_config(mc.provider)
            overrides: dict[str, Any] = {}
            if mc.temperature is not None:
                overrides["api_temperature"] = mc.temperature
            if mc.max_tokens is not None:
                overrides["max_tokens"] = mc.max_tokens
            if mc.top_p is not None:
                overrides["api_top_p"] = mc.top_p
            if task.task_kind == "paper_explain":
                overrides["timeout"] = float(paper_request_timeout_seconds())
            provider_cfg = base_cfg.model_copy(update=overrides)
            default_model = mc.model

            provider = ProviderAdapterCache.get(provider_cfg, default_model=default_model)
            cache[key] = provider

        for task in tasks:
            if not task.model_settings:
                continue
            _add_config(task.model_settings, task)
            for fallback_config in task.fallback_model_configs:
                _add_config(fallback_config, task)
        return cache
