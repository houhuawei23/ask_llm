"""Tests for the API-key pre-flight gate."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ask_llm.core.batch_models import BatchTask, ModelConfig
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.utils.api_key_gate import (
    UnresolvedAPIKeyError,
    ensure_resolved_provider_keys,
)


def _config_manager_with_key(provider: str, api_key: str) -> MagicMock:
    cm = MagicMock()
    cm.unified_config = None
    from ask_llm.core.models import ProviderConfig

    cm.get_provider_config.return_value = ProviderConfig(
        api_provider=provider,
        api_base=f"https://api.{provider}.com/v1",
        api_key=api_key,
        models=["m"],
    )
    return cm


def test_ensure_resolved_provider_keys_passes_with_real_key():
    cm = _config_manager_with_key("deepseek", "sk-real")
    # Should not raise.
    ensure_resolved_provider_keys(cm, ["deepseek"])


def test_ensure_resolved_provider_keys_rejects_placeholder():
    cm = _config_manager_with_key("deepseek", "${DEEPSEEK_API_KEY}")
    with pytest.raises(UnresolvedAPIKeyError, match="DEEPSEEK_API_KEY"):
        ensure_resolved_provider_keys(cm, ["deepseek"])


def test_ensure_resolved_provider_keys_rejects_empty():
    cm = _config_manager_with_key("deepseek", "")
    with pytest.raises(UnresolvedAPIKeyError):
        ensure_resolved_provider_keys(cm, ["deepseek"])


def test_ensure_resolved_provider_keys_skips_ollama():
    cm = _config_manager_with_key("ollama", "")
    # Ollama needs no key; must not raise.
    ensure_resolved_provider_keys(cm, ["ollama"])


def test_run_global_batch_tasks_fails_fast_on_unresolved_key():
    """run_global_batch_tasks must reject an unresolved key before fanning out."""
    cm = _config_manager_with_key("deepseek", "${DEEPSEEK_API_KEY}")
    tasks = [
        BatchTask(
            task_id=1,
            prompt="p",
            content="c",
            model_settings=ModelConfig(provider="deepseek", model="deepseek-chat"),
        )
    ]
    with pytest.raises(UnresolvedAPIKeyError):
        run_global_batch_tasks(tasks, cm, max_workers=2)
