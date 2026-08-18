"""Tests for the global provider adapter cache."""

from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.models import ProviderConfig
from ask_llm.utils.provider_cache import ProviderAdapterCache


def _make_config(**kwargs) -> ProviderConfig:
    return ProviderConfig(
        api_provider=kwargs.get("api_provider", "openai"),
        api_base=kwargs.get("api_base", "https://api.openai.com/v1"),
        api_key=kwargs.get("api_key", "sk-test"),
        models=kwargs.get("models", ["gpt-4"]),
        api_temperature=kwargs.get("api_temperature", 0.7),
        api_top_p=kwargs.get("api_top_p"),
        max_tokens=kwargs.get("max_tokens"),
        timeout=kwargs.get("timeout", 60.0),
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    ProviderAdapterCache.clear()
    yield
    ProviderAdapterCache.clear()


def test_cache_returns_same_adapter_for_same_config():
    config = _make_config()
    with patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_create:
        adapter = MagicMock()
        mock_create.return_value = adapter
        first = ProviderAdapterCache.get(config, default_model="gpt-4")
        second = ProviderAdapterCache.get(config, default_model="gpt-4")

    assert first is second
    assert mock_create.call_count == 1


def test_cache_creates_separate_adapter_for_different_provider():
    config_a = _make_config(api_provider="openai")
    config_b = _make_config(api_provider="deepseek", api_base="https://api.deepseek.com/v1")
    with patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_create:
        mock_create.side_effect = [MagicMock(), MagicMock()]
        ProviderAdapterCache.get(config_a, default_model="gpt-4")
        ProviderAdapterCache.get(config_b, default_model="deepseek-chat")

    assert mock_create.call_count == 2


def test_cache_clear_resets_state():
    config = _make_config()
    with patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_create:
        mock_create.return_value = MagicMock()
        ProviderAdapterCache.get(config, default_model="gpt-4")
        ProviderAdapterCache.clear()
        ProviderAdapterCache.get(config, default_model="gpt-4")

    assert mock_create.call_count == 2


def test_invalid_config_type_rejected():
    """Non-ProviderConfig inputs raise TypeError."""
    with pytest.raises(TypeError):
        ProviderAdapterCache.get("not-a-config", default_model="gpt-4")  # type: ignore[arg-type]
