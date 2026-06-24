"""Tests for the global provider adapter cache."""

from unittest.mock import MagicMock, patch

import pytest

from ask_llm.utils.provider_cache import ProviderAdapterCache


def _make_config(**kwargs):
    config = MagicMock()
    config.api_provider = kwargs.get("api_provider", "openai")
    config.api_base = kwargs.get("api_base", "https://api.openai.com/v1")
    config.api_key = kwargs.get("api_key", "sk-test")
    config.models = kwargs.get("models", ["gpt-4"])
    config.api_temperature = kwargs.get("api_temperature", 0.7)
    config.api_top_p = kwargs.get("api_top_p")
    config.max_tokens = kwargs.get("max_tokens")
    config.timeout = kwargs.get("timeout", 60.0)
    return config


@pytest.fixture(autouse=True)
def _clear_cache():
    ProviderAdapterCache.clear()
    yield
    ProviderAdapterCache.clear()


def test_cache_returns_same_adapter_for_same_config():
    config = _make_config()
    with patch("ask_llm.utils.provider_cache.create_provider_adapter") as mock_create:
        adapter = MagicMock()
        mock_create.return_value = adapter
        first = ProviderAdapterCache.get(config, default_model="gpt-4")
        second = ProviderAdapterCache.get(config, default_model="gpt-4")

    assert first is second
    assert mock_create.call_count == 1


def test_cache_creates_separate_adapter_for_different_provider():
    config_a = _make_config(api_provider="openai")
    config_b = _make_config(api_provider="deepseek", api_base="https://api.deepseek.com/v1")
    with patch("ask_llm.utils.provider_cache.create_provider_adapter") as mock_create:
        mock_create.side_effect = [MagicMock(), MagicMock()]
        ProviderAdapterCache.get(config_a, default_model="gpt-4")
        ProviderAdapterCache.get(config_b, default_model="deepseek-chat")

    assert mock_create.call_count == 2


def test_cache_clear_resets_state():
    config = _make_config()
    with patch("ask_llm.utils.provider_cache.create_provider_adapter") as mock_create:
        mock_create.return_value = MagicMock()
        ProviderAdapterCache.get(config, default_model="gpt-4")
        ProviderAdapterCache.clear()
        ProviderAdapterCache.get(config, default_model="gpt-4")

    assert mock_create.call_count == 2


def test_cache_info_tracks_hits_and_misses():
    config = _make_config()
    with patch("ask_llm.utils.provider_cache.create_provider_adapter") as mock_create:
        mock_create.return_value = MagicMock()
        ProviderAdapterCache.get(config, default_model="gpt-4")
        ProviderAdapterCache.get(config, default_model="gpt-4")

    info = ProviderAdapterCache.info()
    assert info["misses"] == 1
    assert info["hits"] == 1
    assert info["currsize"] == 1


def test_cache_accepts_dict_config():
    config = {
        "api_provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "models": ["gpt-4"],
        "api_temperature": 0.7,
        "api_top_p": None,
        "max_tokens": None,
        "timeout": 60.0,
    }
    with patch("ask_llm.utils.provider_cache.create_provider_adapter") as mock_create:
        mock_create.return_value = MagicMock()
        ProviderAdapterCache.get(config, default_model="gpt-4")

    assert mock_create.call_count == 1
