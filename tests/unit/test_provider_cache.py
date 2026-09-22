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


class TestAudit34Lifecycle:
    """Audit 3.4: key hashing, generation invalidation, close lifecycle."""

    def test_cache_key_does_not_contain_plaintext_key(self):
        """The plaintext API key must not sit in memory-resident cache tuples."""
        import ask_llm.utils.provider_cache as pc_mod

        secret = "sk-super-secret-plaintext"
        config = _make_config(api_key=secret)
        with patch("ask_llm.utils.provider_cache.create_engine_adapter"):
            ProviderAdapterCache.get(config, default_model="gpt-4")

        for key in pc_mod._adapters:
            for element in key:
                assert element != secret
        # And the digest form is what identifies the credential.
        import hashlib

        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        assert any(digest == element for key in pc_mod._adapters for element in key)

    def test_different_keys_never_share_an_adapter(self):
        """Same provider/base with rotated credentials -> distinct adapters."""
        config_a = _make_config(api_key="sk-key-a")
        config_b = _make_config(api_key="sk-key-b")
        with patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_create:
            mock_create.side_effect = [MagicMock(), MagicMock()]
            ProviderAdapterCache.get(config_a, default_model="gpt-4")
            ProviderAdapterCache.get(config_b, default_model="gpt-4")

        assert mock_create.call_count == 2

    def test_generation_bump_invalidates_adapters(self):
        """clear() must prevent reuse of adapters from the previous generation."""
        import ask_llm.utils.provider_cache as pc_mod

        config = _make_config()
        with patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            ProviderAdapterCache.get(config, default_model="gpt-4")
            generation_before = pc_mod._generation
            ProviderAdapterCache.clear()

            assert pc_mod._generation == generation_before + 1
            ProviderAdapterCache.get(config, default_model="gpt-4")

        assert mock_create.call_count == 2

    def test_close_releases_clients(self):
        """close() calls the adapter's close and empties the cache."""
        import ask_llm.utils.provider_cache as pc_mod

        config = _make_config()
        adapter = MagicMock()
        with patch("ask_llm.utils.provider_cache.create_engine_adapter", return_value=adapter):
            ProviderAdapterCache.get(config, default_model="gpt-4")

        ProviderAdapterCache.close()

        adapter.close.assert_called_once()
        assert not pc_mod._adapters

    def test_clear_closes_underlying_http_client(self):
        """Engine adapters without close() still release _provider._client."""
        import ask_llm.utils.provider_cache as pc_mod

        inner = MagicMock()  # llm_engine provider: no close(), holds _client
        inner.close = None  # explicit None: MagicMock auto-creation ignores del
        adapter = MagicMock()
        adapter.close = None
        adapter._client = None
        adapter._provider = inner

        with patch("ask_llm.utils.provider_cache.create_engine_adapter", return_value=adapter):
            ProviderAdapterCache.get(_make_config(), default_model="gpt-4")

        ProviderAdapterCache.clear()

        inner._client.close.assert_called_once()

    def test_eviction_closes_oldest_adapter(self):
        """LRU eviction beyond maxsize closes the evicted adapter."""
        import ask_llm.utils.provider_cache as pc_mod

        adapters = [MagicMock(), MagicMock()]
        with patch(
            "ask_llm.utils.provider_cache.create_engine_adapter", side_effect=adapters
        ) as mock_create:
            monkey_maxsize = 1
            original = pc_mod._CACHE_MAXSIZE
            pc_mod._CACHE_MAXSIZE = monkey_maxsize
            try:
                ProviderAdapterCache.get(_make_config(api_provider="a"), default_model="m")
                ProviderAdapterCache.get(_make_config(api_provider="b"), default_model="m")
            finally:
                pc_mod._CACHE_MAXSIZE = original

        assert mock_create.call_count == 2
        adapters[0].close.assert_called_once()  # oldest evicted and closed
        adapters[1].close.assert_not_called()  # still cached
