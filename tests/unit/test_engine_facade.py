"""Unit tests for ask_llm.utils.engine_facade.

Facade delegation to the litellm-backed adapter, and the providers.yml
catalog fallback (success / failure paths). The adapter itself is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from ask_llm.core.models import ProviderConfig
from ask_llm.utils.engine_facade import (
    create_engine_adapter,
    load_engine_providers_config,
)

_SECRET = "sk-super-secret-123"


def make_provider_config() -> ProviderConfig:
    return ProviderConfig(
        api_provider="openai",
        api_key=SecretStr(_SECRET),
        api_base="https://api.openai.com/v1/",
        models=["gpt-4o", "gpt-4o-mini"],
        api_temperature=0.2,
        api_top_p=0.9,
        max_tokens=1024,
        timeout=30.0,
    )


class TestCreateEngineAdapter:
    def test_delegates_to_litellm_adapter_with_default_model(self):
        pc = make_provider_config()
        adapter = MagicMock()
        with patch(
            "ask_llm.utils.engine_facade.LiteLLMProviderAdapter", return_value=adapter
        ) as mock_cls:
            result = create_engine_adapter(pc, default_model="gpt-4o")

        assert result is adapter
        mock_cls.assert_called_once_with(pc, default_model="gpt-4o")

    def test_default_model_is_none_when_not_given(self):
        pc = make_provider_config()
        with patch("ask_llm.utils.engine_facade.LiteLLMProviderAdapter") as mock_cls:
            create_engine_adapter(pc)

        mock_cls.assert_called_once_with(pc, default_model=None)

    def test_no_engine_config_view_exported(self):
        # The SecretStr-unwrapping view died with the llm-engine boundary;
        # the litellm adapter takes the ProviderConfig directly.
        import ask_llm.utils.engine_facade as facade

        assert not hasattr(facade, "EngineConfigView")


class TestLoadEngineProvidersConfig:
    def test_returns_providers_catalog(self):
        catalog = {"providers": {"deepseek": {"base_url": "https://api.deepseek.com/v1"}}}
        with patch(
            "ask_llm.utils.engine_facade.load_first_providers_yml",
            return_value=(catalog, "/tmp/providers.yml"),
        ) as mock_load:
            assert load_engine_providers_config() == catalog

        # Runtime paths only: the fallback feeds credentials to base_url.
        assert mock_load.call_args.kwargs.get("paths") is not None

    def test_returns_empty_dict_when_no_catalog(self):
        with patch(
            "ask_llm.utils.engine_facade.load_first_providers_yml",
            return_value=(None, None),
        ):
            assert load_engine_providers_config() == {}

    def test_returns_empty_dict_on_loader_failure(self):
        with patch(
            "ask_llm.utils.engine_facade.load_first_providers_yml",
            side_effect=RuntimeError("boom"),
        ):
            assert load_engine_providers_config() == {}
