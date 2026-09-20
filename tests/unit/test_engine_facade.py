"""Unit tests for ask_llm.utils.engine_facade.

EngineConfigView SecretStr unwrapping / repr masking, and parameter routing of
``create_engine_adapter`` (the underlying llm_engine call is mocked).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from ask_llm.core.models import ProviderConfig
from ask_llm.utils.engine_facade import (
    EngineConfigView,
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


class TestEngineConfigView:
    def test_unwraps_secret_str_exactly_once(self):
        view = EngineConfigView(make_provider_config())

        # Plain string at the HTTP boundary, not a SecretStr.
        assert type(view.api_key) is str
        assert view.api_key == _SECRET
        assert view.api_provider == "openai"
        # Trailing slash already stripped by ProviderConfig's validator.
        assert view.api_base == "https://api.openai.com/v1"
        assert view.models == ["gpt-4o", "gpt-4o-mini"]
        assert (view.api_temperature, view.api_top_p, view.max_tokens, view.timeout) == (
            0.2,
            0.9,
            1024,
            30.0,
        )

    def test_repr_and_str_mask_the_key(self):
        view = EngineConfigView(make_provider_config())

        for text in (repr(view), str(view)):
            assert _SECRET not in text
            assert "***" in text
        assert "openai" in repr(view)

    def test_models_list_is_copied(self):
        pc = make_provider_config()
        view = EngineConfigView(pc)

        view.models.append("mutated")

        assert pc.models == ["gpt-4o", "gpt-4o-mini"]
        assert view.models == ["gpt-4o", "gpt-4o-mini", "mutated"]


class TestCreateEngineAdapter:
    def test_wraps_provider_config_and_routes_default_model(self):
        pc = make_provider_config()
        adapter = MagicMock()
        with patch("ask_llm.utils.engine_facade._create_provider_adapter") as mock_create:
            mock_create.return_value = adapter

            result = create_engine_adapter(pc, default_model="gpt-4o")

        assert result is adapter
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        (view,) = args
        assert isinstance(view, EngineConfigView)
        assert view.api_key == _SECRET  # unwrapped for the engine boundary
        assert kwargs == {"default_model": "gpt-4o"}

    def test_prebuilt_view_is_passed_through_unwrapped(self):
        view = EngineConfigView(make_provider_config())
        with patch("ask_llm.utils.engine_facade._create_provider_adapter") as mock_create:
            create_engine_adapter(view)

        # The exact same object is forwarded (no double wrapping).
        assert mock_create.call_args.args == (view,)


class TestLoadEngineProvidersConfig:
    def test_returns_engine_catalog(self):
        catalog = {"deepseek": {"base_url": "https://api.deepseek.com"}}
        with patch("llm_engine.config_loader.load_providers_config", return_value=catalog):
            assert load_engine_providers_config() == catalog

    def test_returns_empty_dict_on_loader_failure(self):
        with patch(
            "llm_engine.config_loader.load_providers_config",
            side_effect=RuntimeError("boom"),
        ):
            assert load_engine_providers_config() == {}
