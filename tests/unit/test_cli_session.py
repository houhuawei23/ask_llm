"""Tests for cli_session bootstrap helpers and BatchTask kind coercion."""

import pytest
import typer

from ask_llm.config.cli_session import resolve_and_prepare
from ask_llm.config.manager import ConfigManager
from ask_llm.core.models import AppConfig, ProviderConfig


def _make_config_manager(default_provider: str = "deepseek") -> ConfigManager:
    app_config = AppConfig(
        default_provider=default_provider,
        providers={
            default_provider: ProviderConfig(
                api_provider=default_provider,
                api_key="test-key",
                api_base="https://api.example.com",
                models=["deepseek-chat"],
            ),
            "openai": ProviderConfig(
                api_provider="openai",
                api_key="test-key",
                api_base="https://api.openai.com",
                models=["gpt-4"],
            ),
        },
    )
    return ConfigManager(app_config)


def test_resolve_and_prepare_uses_cli_args() -> None:
    cm = _make_config_manager()
    provider, model = resolve_and_prepare(
        cm,
        cli_provider="openai",
        cli_model="gpt-4",
    )
    assert provider == "openai"
    assert model == "gpt-4"
    assert cm.get_model_override() == "gpt-4"


def test_resolve_and_prepare_applies_temperature_and_default_model() -> None:
    cm = _make_config_manager()
    provider, model = resolve_and_prepare(cm, temperature=0.3)
    assert provider == "deepseek"
    assert model == "deepseek-chat"
    assert cm.get_provider_config().api_temperature == 0.3


def test_resolve_and_prepare_exits_without_provider() -> None:
    app_config = AppConfig(
        default_provider="",
        providers={
            "deepseek": ProviderConfig(
                api_provider="deepseek",
                api_key="test-key",
                api_base="https://api.example.com",
                models=["deepseek-chat"],
            ),
        },
    )
    cm = ConfigManager(app_config)
    with pytest.raises(typer.Exit):
        resolve_and_prepare(cm)


def test_resolve_and_prepare_exits_without_model() -> None:
    app_config = AppConfig(
        default_provider="deepseek",
        providers={
            "deepseek": ProviderConfig(
                api_provider="deepseek",
                api_key="test-key",
                api_base="https://api.example.com",
                models=[],
            ),
        },
    )
    cm = ConfigManager(app_config)
    with pytest.raises(typer.Exit):
        resolve_and_prepare(cm)


def test_resolve_and_prepare_exits_for_unknown_provider() -> None:
    cm = _make_config_manager()
    with pytest.raises(typer.Exit):
        resolve_and_prepare(cm, cli_provider="nope")


def test_config_manager_falls_back_when_default_provider_invalid() -> None:
    app_config = AppConfig(
        default_provider="ghost",
        providers={
            "deepseek": ProviderConfig(
                api_provider="deepseek",
                api_key="test-key",
                api_base="https://api.example.com",
                models=["deepseek-chat"],
            ),
        },
    )
    cm = ConfigManager(app_config)
    assert cm.current_provider_name == "deepseek"
