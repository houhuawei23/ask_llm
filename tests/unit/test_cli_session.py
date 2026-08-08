"""Tests for cli_session bootstrap helpers and BatchTask kind coercion."""

import pytest
import typer

from ask_llm.config.cli_session import resolve_provider_and_model_or_exit
from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import BatchTask, ModelConfig
from ask_llm.core.models import AppConfig, ProviderConfig
from ask_llm.core.tasks.builders import build_paper_explain_task


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


def test_resolve_provider_and_model_uses_cli_args() -> None:
    cm = _make_config_manager()
    provider, model = resolve_provider_and_model_or_exit(
        cm,
        cli_provider="openai",
        cli_model="gpt-4",
    )
    assert provider == "openai"
    assert model == "gpt-4"


def test_resolve_provider_and_model_falls_back_to_config() -> None:
    cm = _make_config_manager()
    provider, model = resolve_provider_and_model_or_exit(cm)
    assert provider == "deepseek"
    assert model == "deepseek-chat"


def test_resolve_provider_and_model_exits_without_provider() -> None:
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
        resolve_provider_and_model_or_exit(cm)


def test_resolve_provider_and_model_exits_without_model() -> None:
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
        resolve_provider_and_model_or_exit(cm)


def test_batch_task_legacy_paper_mode_sets_kind() -> None:
    mc = ModelConfig(provider="x", model="m")
    t = BatchTask(
        task_id=0,
        prompt="p",
        content="",
        model_settings=mc,
        paper_mode=True,
    )
    assert t.task_kind == "paper_explain"


def test_build_paper_explain_task() -> None:
    mc = ModelConfig(provider="p", model="m", max_tokens=100)
    t = build_paper_explain_task(
        1,
        "full prompt",
        model_settings=mc,
        output_filename="paper:full",
        return_reasoning=True,
    )
    assert t.task_kind == "paper_explain"
    assert t.return_reasoning is True
    assert t.content == ""
