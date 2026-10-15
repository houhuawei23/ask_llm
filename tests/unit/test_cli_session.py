"""Tests for cli_session bootstrap helpers and BatchTask kind coercion."""

from unittest.mock import patch

import pytest
import typer

from ask_llm.config.cli_session import gate_api_key_or_exit, resolve_and_prepare
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


def test_gate_api_key_or_exit_passes_with_resolved_key() -> None:
    gate_api_key_or_exit(_make_config_manager(), "deepseek")


def test_gate_api_key_or_exit_exits_non_interactive_without_key(monkeypatch) -> None:
    """Missing key + non-interactive stdin must exit 1 with the env-var hint."""
    import typer

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    app_config = AppConfig(
        default_provider="deepseek",
        providers={
            "deepseek": ProviderConfig(
                api_provider="deepseek",
                api_key="${DEEPSEEK_API_KEY}",
                api_base="https://api.example.com",
                models=["deepseek-chat"],
            ),
        },
    )
    cm = ConfigManager(app_config)
    with pytest.raises(typer.Exit) as excinfo:
        gate_api_key_or_exit(cm, "deepseek")
    assert excinfo.value.exit_code == 1


# --- preflight_or_exit gate -------------------------------------------------


def _preflight_cm() -> ConfigManager:
    """Config manager whose provider passes the key gate (probe patched below)."""
    app_config = AppConfig(
        default_provider="deepseek",
        providers={
            "deepseek": ProviderConfig(
                api_provider="deepseek",
                api_key="test-key",
                api_base="https://api.example.com",
                models=["deepseek-chat"],
            ),
        },
    )
    from ask_llm.config.unified_config import UnifiedConfig

    return ConfigManager(app_config, UnifiedConfig())


def _outcome(provider: str, model: str, ok: bool, message: str = "x"):
    from ask_llm.utils.preflight import PreflightOutcome

    return PreflightOutcome(provider, model, ok, 0.01 if ok else None, message)


def test_preflight_or_exit_all_healthy_returns(monkeypatch) -> None:
    from ask_llm.config.cli_session import preflight_or_exit

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with patch("ask_llm.config.cli_session.check_provider_pairs") as check:
        check.return_value = [_outcome("deepseek", "deepseek-chat", True)]
        preflight_or_exit(_preflight_cm(), [("deepseek", "deepseek-chat")], command="trans")

    check.assert_called_once()


def test_preflight_or_exit_non_tty_failure_warns_and_continues(monkeypatch) -> None:
    """User decision: non-interactive environments warn and continue (no exit)."""
    from ask_llm.config.cli_session import preflight_or_exit

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with patch("ask_llm.config.cli_session.check_provider_pairs") as check:
        check.return_value = [
            _outcome("deepseek", "deepseek-chat", False, "503 Service is too busy")
        ]
        # Must NOT raise typer.Exit.
        preflight_or_exit(_preflight_cm(), [("deepseek", "deepseek-chat")], command="trans")

    check.assert_called_once()


def test_preflight_or_exit_tty_decline_exits(monkeypatch) -> None:
    from ask_llm.config.cli_session import preflight_or_exit
    from ask_llm.utils.console import console

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(console, "_quiet", False)
    with (
        patch("ask_llm.config.cli_session.check_provider_pairs") as check,
        patch.object(console, "confirm", return_value=False),
    ):
        check.return_value = [_outcome("deepseek", "deepseek-chat", False, "dead")]
        with pytest.raises(typer.Exit) as excinfo:
            preflight_or_exit(_preflight_cm(), [("deepseek", "deepseek-chat")], command="trans")

    assert excinfo.value.exit_code == 1


def test_preflight_or_exit_tty_accept_continues(monkeypatch) -> None:
    from ask_llm.config.cli_session import preflight_or_exit
    from ask_llm.utils.console import console

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(console, "_quiet", False)
    with (
        patch("ask_llm.config.cli_session.check_provider_pairs") as check,
        patch.object(console, "confirm", return_value=True),
    ):
        check.return_value = [_outcome("deepseek", "deepseek-chat", False, "dead")]
        preflight_or_exit(_preflight_cm(), [("deepseek", "deepseek-chat")], command="trans")

    check.assert_called_once()


def test_preflight_or_exit_skip_flag_bypasses_probe(monkeypatch) -> None:
    from ask_llm.config.cli_session import preflight_or_exit

    with patch("ask_llm.config.cli_session.check_provider_pairs") as check:
        preflight_or_exit(
            _preflight_cm(), [("deepseek", "deepseek-chat")], command="trans", skip=True
        )
    check.assert_not_called()


def test_preflight_or_exit_config_disabled_bypasses_probe(monkeypatch) -> None:
    from ask_llm.config.cli_session import preflight_or_exit

    cm = _preflight_cm()
    cm.unified_config.general.preflight_check = False
    with patch("ask_llm.config.cli_session.check_provider_pairs") as check:
        preflight_or_exit(cm, [("deepseek", "deepseek-chat")], command="trans")
    check.assert_not_called()


def test_preflight_or_exit_empty_pairs_no_probe() -> None:
    from ask_llm.config.cli_session import preflight_or_exit

    with patch("ask_llm.config.cli_session.check_provider_pairs") as check:
        preflight_or_exit(_preflight_cm(), [], command="trans")
    check.assert_not_called()
