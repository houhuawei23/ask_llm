"""Integration tests for the pre-run preflight gate (real preflight module).

Exercises ``cli_session.preflight_or_exit`` together with
``utils.preflight.check_provider_pairs`` through the real engine seam
(``ProviderAdapterCache.get`` patched in-process, per the AGENTS.md
fake-adapter pattern) — the unit tests patch the checker itself, so only
these catch wiring regressions (wrong pair set, wrong call order).
"""

from __future__ import annotations

import pytest
import typer
from unittest.mock import patch

from ask_llm.config.cli_session import preflight_or_exit
from ask_llm.config.manager import ConfigManager
from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.models import AppConfig, ProviderConfig
from ask_llm.utils.provider_cache import ProviderAdapterCache


class _FakeAdapter:
    """In-process adapter with a programmable test_connection outcome."""

    def __init__(self, outcome=(True, "ok", 0.01)):
        self.outcome = outcome
        self.test_calls = 0

    def test_connection(self):
        self.test_calls += 1
        return self.outcome


@pytest.fixture
def config_manager():
    app_config = AppConfig(
        default_provider="deepseek",
        providers={
            "deepseek": ProviderConfig(
                api_provider="deepseek",
                api_key="sk-test",
                api_base="https://api.deepseek.com/v1",
                models=["deepseek-chat"],
            ),
        },
    )
    return ConfigManager(app_config, UnifiedConfig())


def test_unhealthy_endpoint_non_tty_warns_and_continues(config_manager, monkeypatch):
    """User decision: non-interactive runs warn but keep going after a failed probe."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    adapter = _FakeAdapter(outcome=(False, "Error code: 503 - Service is too busy", 0.05))
    with patch.object(ProviderAdapterCache, "get", return_value=adapter) as get:
        # No typer.Exit: the run proceeds despite the failed probe.
        preflight_or_exit(
            config_manager,
            [("deepseek", "deepseek-chat")],
            command="trans",
        )

    assert get.call_count == 1
    assert adapter.test_calls == 1


def test_healthy_endpoint_gate_is_silent_passthrough(config_manager, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    adapter = _FakeAdapter()
    with patch.object(ProviderAdapterCache, "get", return_value=adapter):
        preflight_or_exit(
            config_manager,
            [("deepseek", "deepseek-chat")],
            command="trans",
        )
    assert adapter.test_calls == 1


def test_no_preflight_flag_never_touches_the_engine(config_manager, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with patch.object(ProviderAdapterCache, "get", side_effect=AssertionError("no probe")):
        preflight_or_exit(
            config_manager,
            [("deepseek", "deepseek-chat")],
            command="trans",
            skip=True,
        )


def test_tty_decline_exits_code_1(config_manager, monkeypatch):
    from ask_llm.utils.console import console

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(console, "_quiet", False)
    adapter = _FakeAdapter(outcome=(False, "dead endpoint", 0.01))
    with (
        patch.object(ProviderAdapterCache, "get", return_value=adapter),
        patch.object(console, "confirm", return_value=False),
        pytest.raises(typer.Exit) as excinfo,
    ):
        preflight_or_exit(
            config_manager,
            [("deepseek", "deepseek-chat")],
            command="trans",
        )
    assert excinfo.value.exit_code == 1


def test_config_disabled_gate_does_not_probe(config_manager, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    config_manager.unified_config.general.preflight_check = False
    with patch.object(ProviderAdapterCache, "get", side_effect=AssertionError("no probe")):
        preflight_or_exit(
            config_manager,
            [("deepseek", "deepseek-chat")],
            command="trans",
        )


def test_trans_dry_run_never_probes(sample_config_file, temp_dir, monkeypatch):
    """--dry-run must not touch the network: the preflight gate sits after it."""
    from typer.testing import CliRunner

    from ask_llm.cli import app

    monkeypatch.chdir(temp_dir)
    input_file = temp_dir / "paper.md"
    input_file.write_text("# Title\n\nSome content to estimate.\n", encoding="utf-8")

    def _fail(*args, **kwargs):
        raise AssertionError("preflight probe must not run during --dry-run")

    runner = CliRunner()
    with patch.object(ProviderAdapterCache, "get", side_effect=_fail):
        result = runner.invoke(
            app,
            [
                "trans",
                str(input_file),
                "--config",
                str(sample_config_file),
                "--dry-run",
                "--no-fallback",
            ],
        )
    assert result.exit_code == 0, result.output
