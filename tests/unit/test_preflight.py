"""Unit tests for the shared pre-run API health probe (utils/preflight)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch_models import ModelConfig
from ask_llm.core.models import AppConfig, ProviderConfig
from ask_llm.utils.preflight import (
    PREFLIGHT_TIMEOUT_SECONDS,
    check_provider_pair,
    check_provider_pairs,
)


def _make_config_manager(providers: dict[str, ProviderConfig]) -> ConfigManager:
    return ConfigManager(AppConfig(default_provider=next(iter(providers)), providers=providers))


def _manager() -> ConfigManager:
    return _make_config_manager(
        {
            "deepseek": ProviderConfig(
                api_provider="deepseek",
                api_key="sk-test",
                api_base="https://api.deepseek.com/v1",
                models=["deepseek-chat"],
                timeout=120.0,
            ),
            "ollama": ProviderConfig(
                api_provider="ollama",
                api_key="",
                api_base="http://localhost:11434/v1",
                models=["llama3"],
            ),
            "broken": ProviderConfig(
                api_provider="openai",
                api_key="${UNSET_VAR}",
                api_base="https://broken.example.com",
                models=["m1"],
            ),
        }
    )


class _FakeAdapter:
    def __init__(self, outcome: tuple[bool, str, float] | Exception):
        self.outcome = outcome
        self.received_config = None
        self.received_default_model = None

    def as_get(self):
        """Side-effect callable standing in for ProviderAdapterCache.get."""

        def _get(config, *, default_model=None):
            self.received_config = config
            self.received_default_model = default_model
            return self

        return _get

    def test_connection(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_healthy_pair_reports_ok_with_latency() -> None:
    adapter = _FakeAdapter((True, "Response: Hi", 0.42))
    with patch("ask_llm.utils.preflight.ProviderAdapterCache.get", return_value=adapter):
        outcome = check_provider_pair(_manager(), "deepseek", "deepseek-chat")

    assert outcome.ok is True
    assert outcome.latency == pytest.approx(0.42)
    assert outcome.message == "0.42s"


def test_failed_connection_reports_reason() -> None:
    adapter = _FakeAdapter((False, "Error code: 503 - Service is too busy", 0.1))
    with patch("ask_llm.utils.preflight.ProviderAdapterCache.get", return_value=adapter):
        outcome = check_provider_pair(_manager(), "deepseek", "deepseek-chat")

    assert outcome.ok is False
    assert outcome.latency is None
    assert "503" in outcome.message


def test_adapter_creation_error_is_swallowed() -> None:
    with patch(
        "ask_llm.utils.preflight.ProviderAdapterCache.get",
        side_effect=RuntimeError("no engine"),
    ):
        outcome = check_provider_pair(_manager(), "deepseek", "deepseek-chat")

    assert outcome.ok is False
    assert "Failed to create provider adapter" in outcome.message


def test_unresolved_key_fails_without_building_adapter() -> None:
    adapter = _FakeAdapter((True, "ok", 0.01))
    with patch("ask_llm.utils.preflight.ProviderAdapterCache.get", return_value=adapter) as get:
        outcome = check_provider_pair(_manager(), "broken", "m1")

    assert outcome.ok is False
    assert "API key not configured" in outcome.message
    assert "BROKEN" in outcome.message or "OPENAI" in outcome.message or "UNSET" in outcome.message
    get.assert_not_called()


def test_keyless_provider_is_still_probed() -> None:
    """ollama-style providers are exempt from the key check but still probed."""
    adapter = _FakeAdapter((True, "ok", 0.01))
    with patch("ask_llm.utils.preflight.ProviderAdapterCache.get", return_value=adapter) as get:
        outcome = check_provider_pair(_manager(), "ollama", "llama3")

    assert outcome.ok is True
    get.assert_called_once()


def test_unknown_provider_fails() -> None:
    outcome = check_provider_pair(_manager(), "nonexistent", "m")
    assert outcome.ok is False
    assert outcome.message == "Provider not found"


def test_probe_config_carries_short_timeout() -> None:
    """The probe must not inherit the run timeout (120s) — dead endpoints bound at 10s."""
    adapter = _FakeAdapter((True, "ok", 0.01))
    with patch("ask_llm.utils.preflight.ProviderAdapterCache.get", side_effect=adapter.as_get()):
        check_provider_pair(_manager(), "deepseek", "deepseek-chat")

    probe_config = adapter.received_config
    assert probe_config is not None
    assert float(probe_config.timeout) == PREFLIGHT_TIMEOUT_SECONDS


def test_sampling_overrides_applied() -> None:
    adapter = _FakeAdapter((True, "ok", 0.01))
    overrides = ModelConfig(
        provider="deepseek", model="deepseek-chat", temperature=0.3, max_tokens=777
    )
    with patch("ask_llm.utils.preflight.ProviderAdapterCache.get", side_effect=adapter.as_get()):
        check_provider_pair(_manager(), "deepseek", "deepseek-chat", sampling_overrides=overrides)

    probe_config = adapter.received_config
    assert probe_config.api_temperature == 0.3
    assert probe_config.max_tokens == 777


def test_pairs_dedup_preserving_order() -> None:
    adapter = _FakeAdapter((True, "ok", 0.01))
    pairs = [
        ("deepseek", "deepseek-chat"),
        ("ollama", "llama3"),
        ("deepseek", "deepseek-chat"),  # duplicate
    ]
    with patch("ask_llm.utils.preflight.ProviderAdapterCache.get", return_value=adapter):
        outcomes = check_provider_pairs(_manager(), pairs)

    assert [(o.provider, o.model) for o in outcomes] == [
        ("deepseek", "deepseek-chat"),
        ("ollama", "llama3"),
    ]


def test_pairs_empty_input() -> None:
    assert check_provider_pairs(_manager(), []) == []
