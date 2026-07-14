"""Unit tests for configuration modules."""

import os
from pathlib import Path

import pytest
import yaml

from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.models import AppConfig, ProviderConfig


class TestConfigLoader:
    """Test ConfigLoader."""

    def test_load_valid_config(self, sample_config_file, sample_config_dict):
        """Test loading valid config file."""
        load_result = ConfigLoader.load(sample_config_file)
        config = load_result.app_config

        assert config.default_provider == "test_provider"
        assert "test_provider" in config.providers
        assert "another_provider" in config.providers
        assert load_result.unified_config.translation.threads == 5  # From sample_config

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load(temp_dir / "nonexistent.yml")

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML raises error."""
        config_path = temp_dir / "invalid.yml"
        config_path.write_text("not valid yaml: [")

        with pytest.raises(ValueError):
            ConfigLoader.load(config_path)

    def test_load_missing_providers(self, temp_dir):
        """Test loading config with empty providers raises error."""
        config_path = temp_dir / "bad_config.yml"
        with open(config_path, "w") as f:
            yaml.dump({"providers": {}}, f)  # Explicitly empty providers

        with pytest.raises(ValueError):
            ConfigLoader.load(config_path)

    def test_load_unsupported_format(self, temp_dir):
        """Test loading unsupported file format raises error."""
        config_path = temp_dir / "config.json"
        config_path.write_text('{"providers": {}}')

        with pytest.raises(ValueError) as exc_info:
            ConfigLoader.load(config_path)
        assert "Unsupported config file format" in str(exc_info.value)

    def test_env_var_override(self, sample_config_file):
        """Test ASK_LLM_* environment variables override config."""
        try:
            os.environ["ASK_LLM_TRANSLATION_TARGET_LANGUAGE"] = "en"
            load_result = ConfigLoader.load(sample_config_file)
            assert load_result.unified_config.translation.target_language == "en"
        finally:
            os.environ.pop("ASK_LLM_TRANSLATION_TARGET_LANGUAGE", None)

    def test_conflicting_env_overrides_warns_and_last_wins(self, sample_config_file):
        """P2.7: two env vars on the same key warn; the canonical one wins."""
        from unittest.mock import patch

        try:
            os.environ["ASK_LLM_TRANSLATION_THREADS"] = "3"
            os.environ["ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS"] = "7"
            with patch("ask_llm.config.loader.logger") as mock_logger:
                load_result = ConfigLoader.load(sample_config_file)
            # MAX_CONCURRENT_API_CALLS is later in ENV_TO_CONFIG order -> wins.
            assert load_result.unified_config.translation.max_concurrent_api_calls == 7
            # A conflict warning was emitted naming the winning var.
            warning_msgs = " ".join(
                str(c) for c in mock_logger.warning.call_args_list
            )
            assert "Conflicting env overrides" in warning_msgs
            assert "ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS" in warning_msgs
        finally:
            os.environ.pop("ASK_LLM_TRANSLATION_THREADS", None)
            os.environ.pop("ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS", None)

    def test_single_env_override_does_not_warn_conflict(self, sample_config_file):
        """P2.7: a single env var on a shared key does not trigger the conflict warning."""
        from unittest.mock import patch

        try:
            os.environ["ASK_LLM_TRANSLATION_THREADS"] = "4"
            with patch("ask_llm.config.loader.logger") as mock_logger:
                load_result = ConfigLoader.load(sample_config_file)
            assert load_result.unified_config.translation.max_concurrent_api_calls == 4
            warning_msgs = " ".join(
                str(c) for c in mock_logger.warning.call_args_list
            )
            assert "Conflicting env overrides" not in warning_msgs
        finally:
            os.environ.pop("ASK_LLM_TRANSLATION_THREADS", None)


class TestKimiProviderConfig:
    """Test Kimi provider configuration."""

    def test_kimi_provider_in_default_config(self):
        """Test that kimi provider is defined in default config."""
        from ask_llm.config.loader import ConfigLoader

        pkg_path = ConfigLoader._get_package_config_path()
        load_result = ConfigLoader.load(pkg_path)
        config = load_result.app_config

        # Check kimi (Kimi Code) provider exists
        assert "kimi-code" in config.providers, "kimi-code provider should be in default config"

        kimi_config = config.providers["kimi-code"]

        # Check required fields
        assert kimi_config.api_provider == "kimi-code"
        assert kimi_config.api_base == "https://api.kimi.com/coding/v1"
        assert "kimi-k2-0711-preview" in kimi_config.models

    def test_kimi_provider_config_structure(self):
        """Test kimi provider config has correct structure."""
        from ask_llm.config.loader import ConfigLoader

        pkg_path = ConfigLoader._get_package_config_path()
        load_result = ConfigLoader.load(pkg_path)
        config = load_result.app_config

        kimi_config = config.providers["kimi-code"]

        # Validate ProviderConfig structure
        assert isinstance(kimi_config, ProviderConfig)
        assert kimi_config.api_temperature == 0.7
        assert kimi_config.timeout == 120.0

    def test_kimi_provider_models(self):
        """Test kimi provider has expected models."""
        from ask_llm.config.loader import ConfigLoader

        pkg_path = ConfigLoader._get_package_config_path()
        load_result = ConfigLoader.load(pkg_path)
        config = load_result.app_config

        kimi_config = config.providers["kimi-code"]
        expected_models = [
            "kimi-k2-0711-preview",
            "kimi-k2-0711-preview-longcontext",
            "kimi-k2.5-preview",
        ]

        for model in expected_models:
            assert model in kimi_config.models, f"Model {model} should be in kimi-code provider"


class TestConfigManager:
    """Test ConfigManager."""

    def test_init(self, app_config):
        """Test initialization."""
        manager = ConfigManager(app_config)

        assert manager.config == app_config
        assert manager.current_provider_name == "test"

    def test_set_provider(self, app_config):
        """Test setting provider."""
        # Create config with multiple providers
        config = AppConfig(
            default_provider="provider1",
            default_model="model1",
            providers={
                "provider1": ProviderConfig(
                    api_provider="provider1",
                    api_key="key1",
                    api_base="https://api1.com",
                    models=["model1"],
                ),
                "provider2": ProviderConfig(
                    api_provider="provider2",
                    api_key="key2",
                    api_base="https://api2.com",
                    models=["model2"],
                ),
            },
        )

        manager = ConfigManager(config)
        manager.set_provider("provider2")

        assert manager.current_provider_name == "provider2"

    def test_set_invalid_provider(self, app_config):
        """Test setting invalid provider raises error."""
        manager = ConfigManager(app_config)

        with pytest.raises(ValueError):
            manager.set_provider("nonexistent")

    def test_get_provider_config(self, app_config):
        """Test getting provider config."""
        manager = ConfigManager(app_config)
        config = manager.get_provider_config()

        assert isinstance(config, ProviderConfig)
        assert config.api_provider == "test"

    def test_apply_overrides(self, app_config):
        """Test applying CLI overrides."""
        manager = ConfigManager(app_config)
        manager.apply_overrides(model="new-model", temperature=0.9)

        assert manager.get_model_override() == "new-model"
        config = manager.get_provider_config()
        assert config.api_temperature == 0.9

    def test_clear_overrides(self, app_config):
        """Test clearing overrides."""
        manager = ConfigManager(app_config)
        manager.apply_overrides(model="new-model")
        manager.clear_overrides()

        assert manager.get_model_override() is None
        assert manager.get_default_model() == "test-model"  # back to original

    def test_override_sources_tracking(self, app_config):
        """Test that override provenance is recorded for transparency."""
        manager = ConfigManager(app_config)
        manager.apply_overrides(model="new-model", temperature=0.9, api_key="secret")

        sources = manager.get_override_sources()
        assert sources["model"].startswith("CLI: new-model")
        assert sources["temperature"].startswith("CLI: 0.9")
        # api_key value must be masked in source label
        assert "secret" not in sources["api_key"]
        assert sources["api_key"].startswith("CLI: ***")

    def test_override_sources_custom_label(self, app_config):
        """Test custom source label is recorded."""
        manager = ConfigManager(app_config)
        manager.apply_overrides(model="env-model", source="ENV")

        sources = manager.get_override_sources()
        assert sources["model"] == "ENV: env-model"

    def test_set_provider_records_source(self, app_config):
        """Test that provider switch records its source."""
        config = AppConfig(
            default_provider="provider1",
            default_model="model1",
            providers={
                "provider1": ProviderConfig(
                    api_provider="provider1",
                    api_key="key1",
                    api_base="https://api1.com",
                    models=["model1"],
                ),
            },
        )
        manager = ConfigManager(config)
        manager.set_provider("provider1", source="CLI")

        sources = manager.get_override_sources()
        assert sources["provider"].startswith("CLI: provider1")

    def test_clear_overrides_clears_sources(self, app_config):
        """Test that clearing overrides also clears sources."""
        manager = ConfigManager(app_config)
        manager.apply_overrides(model="new-model")
        manager.clear_overrides()

        assert manager.get_override_sources() == {}

    def test_get_available_providers(self, app_config):
        """Test getting available providers."""
        manager = ConfigManager(app_config)
        providers = manager.get_available_providers()

        assert providers == ["test"]

    def test_get_available_models(self, app_config):
        """Test getting available models."""
        manager = ConfigManager(app_config)
        models = manager.get_available_models()

        assert "test-model" in models
