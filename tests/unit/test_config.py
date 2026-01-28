"""Unit tests for configuration modules."""

import json
from pathlib import Path

import pytest

from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.models import AppConfig, ProviderConfig


class TestConfigLoader:
    """Test ConfigLoader."""
    
    def test_load_valid_config(self, sample_config_file, sample_config_dict):
        """Test loading valid config file."""
        config = ConfigLoader.load(sample_config_file)
        
        assert isinstance(config, AppConfig)
        assert config.default_provider == "test_provider"
        assert len(config.providers) == 2
        assert "test_provider" in config.providers
        assert "another_provider" in config.providers
    
    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load(temp_dir / "nonexistent.json")
    
    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON raises error."""
        config_path = temp_dir / "invalid.json"
        config_path.write_text("not valid json")
        
        with pytest.raises(ValueError):
            ConfigLoader.load(config_path)
    
    def test_load_missing_providers(self, temp_dir):
        """Test loading config without providers raises error."""
        config_path = temp_dir / "bad_config.json"
        with open(config_path, "w") as f:
            json.dump({"default_provider": "test"}, f)
        
        with pytest.raises(ValueError):
            ConfigLoader.load(config_path)
    
    def test_create_example_config(self, temp_dir):
        """Test creating example config."""
        config_path = temp_dir / "example.json"
        ConfigLoader.create_example_config(config_path)
        
        assert config_path.exists()
        
        with open(config_path) as f:
            data = json.load(f)
        
        assert "default_provider" in data
        assert "providers" in data
        assert "deepseek" in data["providers"]


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
            }
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
