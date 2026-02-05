"""Tests for translation config loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ask_llm.utils.trans_config_loader import TransConfig, TransConfigLoader


class TestTransConfig:
    """Test TransConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TransConfig()
        assert config.target_language == "zh"
        assert config.source_language == "auto"
        assert config.style == "formal"
        assert config.threads == 5
        assert config.retries == 3
        assert config.max_chunk_size == 2000

    def test_custom_config(self):
        """Test custom configuration."""
        config = TransConfig(
            target_language="en",
            source_language="zh",
            style="casual",
            threads=10,
            retries=5,
            max_chunk_size=3000,
        )
        assert config.target_language == "en"
        assert config.source_language == "zh"
        assert config.style == "casual"
        assert config.threads == 10
        assert config.retries == 5
        assert config.max_chunk_size == 3000

    def test_validate_style(self):
        """Test style validation."""
        config = TransConfig(style="formal")
        assert config.style == "formal"

        config = TransConfig(style="casual")
        assert config.style == "casual"

        config = TransConfig(style="technical")
        assert config.style == "technical"

        # Invalid style should default to formal
        config = TransConfig(style="invalid")
        assert config.style == "formal"

    def test_validate_threads_range(self):
        """Test threads range validation."""
        # Valid range
        config = TransConfig(threads=1)
        assert config.threads == 1

        config = TransConfig(threads=50)
        assert config.threads == 50

        # Out of range should raise validation error
        with pytest.raises(Exception):
            TransConfig(threads=0)

        with pytest.raises(Exception):
            TransConfig(threads=51)


class TestTransConfigLoader:
    """Test TransConfigLoader."""

    def test_load_default_config(self):
        """Test loading default config when file doesn't exist."""
        config = TransConfigLoader.load("nonexistent.yml")
        assert config is None

        default_config = TransConfigLoader.get_default_config()
        assert isinstance(default_config, TransConfig)

    def test_load_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "target_language": "en",
            "source_language": "zh",
            "style": "casual",
            "threads": 10,
            "retries": 5,
            "max_chunk_size": 3000,
            "provider": "test-provider",
            "model": "test-model",
            "temperature": 0.8,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = TransConfigLoader.load(temp_path)
            assert config is not None
            assert config.target_language == "en"
            assert config.source_language == "zh"
            assert config.style == "casual"
            assert config.threads == 10
            assert config.retries == 5
            assert config.max_chunk_size == 3000
            assert config.provider == "test-provider"
            assert config.model == "test-model"
            assert config.temperature == 0.8
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                TransConfigLoader.load(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_partial_config(self):
        """Test loading partial configuration (some fields missing)."""
        config_data = {
            "target_language": "en",
            "threads": 10,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = TransConfigLoader.load(temp_path)
            assert config is not None
            assert config.target_language == "en"
            assert config.threads == 10
            # Should use defaults for missing fields
            assert config.source_language == "auto"
            assert config.style == "formal"
        finally:
            Path(temp_path).unlink()
