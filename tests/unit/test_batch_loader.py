"""Unit tests for batch configuration loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ask_llm.utils.batch_loader import BatchConfigLoader


class TestBatchConfigLoader:
    """Test BatchConfigLoader."""

    def test_load_prompt_contents_format(self, temp_dir):
        """Test loading prompt-contents format."""
        config_content = """
provider-models:
  - provider: deepseek
    models:
      - model: deepseek-chat
      - model: deepseek-reasoner
        temperature: 1.0
        top_p: 0.9
prompt: "You are a helpful assistant"
contents:
  - "Content 1"
  - "Content 2"
  - "Content 3"
"""
        config_file = temp_dir / "test_config.yml"
        config_file.write_text(config_content)

        result = BatchConfigLoader.load(str(config_file))

        assert result["mode"] == "prompt-contents"
        assert len(result["tasks"]) == 3
        assert len(result["provider_models"]) == 2

        # Check tasks
        assert result["tasks"][0].task_id == 1
        assert result["tasks"][0].prompt == "You are a helpful assistant"
        assert result["tasks"][0].content == "Content 1"

        # Check model configs
        assert result["provider_models"][0].provider == "deepseek"
        assert result["provider_models"][0].model == "deepseek-chat"
        assert result["provider_models"][1].model == "deepseek-reasoner"
        assert result["provider_models"][1].temperature == 1.0
        assert result["provider_models"][1].top_p == 0.9

    def test_load_prompt_contents_without_models(self, temp_dir):
        """Test loading prompt-contents format without provider-models."""
        config_content = """
prompt: "You are a helpful assistant"
contents:
  - "Content 1"
  - "Content 2"
"""
        config_file = temp_dir / "test_config.yml"
        config_file.write_text(config_content)

        result = BatchConfigLoader.load(str(config_file))

        assert result["mode"] == "prompt-contents"
        assert len(result["tasks"]) == 2
        assert result["provider_models"] == []

    def test_load_prompt_content_pairs_format(self, temp_dir):
        """Test loading prompt-content-pairs format."""
        config_content = """---
prompt: "Prompt 1"
content: "Content 1"
---
prompt: "Prompt 2"
content: "Content 2"
"""
        config_file = temp_dir / "test_config.yml"
        config_file.write_text(config_content)

        result = BatchConfigLoader.load(str(config_file))

        assert result["mode"] == "prompt-content-pairs"
        assert len(result["tasks"]) == 2

        assert result["tasks"][0].task_id == 1
        assert result["tasks"][0].prompt == "Prompt 1"
        assert result["tasks"][0].content == "Content 1"

        assert result["tasks"][1].task_id == 2
        assert result["tasks"][1].prompt == "Prompt 2"
        assert result["tasks"][1].content == "Content 2"

    def test_load_prompt_content_pairs_with_provider_models(self, temp_dir):
        """Test loading prompt-content-pairs format with provider-models in first doc."""
        config_content = """---
provider-models:
  - provider: test
    models:
      - model: test-model
---
prompt: "Prompt 1"
content: "Content 1"
"""
        config_file = temp_dir / "test_config.yml"
        config_file.write_text(config_content)

        result = BatchConfigLoader.load(str(config_file))

        assert result["mode"] == "prompt-content-pairs"
        assert len(result["tasks"]) == 1
        assert len(result["provider_models"]) == 1

    def test_load_invalid_file(self, temp_dir):
        """Test loading non-existent file."""
        config_file = temp_dir / "nonexistent.yml"

        with pytest.raises(FileNotFoundError):
            BatchConfigLoader.load(str(config_file))

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML."""
        config_content = "invalid: yaml: content: ["
        config_file = temp_dir / "invalid.yml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="Invalid YAML"):
            BatchConfigLoader.load(str(config_file))

    def test_load_missing_prompt(self, temp_dir):
        """Test loading config with missing prompt."""
        config_content = """
contents:
  - "Content 1"
"""
        config_file = temp_dir / "missing_prompt.yml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="Missing required field"):
            BatchConfigLoader.load(str(config_file))

    def test_load_missing_contents(self, temp_dir):
        """Test loading config with missing contents."""
        config_content = """
prompt: "Test prompt"
"""
        config_file = temp_dir / "missing_contents.yml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="Missing required field"):
            BatchConfigLoader.load(str(config_file))

    def test_load_empty_contents(self, temp_dir):
        """Test loading config with empty contents."""
        config_content = """
prompt: "Test prompt"
contents: []
"""
        config_file = temp_dir / "empty_contents.yml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="cannot be empty"):
            BatchConfigLoader.load(str(config_file))

    def test_parse_provider_models_string_format(self, temp_dir):
        """Test parsing provider-models with string model names."""
        config_content = """
provider-models:
  - provider: test
    models:
      - test-model-1
      - test-model-2
prompt: "Test"
contents:
  - "Content"
"""
        config_file = temp_dir / "string_models.yml"
        config_file.write_text(config_content)

        result = BatchConfigLoader.load(str(config_file))

        assert len(result["provider_models"]) == 2
        assert result["provider_models"][0].model == "test-model-1"
        assert result["provider_models"][1].model == "test-model-2"
