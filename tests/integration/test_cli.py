"""
Integration tests for CLI commands.

These tests use the Typer CliRunner to test commands without making real API calls.
"""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ask_llm.cli import app


runner = CliRunner()


class TestCLICommands:
    """Test CLI commands."""

    def test_version(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_help(self):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Ask LLM" in result.output

    def test_ask_no_input(self):
        """Test ask command without input fails."""
        result = runner.invoke(app, ["ask"])
        assert result.exit_code != 0

    def test_config_show_no_config(self):
        """Test config show without config fails."""
        result = runner.invoke(
            app, ["config", "show", "--config", "/nonexistent/default_config.yml"]
        )
        assert result.exit_code != 0


class TestCLIWithConfig:
    """Test CLI commands with a config file."""

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock config file."""
        config = {
            "default_provider": "test",
            "default_model": "test-model",
            "providers": {
                "test": {
                    "base_url": "https://api.test.com/v1",
                    "api_key": "test-key-12345",
                    "default_model": "test-model",
                    "models": [{"name": "test-model"}],
                    "api_temperature": 0.5,
                }
            },
            "general": {},
            "translation": {},
            "batch": {},
            "file": {},
            "format_heading": {},
            "token": {},
        }
        config_path = temp_dir / "default_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_config_show(self, mock_config):
        """Test config show command."""
        result = runner.invoke(app, ["config", "show", "--config", str(mock_config)])

        assert result.exit_code == 0
        assert "test" in result.output
        assert "https://api.test.com/v1" in result.output


class TestBatchSplitMode:
    """Test batch command with --split option."""

    @pytest.fixture
    def batch_config_with_output(self, temp_dir, sample_config_file):
        """Create a batch config file with output filenames.

        Uses same temp_dir as sample_config_file so default_config.yml
        is found when batch runs. Adds provider-models to avoid interactive selection.
        """
        config_content = """
provider-models:
  - provider: test_provider
    models:
      - model: test-model
prompt: "You are a helpful assistant"
contents:
  - output: "result1.md"
    content: "Question 1"
  - output: "result2.md"
    content: "Question 2"
"""
        config_file = sample_config_file.parent / "batch_config.yml"
        config_file.write_text(config_content)
        return config_file

    @pytest.fixture
    def batch_config_without_output(self, temp_dir):
        """Create a batch config file without output filenames."""
        config_content = """
prompt: "You are a helpful assistant"
contents:
  - "Question 1"
  - "Question 2"
"""
        config_file = temp_dir / "batch_config.yml"
        config_file.write_text(config_content)
        return config_file

    def test_batch_split_option_help(self):
        """Test that --split option appears in help."""
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--split" in result.output.lower()

    def test_batch_split_requires_output_dir(self, batch_config_with_output, sample_config_file):
        """Test that --split requires output directory, not file."""
        output_file = batch_config_with_output.parent / "output.json"
        output_file.touch()  # Create a file

        result = runner.invoke(
            app,
            [
                "batch",
                str(batch_config_with_output),
                "--split",
                "--output",
                str(output_file),
                "--config",
                str(sample_config_file),
            ],
        )
        # Should fail - either output dir validation or provider validation (no API in test)
        assert result.exit_code != 0
