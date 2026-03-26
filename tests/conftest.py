"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ask_llm.core.models import AppConfig, ProviderConfig, ChatHistory, ChatMessage, MessageRole


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary (default_config.yml format)."""
    return {
        "default_provider": "test_provider",
        "default_model": "test-model",
        "providers": {
            "test_provider": {
                "base_url": "https://api.test.com/v1",
                "api_key": "test-api-key-123",
                "default_model": "test-model",
                "models": [{"name": "test-model"}, {"name": "test-model-2"}],
                "api_temperature": 0.5,
                "api_top_p": 0.9,
            },
            "another_provider": {
                "base_url": "https://api.another.com/v1",
                "api_key": "another-api-key",
                "models": [{"name": "another-model"}],
                "api_temperature": 0.7,
            },
        },
        "general": {"default_prompt_template": "Test: {content}", "default_output_filename": "output.txt"},
        "translation": {"target_language": "zh", "threads": 5, "retries": 3, "max_chunk_size": 2000},
        "batch": {"threads": 5, "retries": 3},
        "file": {"chunk_size": 8192, "default_output_suffix": "_output"},
        "format_heading": {"batch_size": 80, "concurrency": 4},
        "text_splitter": {"max_chunk_size": 2000},
        "token": {"default_encoding": "cl100k_base"},
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict):
    """Create sample config file."""
    config_path = temp_dir / "default_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def provider_config():
    """Sample provider config."""
    return ProviderConfig(
        api_provider="test",
        api_key="test-key",
        api_base="https://api.test.com/v1",
        models=["test-model"],
        api_temperature=0.5,
    )


@pytest.fixture
def app_config(provider_config):
    """Sample app config."""
    return AppConfig(
        default_provider="test",
        default_model="test-model",
        providers={"test": provider_config}
    )


@pytest.fixture
def sample_chat_history():
    """Sample chat history."""
    history = ChatHistory(provider="test", model="test-model")
    history.add_message(MessageRole.SYSTEM, "You are helpful")
    history.add_message(MessageRole.USER, "Hello")
    history.add_message(MessageRole.ASSISTANT, "Hi there!")
    return history
