"""Unit tests for data models."""

import pytest
from pydantic import ValidationError

from ask_llm.core.models import (
    ProviderConfig,
    AppConfig,
    ChatMessage,
    ChatHistory,
    MessageRole,
    RequestMetadata,
)


class TestProviderConfig:
    """Test ProviderConfig model."""

    def test_valid_config(self):
        """Test creating valid config."""
        config = ProviderConfig(
            api_provider="test",
            api_key="valid-key",
            api_base="https://api.test.com",
            models=["test-model"],
        )
        assert config.api_provider == "test"
        assert config.api_temperature == 0.7  # default

    def test_invalid_api_key_placeholder(self):
        """Test placeholder API key is allowed (validation moved to upstream gate)."""
        config = ProviderConfig(
            api_provider="test",
            api_key="your-api-key-here",
            api_base="https://api.test.com",
            models=["test-model"],
        )
        assert config.api_key == "your-api-key-here"

    def test_invalid_api_key_empty(self):
        """Test empty API key is allowed (ollama uses no key; gate enforces for others)."""
        config = ProviderConfig(
            api_provider="test",
            api_key="",
            api_base="https://api.test.com",
            models=["test-model"],
        )
        assert config.api_key == ""

    def test_unresolved_env_var_placeholder_warns(self):
        """An unresolved ${VAR} placeholder warns loudly (not silent), still loads.

        Hard-rejection happens at the run-boundary gate, not at load time, so that
        config tooling can load providers.yml with env vars unset.
        """
        from unittest.mock import patch

        with patch("ask_llm.core.models.logger") as mock_logger:
            config = ProviderConfig(
                api_provider="deepseek",
                api_key="${DEEPSEEK_API_KEY}",
                api_base="https://api.test.com",
                models=["test-model"],
            )
        assert config.api_key == "${DEEPSEEK_API_KEY}"
        mock_logger.warning.assert_called_once()
        message = mock_logger.warning.call_args[0][0].lower()
        assert "unresolved" in message
        assert "deepseek_api_key" in message

    def test_partial_env_var_placeholder_warns(self):
        """A key containing an embedded ${VAR} also warns."""
        from unittest.mock import patch

        with patch("ask_llm.core.models.logger") as mock_logger:
            config = ProviderConfig(
                api_provider="test",
                api_key="prefix-${MISSING_VAR}-suffix",
                api_base="https://api.test.com",
                models=["test-model"],
            )
        assert config.api_key == "prefix-${MISSING_VAR}-suffix"
        mock_logger.warning.assert_called_once()
        assert "unresolved" in mock_logger.warning.call_args[0][0].lower()

    def test_invalid_api_base(self):
        """Test validation rejects invalid API base URL."""
        with pytest.raises(ValidationError):
            ProviderConfig(
                api_provider="test",
                api_key="valid-key",
                api_base="not-a-url",
                models=["test-model"],
            )

    def test_temperature_range(self):
        """Test temperature must be in valid range."""
        # Valid values
        ProviderConfig(
            api_provider="test",
            api_key="key",
            api_base="https://api.test.com",
            models=["model"],
            api_temperature=0.0,
        )
        ProviderConfig(
            api_provider="test",
            api_key="key",
            api_base="https://api.test.com",
            models=["model"],
            api_temperature=2.0,
        )

        # Invalid values
        with pytest.raises(ValidationError):
            ProviderConfig(
                api_provider="test",
                api_key="key",
                api_base="https://api.test.com",
                models=["model"],
                api_temperature=-0.1,
            )
        with pytest.raises(ValidationError):
            ProviderConfig(
                api_provider="test",
                api_key="key",
                api_base="https://api.test.com",
                models=["model"],
                api_temperature=2.1,
            )


class TestAppConfig:
    """Test AppConfig model."""

    def test_valid_config(self, app_config):
        """Test valid app config."""
        assert app_config.default_provider == "test"
        assert "test" in app_config.providers

    def test_empty_providers(self):
        """Test validation rejects empty providers."""
        with pytest.raises(ValidationError):
            AppConfig(default_provider="test", providers={})

    def test_get_provider_config(self, app_config):
        """Test getting provider config."""
        config = app_config.get_provider_config("test")
        assert config.api_provider == "test"

    def test_get_provider_config_default(self, app_config):
        """Test getting default provider config."""
        config = app_config.get_provider_config()
        assert config.api_provider == "test"

    def test_get_provider_config_missing(self, app_config):
        """Test getting non-existent provider raises error."""
        with pytest.raises(ValueError):
            app_config.get_provider_config("nonexistent")


class TestChatMessage:
    """Test ChatMessage model."""

    def test_create_message(self):
        """Test creating a message."""
        msg = ChatMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.timestamp is not None


class TestChatHistory:
    """Test ChatHistory model."""

    def test_create_history(self):
        """Test creating history."""
        history = ChatHistory()
        assert len(history.messages) == 0

    def test_add_message(self, sample_chat_history):
        """Test adding messages."""
        initial_count = len(sample_chat_history.messages)
        sample_chat_history.add_message(MessageRole.USER, "New message")
        assert len(sample_chat_history.messages) == initial_count + 1

    def test_get_messages(self, sample_chat_history):
        """Test getting messages as dicts."""
        messages = sample_chat_history.get_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    def test_get_messages_no_system(self, sample_chat_history):
        """Test getting messages without system."""
        messages = sample_chat_history.get_messages(include_system=False)
        assert len(messages) == 2
        assert all(m["role"] != "system" for m in messages)

    def test_clear(self, sample_chat_history):
        """Test clearing history."""
        sample_chat_history.clear()
        assert len(sample_chat_history.messages) == 0

    def test_clear_keep_system(self, sample_chat_history):
        """Test clearing history while keeping system prompt."""
        sample_chat_history.clear(keep_system=True)
        assert len(sample_chat_history.messages) == 1
        assert sample_chat_history.messages[0].role == MessageRole.SYSTEM


class TestRequestMetadata:
    """Test RequestMetadata model."""

    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = RequestMetadata(
            provider="test",
            model="model",
            temperature=0.5,
            latency=1.23,
        )
        assert metadata.provider == "test"
        assert metadata.latency == 1.23

    def test_format_metadata(self):
        """Test metadata formatting."""
        metadata = RequestMetadata(
            provider="test",
            model="model",
            temperature=0.5,
            input_words=10,
            input_tokens=15,
            output_words=20,
            output_tokens=25,
            latency=1.23,
        )
        formatted = metadata.format()
        assert "test" in formatted
        assert "model" in formatted
        assert "1.23s" in formatted
