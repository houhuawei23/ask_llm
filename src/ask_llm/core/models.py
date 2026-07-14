"""Data models for Ask LLM using Pydantic."""

from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single chat message."""

    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime | None = Field(
        default_factory=datetime.now, description="Message timestamp"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                    "timestamp": "2024-01-15T10:30:00",
                }
            ]
        }
    }


class ChatHistory(BaseModel):
    """Chat conversation history."""

    messages: list[ChatMessage] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the history."""
        self.messages.append(ChatMessage(role=role, content=content))

    def get_messages(self, include_system: bool = True) -> list[dict[str, str]]:
        """Get messages as dictionaries for API calls."""
        msgs = self.messages
        if not include_system:
            msgs = [m for m in msgs if m.role != MessageRole.SYSTEM]
        return [{"role": m.role.value, "content": m.content} for m in msgs]

    def clear(self, keep_system: bool = False) -> None:
        """Clear conversation history."""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        else:
            self.messages.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                }
                for m in self.messages
            ],
        }


class FallbackConfig(BaseModel):
    """Fallback provider/model configuration."""

    provider: str = Field(..., description="Fallback provider name")
    model: str = Field(..., description="Fallback model name")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_provider: str = Field(..., description="Provider identifier")
    api_key: str = Field(default="", description="API key for authentication")
    api_base: str = Field(..., description="API base URL")
    models: list[str] = Field(default_factory=list, description="Available models")
    api_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    api_top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    max_tokens: int | None = Field(default=None, gt=0, description="Maximum tokens to generate")
    timeout: float = Field(default=60.0, gt=0, description="API timeout in seconds")
    fallback_to: list[FallbackConfig] = Field(
        default_factory=list, description="Ordered fallback provider/model chain"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str, info) -> str:
        """Validate API key: warn loudly if empty or carries an unresolved placeholder.

        ``resolve_env_vars`` leaves a literal ``${VAR}`` in place when the referenced
        environment variable is unset (it only logs at DEBUG). Such a placeholder
        must never ship as a real key. We cannot hard-reject at load time because
        config tooling (``config show``/``config test``) and tests legitimately load
        ``providers.yml`` with env vars unset; instead we warn loudly here so the
        problem is never silent, and the run-boundary gate
        (``ask_llm.utils.api_key_gate``) hard-blocks before any network call.
        """
        provider_name = info.data.get("api_provider", "unknown")
        if "${" in v and "}" in v:
            logger.warning(
                f"Provider '{provider_name}' API key has unresolved ${{...}} "
                f"placeholder: {v!r}. Set ASK_LLM_{provider_name.upper()}_API_KEY "
                f"(or the referenced variable) before running; the API-key gate "
                f"will block calls until it resolves."
            )
        elif not v or v.strip() == "":
            logger.warning(
                f"Provider '{provider_name}' has empty API key. "
                f"Set ASK_LLM_{provider_name.upper()}_API_KEY environment variable "
                f"or configure in default_config.yml"
            )
        return v

    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, v: str) -> str:
        """Validate API base URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("API base URL must start with http:// or https://")
        return v.rstrip("/")


class AppConfig(BaseModel):
    """Application configuration."""

    default_provider: str = Field(..., description="Default provider name")
    default_model: str | None = Field(default=None, description="Default model name")
    providers: dict[str, ProviderConfig] = Field(..., description="Provider configurations")

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, v: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        """Validate providers dictionary is not empty."""
        if not v:
            raise ValueError("At least one provider must be configured")
        return v

    def get_provider_config(self, name: str | None = None) -> ProviderConfig:
        """Get configuration for a provider."""
        provider_name = name or self.default_provider
        if provider_name not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(f"Provider '{provider_name}' not found. Available: {available}")
        return self.providers[provider_name]


class RequestMetadata(BaseModel):
    """Metadata for an API request."""

    provider: str = Field(..., description="Provider name")
    model: str = Field(..., description="Model name")
    temperature: float = Field(..., description="Temperature used")
    input_words: int = Field(default=0, description="Input word count")
    input_tokens: int = Field(default=0, description="Input token count")
    output_words: int = Field(default=0, description="Output word count")
    output_tokens: int = Field(default=0, description="Output token count")
    latency: float = Field(..., description="Request latency in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_execution(
        cls,
        *,
        provider_name: str,
        model: str,
        temperature: float | None,
        default_temperature: float,
        input_stats: dict[str, int],
        output_words: int,
        output_tokens: int,
        latency: float,
    ) -> "RequestMetadata":
        """Build metadata for a completed request from its execution stats.

        Centralizes the temperature resolution (per-request override falling
        back to the provider default) that was previously triplicated across
        ``batch_processor`` and ``processor`` — the same code path that caused
        the v2.15.1 adapter dict-vs-object crash (B8 root cause;
        ARCHITECTURE_REVIEW.md §4.1.2 / P1.6).
        """
        return cls(
            provider=provider_name,
            model=model,
            temperature=temperature if temperature is not None else default_temperature,
            input_words=input_stats["word_count"],
            input_tokens=input_stats["token_count"],
            output_words=output_words,
            output_tokens=output_tokens,
            latency=latency,
        )

    def format(self) -> str:
        """Format metadata as a string."""
        lines = [
            "=== API Call Metadata ===",
            f"Provider: {self.provider}",
            f"Model: {self.model}",
            f"Temperature: {self.temperature}",
            f"Input Words: {self.input_words}",
            f"Input Tokens: {self.input_tokens}",
            f"Output Words: {self.output_words}",
            f"Output Tokens: {self.output_tokens}",
            f"Latency: {self.latency:.2f}s",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "========================",
            "",
        ]
        return "\n".join(lines)


class ProcessingResult(BaseModel):
    """Result of processing a request."""

    content: str = Field(..., description="Response content")
    metadata: RequestMetadata | None = None
    output_path: str | None = None
    reasoning: str | None = Field(
        default=None,
        description="Chain-of-thought / reasoning_content when using reasoner models",
    )
