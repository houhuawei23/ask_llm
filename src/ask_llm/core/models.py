"""Data models for Ask LLM using Pydantic."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
    timestamp: Optional[datetime] = Field(
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

    messages: List[ChatMessage] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the history."""
        self.messages.append(ChatMessage(role=role, content=content))

    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
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

    def to_dict(self) -> Dict[str, Any]:
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


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_provider: str = Field(..., description="Provider identifier")
    api_key: str = Field(..., description="API key for authentication")
    api_base: str = Field(..., description="API base URL")
    models: List[str] = Field(default_factory=list, description="Available models")
    api_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    api_top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens to generate")
    timeout: float = Field(default=60.0, gt=0, description="API timeout in seconds")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not placeholder."""
        if v.strip().lower() in ("your-api-key-here", "", "placeholder"):
            raise ValueError("API key cannot be placeholder or empty")
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
    default_model: Optional[str] = Field(default=None, description="Default model name")
    providers: Dict[str, ProviderConfig] = Field(..., description="Provider configurations")

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, v: Dict[str, ProviderConfig]) -> Dict[str, ProviderConfig]:
        """Validate providers dictionary is not empty."""
        if not v:
            raise ValueError("At least one provider must be configured")
        return v

    def get_provider_config(self, name: Optional[str] = None) -> ProviderConfig:
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
    metadata: Optional[RequestMetadata] = None
    output_path: Optional[str] = None
