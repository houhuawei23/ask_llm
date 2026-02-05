"""Translation configuration loader."""

from pathlib import Path
from typing import ClassVar, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from ask_llm.config.loader import resolve_env_vars


class TransConfig(BaseModel):
    """Translation configuration model."""

    target_language: str = Field(default="zh", description="Target language code")
    source_language: str = Field(
        default="auto", description="Source language code (auto for auto-detection)"
    )
    style: str = Field(default="formal", description="Translation style: formal/casual/technical")
    prompt_template: Optional[str] = Field(default=None, description="Custom prompt template")
    prompt_file: Optional[str] = Field(
        default=None, description="Path to prompt template file (supports @ prefix)"
    )
    threads: int = Field(default=5, ge=1, le=50, description="Number of concurrent threads")
    retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    max_chunk_size: int = Field(default=2000, gt=0, description="Maximum chunk size in characters")
    provider: Optional[str] = Field(default=None, description="API provider name")
    model: Optional[str] = Field(default=None, description="Model name")
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        """Validate translation style."""
        valid_styles = ["formal", "casual", "technical"]
        if v.lower() not in valid_styles:
            logger.warning(f"Invalid style '{v}', using 'formal'")
            return "formal"
        return v.lower()


class TransConfigLoader:
    """Load and parse translation configuration files."""

    DEFAULT_CONFIG_PATHS: ClassVar[list[Path]] = [
        Path("trans_config.yml"),
        Path.home() / ".config" / "ask_llm" / "trans_config.yml",
        Path("/etc/ask_llm/trans_config.yml"),
    ]

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> Optional[TransConfig]:
        """
        Load translation configuration from file.

        Args:
            config_path: Path to configuration file. If None, searches default paths.

        Returns:
            Parsed translation configuration, or None if not found
        """
        path = cls._resolve_config_path(config_path)

        if not path.exists():
            logger.debug(f"Translation config not found at {path}, using defaults")
            return None

        logger.debug(f"Loading translation configuration from: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not data:
                    data = {}
                # Resolve environment variables
                data = resolve_env_vars(data)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in translation config file: {e}")
            raise ValueError(f"Invalid YAML in config file: {e}") from e
        except Exception as e:
            logger.error(f"Failed to read translation config file: {e}")
            raise OSError(f"Failed to read config file: {e}") from e

        try:
            config = TransConfig.model_validate(data)
            logger.info(f"Translation configuration loaded successfully from {path}")
            return config
        except Exception as e:
            logger.error(f"Failed to validate translation config: {e}")
            raise ValueError(f"Invalid translation config: {e}") from e

    @classmethod
    def _resolve_config_path(cls, config_path: Optional[str] = None) -> Path:
        """
        Resolve configuration file path.

        Args:
            config_path: Optional explicit config path

        Returns:
            Resolved path
        """
        if config_path:
            return Path(config_path)

        for path in cls.DEFAULT_CONFIG_PATHS:
            if path.exists():
                logger.debug(f"Found translation config at: {path}")
                return path

        # Return first default path if none found (for error message)
        return cls.DEFAULT_CONFIG_PATHS[0]

    @classmethod
    def get_default_config(cls) -> TransConfig:
        """
        Get default translation configuration.

        Returns:
            Default configuration instance
        """
        return TransConfig()
