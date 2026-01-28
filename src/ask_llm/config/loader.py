"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from ask_llm.core.models import AppConfig, ProviderConfig


class ConfigLoader:
    """Load and parse configuration files."""
    
    DEFAULT_CONFIG_PATHS = [
        Path("config.json"),
        Path.home() / ".config" / "ask_llm" / "config.json",
        Path("/etc/ask_llm/config.json"),
    ]
    
    @classmethod
    def load(
        cls,
        config_path: Optional[Union[str, Path]] = None
    ) -> AppConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file. If None, searches default paths.
            
        Returns:
            Parsed application configuration
            
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        path = cls._resolve_config_path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Searched paths: {[str(p) for p in cls.DEFAULT_CONFIG_PATHS]}"
            )
        
        logger.debug(f"Loading configuration from: {path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e
        except Exception as e:
            raise IOError(f"Failed to read config file: {e}") from e
        
        config = cls._parse_config(data)
        logger.info(f"Configuration loaded successfully from {path}")
        return config
    
    @classmethod
    def _resolve_config_path(
        cls,
        config_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Resolve configuration file path."""
        if config_path:
            return Path(config_path)
        
        for path in cls.DEFAULT_CONFIG_PATHS:
            if path.exists():
                logger.debug(f"Found config at: {path}")
                return path
        
        # Return first default path if none found (for error message)
        return cls.DEFAULT_CONFIG_PATHS[0]
    
    @classmethod
    def _parse_config(cls, data: Dict[str, Any]) -> AppConfig:
        """Parse configuration dictionary into AppConfig."""
        if "providers" not in data:
            raise ValueError("Config must contain 'providers' key")
        
        if not isinstance(data["providers"], dict):
            raise ValueError("'providers' must be a dictionary")
        
        default_provider = data.get("default_provider")
        if not default_provider:
            # Use first provider as default if not specified
            default_provider = list(data["providers"].keys())[0]
            logger.warning(f"No default_provider specified, using: {default_provider}")
        
        providers = {}
        for name, config_data in data["providers"].items():
            # Add provider name to config data
            config_data = {**config_data, "api_provider": name}
            try:
                providers[name] = ProviderConfig.model_validate(config_data)
            except Exception as e:
                logger.error(f"Failed to validate config for provider '{name}': {e}")
                raise ValueError(f"Invalid config for provider '{name}': {e}") from e
        
        return AppConfig(
            default_provider=default_provider,
            providers=providers
        )
    
    @classmethod
    def create_example_config(cls, path: Union[str, Path]) -> None:
        """Create an example configuration file."""
        example = {
            "default_provider": "deepseek",
            "providers": {
                "deepseek": {
                    "api_provider": "deepseek",
                    "api_key": "your-api-key-here",
                    "api_base": "https://api.deepseek.com/v1",
                    "api_model": "deepseek-chat",
                    "models": ["deepseek-chat", "deepseek-reasoner"],
                    "api_temperature": 0.7,
                    "api_top_p": 0.95
                },
                "qwen": {
                    "api_provider": "qwen",
                    "api_key": "your-api-key-here",
                    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_model": "qwen-turbo",
                    "models": ["qwen-turbo", "qwen-plus", "qwen-max"],
                    "api_temperature": 0.7,
                    "api_top_p": 0.8
                }
            }
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Example configuration created at: {path}")
