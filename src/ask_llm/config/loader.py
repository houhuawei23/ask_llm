"""Configuration loading utilities."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from loguru import logger

from ask_llm.core.models import AppConfig, ProviderConfig


def resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variable references.

    Supports ${VAR_NAME} format environment variable references.

    Args:
        value: Value that may contain environment variable references

    Returns:
        Resolved value
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} format
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)

        if matches:
            for var_name in matches:
                env_value = os.getenv(var_name)
                if env_value:
                    value = value.replace(f"${{{var_name}}}", env_value)
                else:
                    logger.warning(f"Environment variable {var_name} not set")

        return value
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    else:
        return value


class ConfigLoader:
    """Load and parse configuration files."""
    
    DEFAULT_CONFIG_PATHS = [
        Path("providers.yml"),
        Path.home() / ".config" / "ask_llm" / "providers.yml",
        Path("/etc/ask_llm/providers.yml"),
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
        
        # Only support YAML format
        if path.suffix not in (".yml", ".yaml"):
            raise ValueError(
                f"Unsupported config file format: {path.suffix}. "
                f"Only YAML (.yml, .yaml) files are supported."
            )
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not data:
                    data = {}
                # Resolve environment variables
                data = resolve_env_vars(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}") from e
        except Exception as e:
            raise IOError(f"Failed to read config file: {e}") from e
        
        # Convert providers.yml format to AppConfig format
        data = cls._convert_providers_yml_format(data)
        
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
    def _convert_providers_yml_format(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert providers.yml format to AppConfig format.
        
        providers.yml format:
        {
            "providers": {
                "deepseek": {
                    "base_url": "...",
                    "api_key": "...",
                    "default_model": "...",
                    "models": [{"name": "..."}, ...]
                }
            }
        }
        
        AppConfig format:
        {
            "default_provider": "...",
            "default_model": "...",
            "providers": {
                "deepseek": {
                    "api_provider": "deepseek",
                    "api_base": "...",
                    "api_key": "...",
                    "models": ["...", ...],
                    ...
                }
            }
        }
        """
        if "providers" not in data:
            return data
        
        providers = data["providers"]
        if not isinstance(providers, dict):
            return data
        
        # Determine default provider
        default_provider = data.get("default_provider")
        default_model = data.get("default_model")
        
        converted_providers = {}
        for name, provider_config in providers.items():
            if not isinstance(provider_config, dict):
                continue
            
            # Convert provider config
            base_url = provider_config.get("base_url", "")
            # If base_url is empty or None, try to get default from llm-engine config
            if not base_url:
                try:
                    from llm_engine.config_loader import load_providers_config, get_model_info
                    providers_config = load_providers_config()
                    if providers_config and name in providers_config.get("providers", {}):
                        base_url = providers_config["providers"][name].get("base_url", "")
                except Exception:
                    pass
            
            # If still empty, use a placeholder that passes validation
            # The actual base_url will be determined by the provider implementation
            if not base_url:
                base_url = "https://api.example.com/v1"
            
            converted_config = {
                "api_provider": name,
                "api_key": provider_config.get("api_key", ""),
                "api_base": base_url,
            }
            
            # Handle models list - convert from list of dicts to list of strings
            models = provider_config.get("models", [])
            provider_default_model = provider_config.get("default_model")
            
            if models:
                model_names = []
                for model in models:
                    if isinstance(model, dict):
                        model_name = model.get("name")
                        if model_name:
                            model_names.append(model_name)
                    elif isinstance(model, str):
                        model_names.append(model)
                
                # Ensure default_model is first in the list if specified
                if provider_default_model:
                    if provider_default_model in model_names:
                        # Move default_model to first position
                        model_names.remove(provider_default_model)
                    model_names.insert(0, provider_default_model)
                
                converted_config["models"] = model_names
                
                # Set global default_model if not already set (use first provider's default)
                if not default_model and provider_default_model:
                    default_model = provider_default_model
                elif not default_model and model_names:
                    default_model = model_names[0]
            elif provider_default_model:
                converted_config["models"] = [provider_default_model]
                if not default_model:
                    default_model = provider_default_model
            else:
                # No models and no default_model - this is an error but we'll handle it later
                converted_config["models"] = []
            
            # Set default_provider if not already set
            if not default_provider:
                default_provider = name
            
            # Add other optional fields
            if "api_temperature" in provider_config:
                converted_config["api_temperature"] = provider_config["api_temperature"]
            if "api_top_p" in provider_config:
                converted_config["api_top_p"] = provider_config["api_top_p"]
            if "max_tokens" in provider_config:
                converted_config["max_tokens"] = provider_config["max_tokens"]
            if "timeout" in provider_config:
                converted_config["timeout"] = provider_config["timeout"]
            
            converted_providers[name] = converted_config
        
        return {
            "default_provider": default_provider,
            "default_model": default_model,
            "providers": converted_providers,
        }
    
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
        
        default_model = data.get("default_model")
        
        providers = {}
        for name, config_data in data["providers"].items():
            # Add provider name to config data
            config_data = {**config_data, "api_provider": name}
            # Remove api_model if present (for backward compatibility)
            config_data.pop("api_model", None)
            try:
                providers[name] = ProviderConfig.model_validate(config_data)
            except Exception as e:
                logger.error(f"Failed to validate config for provider '{name}': {e}")
                raise ValueError(f"Invalid config for provider '{name}': {e}") from e
        
        return AppConfig(
            default_provider=default_provider,
            default_model=default_model,
            providers=providers
        )
    
