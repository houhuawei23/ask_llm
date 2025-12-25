"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If config structure is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {str(e)}") from e
    
    # Validate config structure
    if 'providers' not in config:
        raise ValueError("Config file must contain 'providers' key")
    
    if not isinstance(config['providers'], dict):
        raise ValueError("'providers' must be a dictionary")
    
    return config


def get_provider_config(
    config: Dict[str, Any],
    provider_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get configuration for a specific provider.
    
    Args:
        config: Full configuration dictionary
        provider_name: Name of the provider (if None, uses default)
        
    Returns:
        Provider configuration dictionary
        
    Raises:
        ValueError: If provider not found
    """
    providers = config.get('providers', {})
    
    # Determine which provider to use
    if provider_name is None:
        provider_name = config.get('default_provider')
        if provider_name is None:
            raise ValueError(
                "No provider specified and no 'default_provider' in config"
            )
    
    if provider_name not in providers:
        raise ValueError(
            f"Provider '{provider_name}' not found in config. "
            f"Available providers: {', '.join(providers.keys())}"
        )
    
    provider_config = providers[provider_name].copy()
    provider_config['api_provider'] = provider_name
    
    return provider_config


def merge_cli_overrides(
    base_config: Dict[str, Any],
    cli_args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge CLI arguments into base configuration.
    
    Args:
        base_config: Base provider configuration
        cli_args: CLI argument overrides (model, temperature, etc.)
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    if 'model' in cli_args and cli_args['model']:
        merged['api_model'] = cli_args['model']
    
    if 'temperature' in cli_args and cli_args['temperature'] is not None:
        merged['api_temperature'] = cli_args['temperature']
    
    return merged

