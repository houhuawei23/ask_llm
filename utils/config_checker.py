"""Configuration checking and API testing utilities."""

import time
from typing import Dict, Any, List, Optional, Tuple

from .logger import logger
from .config_loader import load_config, get_provider_config
from providers import OpenAICompatibleProvider


def check_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Check configuration file and return status information.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration status
    """
    try:
        config = load_config(config_path)
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'providers': {}
        }
    
    providers_status = {}
    providers = config.get('providers', {})
    default_provider = config.get('default_provider')
    
    for provider_name, provider_config in providers.items():
        status = {
            'name': provider_name,
            'is_default': provider_name == default_provider,
            'has_api_key': bool(provider_config.get('api_key') and 
                              provider_config.get('api_key') != 'your-api-key-here'),
            'api_key_status': 'configured' if (provider_config.get('api_key') and 
                                               provider_config.get('api_key') != 'your-api-key-here') else 'not configured',
            'api_base': provider_config.get('api_base', 'missing'),
            'api_model': provider_config.get('api_model', 'missing'),
            'models': provider_config.get('models', []),
            'temperature': provider_config.get('api_temperature', 'default'),
            'missing_fields': [],
        }
        
        # Check required fields
        required_fields = ['api_key', 'api_base', 'api_model']
        for field in required_fields:
            if field not in provider_config or not provider_config[field]:
                status['missing_fields'].append(field)
            elif field == 'api_key' and provider_config[field] == 'your-api-key-here':
                status['missing_fields'].append(field)
        
        providers_status[provider_name] = status
    
    return {
        'valid': True,
        'config_path': config_path,
        'default_provider': default_provider,
        'providers': providers_status
    }


def test_api(provider_name: str, config_path: str = "config.json", 
             test_message: str = "Hello") -> Tuple[bool, str, Optional[float]]:
    """
    Test API connection for a specific provider.
    
    Args:
        provider_name: Name of the provider to test
        config_path: Path to configuration file
        test_message: Test message to send
        
    Returns:
        Tuple of (success, message, latency)
    """
    try:
        config = load_config(config_path)
        provider_config = get_provider_config(config, provider_name)
        
        # Initialize provider
        provider = OpenAICompatibleProvider(provider_config)
        
        # Send test request
        start_time = time.time()
        response = provider.call(prompt=test_message, max_tokens=10)
        latency = time.time() - start_time
        
        return True, f"API test successful. Response: {response[:50]}...", latency
        
    except Exception as e:
        return False, f"API test failed: {str(e)}", None


def print_config_status(config_status: Dict[str, Any], test_api_flag: bool = False) -> None:
    """
    Print configuration status in a formatted way.
    
    Args:
        config_status: Status dictionary from check_config()
        test_api_flag: Whether to test API connections
    """
    if not config_status['valid']:
        print(f"❌ Configuration Error: {config_status.get('error', 'Unknown error')}")
        return
    
    print(f"\n{'='*60}")
    print(f"Configuration File: {config_status['config_path']}")
    print(f"Default Provider: {config_status.get('default_provider', 'not set')}")
    print(f"{'='*60}\n")
    
    providers = config_status['providers']
    
    for provider_name, status in providers.items():
        default_marker = " (default)" if status['is_default'] else ""
        print(f"Provider: {provider_name}{default_marker}")
        print(f"  API Key: {status['api_key_status']}")
        print(f"  API Base: {status['api_base']}")
        print(f"  Default Model: {status['api_model']}")
        
        if status['models']:
            models_str = ', '.join(status['models'])
            print(f"  Available Models: {models_str}")
        
        print(f"  Temperature: {status['temperature']}")
        
        if status['missing_fields']:
            print(f"  ⚠️  Missing Fields: {', '.join(status['missing_fields'])}")
        
        # Test API if requested
        if test_api_flag and status['has_api_key']:
            print(f"  Testing API connection...", end=' ', flush=True)
            success, message, latency = test_api(provider_name, config_status['config_path'])
            if success:
                print(f"✓ ({latency:.2f}s)")
                print(f"    {message}")
            else:
                print(f"✗")
                print(f"    {message}")
        
        print()


def check_and_print_config(config_path: str = "config.json", 
                          test_api_flag: bool = False,
                          provider_name: Optional[str] = None) -> None:
    """
    Check configuration and print status.
    
    Args:
        config_path: Path to configuration file
        test_api_flag: Whether to test API connections
        provider_name: If specified, only test this provider
    """
    config_status = check_config(config_path)
    
    if provider_name:
        # Only check/test specified provider
        if provider_name in config_status['providers']:
            provider_status = config_status['providers'][provider_name]
            if test_api_flag and provider_status['has_api_key']:
                print(f"\nTesting API for provider: {provider_name}")
                success, message, latency = test_api(provider_name, config_path)
                if success:
                    print(f"✓ API test successful ({latency:.2f}s)")
                    print(f"Response: {message}")
                else:
                    print(f"✗ API test failed")
                    print(f"Error: {message}")
                print()
            else:
                # Just print config status for this provider
                print(f"\nProvider: {provider_name}")
                print(f"  API Key: {provider_status['api_key_status']}")
                print(f"  API Base: {provider_status['api_base']}")
                print(f"  Default Model: {provider_status['api_model']}")
                if provider_status['models']:
                    print(f"  Available Models: {', '.join(provider_status['models'])}")
                print()
        else:
            print(f"❌ Provider '{provider_name}' not found in configuration.")
    else:
        print_config_status(config_status, test_api_flag)

