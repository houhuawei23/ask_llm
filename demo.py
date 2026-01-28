#!/usr/bin/env python3
"""
Demo script for Ask LLM.

This script demonstrates the main functionality without making API calls.
Run this to verify the installation and basic functionality.
"""

import tempfile
import json
from pathlib import Path

def test_configuration():
    """Test 1: Configuration Loading"""
    print("=" * 50)
    print("Test 1: Configuration Loading")
    print("=" * 50)
    
    from ask_llm.config.loader import ConfigLoader
    from ask_llm.core.models import ProviderConfig, AppConfig
    
    # Create a test config
    config_data = {
        "default_provider": "demo",
        "providers": {
            "demo": {
                "api_provider": "demo",
                "api_key": "demo-key-12345",
                "api_base": "https://api.demo.com/v1",
                "api_model": "demo-model",
                "models": ["demo-model"],
                "api_temperature": 0.5,
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        config = ConfigLoader.load(config_path)
        print(f"✓ Loaded config: {config.default_provider}")
        print(f"✓ Providers: {list(config.providers.keys())}")
        print(f"✓ Models for demo: {config.providers['demo'].models}")
    finally:
        Path(config_path).unlink()

def test_token_counter():
    """Test 2: Token Counter"""
    print()
    print("=" * 50)
    print("Test 2: Token Counter")
    print("=" * 50)
    
    from ask_llm.utils.token_counter import TokenCounter
    
    text = "Hello, this is a test message for token counting."
    stats = TokenCounter.estimate_tokens(text)
    print(f"✓ Text: {text}")
    print(f"  Words: {stats['word_count']}")
    print(f"  Tokens: {stats['token_count']}")
    print(f"  Chars: {stats['char_count']}")

def test_file_handler():
    """Test 3: File Handler"""
    print()
    print("=" * 50)
    print("Test 3: File Handler")
    print("=" * 50)
    
    from ask_llm.utils.file_handler import FileHandler
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test
        test_file = Path(tmpdir) / "test.txt"
        FileHandler.write(test_file, "Test content for file handler")
        print(f"✓ Wrote to {test_file}")
        
        # Read test
        content = FileHandler.read(test_file)
        print(f"✓ Read content: {content}")
        
        # Output path generation
        output = FileHandler.generate_output_path(test_file)
        print(f"✓ Generated output path: {output}")

def test_chat_history():
    """Test 4: Chat History"""
    print()
    print("=" * 50)
    print("Test 4: Chat History")
    print("=" * 50)
    
    from ask_llm.core.models import ChatHistory, MessageRole
    
    history = ChatHistory(provider="demo", model="demo-model")
    history.add_message(MessageRole.SYSTEM, "You are a helpful assistant.")
    history.add_message(MessageRole.USER, "Hello!")
    history.add_message(MessageRole.ASSISTANT, "Hi there!")
    
    print(f"✓ Created history with {len(history.messages)} messages")
    messages = history.get_messages()
    print(f"✓ Exported {len(messages)} messages for API")
    print(f"✓ Message roles: {[m['role'] for m in messages]}")

def test_console():
    """Test 5: Console Output"""
    print()
    print("=" * 50)
    print("Test 5: Console Output")
    print("=" * 50)
    
    from ask_llm.utils.console import console
    
    console.setup(quiet=False, debug=False)
    console.print_success("Success message test")
    console.print_info("Info message test")
    console.print_warning("Warning message test")
    console.print("Regular message test")
    
    # Test table
    console.print_table(
        headers=["Provider", "Model", "Status"],
        rows=[["DeepSeek", "deepseek-chat", "✓"], ["Qwen", "qwen-turbo", "✓"]]
    )

def test_models():
    """Test 6: Data Models"""
    print()
    print("=" * 50)
    print("Test 6: Data Models")
    print("=" * 50)
    
    from ask_llm.core.models import ProviderConfig, RequestMetadata
    
    # ProviderConfig
    config = ProviderConfig(
        api_provider="test",
        api_key="test-key-12345",
        api_base="https://api.test.com/v1",
        api_model="test-model",
        api_temperature=0.5,
    )
    print(f"✓ ProviderConfig: {config.api_provider} -> {config.api_model}")
    
    # RequestMetadata
    metadata = RequestMetadata(
        provider="test",
        model="test-model",
        temperature=0.5,
        input_tokens=100,
        output_tokens=50,
        latency=1.23,
    )
    print(f"✓ RequestMetadata: {metadata.input_tokens} -> {metadata.output_tokens} tokens")

def test_config_manager():
    """Test 7: Config Manager"""
    print()
    print("=" * 50)
    print("Test 7: Config Manager")
    print("=" * 50)
    
    from ask_llm.config.manager import ConfigManager
    from ask_llm.core.models import AppConfig, ProviderConfig
    
    config = AppConfig(
        default_provider="provider1",
        providers={
            "provider1": ProviderConfig(
                api_provider="provider1",
                api_key="key1",
                api_base="https://api1.com",
                api_model="model1",
            ),
        }
    )
    
    manager = ConfigManager(config)
    print(f"✓ Current provider: {manager.current_provider_name}")
    
    manager.apply_overrides(model="new-model", temperature=0.9)
    provider_config = manager.get_provider_config()
    print(f"✓ After override: {provider_config.api_model}, temp={provider_config.api_temperature}")

def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Ask LLM Demo - Testing Core Functionality")
    print("=" * 50 + "\n")
    
    try:
        test_configuration()
        test_token_counter()
        test_file_handler()
        test_chat_history()
        test_console()
        test_models()
        test_config_manager()
        
        print()
        print("=" * 50)
        print("✓ All demo tests completed successfully!")
        print("=" * 50)
        print()
        
    except Exception as e:
        print()
        print("=" * 50)
        print(f"✗ Test failed: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
