"""
Integration tests for CLI commands.

These tests use the Typer CliRunner to test commands without making real API calls.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ask_llm.cli import app


runner = CliRunner()


class TestCLICommands:
    """Test CLI commands."""
    
    def test_version(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()
    
    def test_help(self):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Ask LLM" in result.output
    
    def test_ask_no_input(self):
        """Test ask command without input fails."""
        result = runner.invoke(app, ["ask"])
        assert result.exit_code != 0
    
    def test_config_init(self, temp_dir):
        """Test config init command."""
        config_path = temp_dir / "test_config.json"
        
        result = runner.invoke(app, ["config", "init", "--config", str(config_path)])
        
        assert result.exit_code == 0
        assert config_path.exists()
        
        # Verify it's valid JSON
        with open(config_path) as f:
            data = json.load(f)
        assert "providers" in data
    
    def test_config_show_no_config(self):
        """Test config show without config fails."""
        result = runner.invoke(app, ["config", "show", "--config", "/nonexistent/config.json"])
        assert result.exit_code != 0


class TestCLIWithConfig:
    """Test CLI commands with a config file."""
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock config file."""
        config = {
            "default_provider": "test",
            "providers": {
                "test": {
                    "api_provider": "test",
                    "api_key": "test-key-12345",
                    "api_base": "https://api.test.com/v1",
                    "api_model": "test-model",
                    "models": ["test-model"],
                    "api_temperature": 0.5,
                }
            }
        }
        config_path = temp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path
    
    def test_config_show(self, mock_config):
        """Test config show command."""
        result = runner.invoke(app, ["config", "show", "--config", str(mock_config)])
        
        assert result.exit_code == 0
        assert "test" in result.output
        assert "https://api.test.com/v1" in result.output


class TestDemoScript:
    """Test/demo script for manual verification."""
    
    def test_demo_readme(self, temp_dir):
        """
        Create a demo script that exercises main functionality.
        
        This test creates a demo script that can be run manually.
        """
        demo_script = temp_dir / "demo.py"
        
        script_content = '''
#!/usr/bin/env python3
\"\"\"
Demo script for Ask LLM.

This script demonstrates the main functionality without making API calls.
Run this to verify the installation and basic functionality.
\"\"\"

import tempfile
import json
from pathlib import Path

# Test 1: Configuration
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
finally:
    Path(config_path).unlink()

# Test 2: Token Counter
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

# Test 3: File Handler
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

# Test 4: Chat History
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

# Test 5: Console Output
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

print()
print("=" * 50)
print("All demo tests completed successfully!")
print("=" * 50)
'''
        
        demo_script.write_text(script_content)
        
        # Verify the script was created
        assert demo_script.exists()
        
        print(f"Demo script created at: {demo_script}")
        print("Run it with: python demo.py")
