# Ask LLM v2.0

A modern command-line tool for calling multiple LLM APIs (DeepSeek, Qwen, etc.) with an elegant interface.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Features

- ✨ **Modern CLI** - Built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
- 🔧 **Type Safe** - Full type hints and Pydantic validation
- 📊 **Progress Bars** - Visual feedback for file operations
- 📝 **Rich Logging** - Powered by Loguru
- 💬 **Interactive Chat** - Multi-turn conversations with command support
- 🔌 **Multiple Providers** - Support for OpenAI-compatible APIs
- 📦 **Batch Processing** - Process multiple tasks concurrently with multi-threading

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ask_llm

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Configuration

```bash
# Create default_config.yml template
ask-llm config init

# Edit default_config.yml with your API keys (use ${VAR} for environment variables)
# Config priority: CLI args > env vars (ASK_LLM_*) > user config > package default
# Config is searched in: --config > ./default_config.yml > ~/.config/ask_llm/ > /etc/ask_llm/
# Then verify
ask-llm config test
```

### Usage

```bash
# Process a file
ask-llm ask input.txt

# Direct text input
ask-llm "Translate to Chinese: Hello world"

# Interactive chat mode
ask-llm chat

# With initial context
ask-llm chat -i context.txt -s "You are a helpful assistant"

# Translation (file, directory, or glob)
ask-llm trans document.md
ask-llm trans /path/to/dir/ -o translated/
ask-llm trans *.md --max-parallel-files 5

# Batch processing
ask-llm batch batch-examples/prompt-contents.yml -o results.json
```

## Commands

| Command | Description |
|---------|-------------|
| `ask-llm ask [INPUT]` | Process input with LLM |
| `ask-llm chat` | Start interactive chat |
| `ask-llm trans [FILES...]` | Translate files (supports directory and glob) |
| `ask-llm batch [CONFIG]` | Process batch tasks from YAML config |
| `ask-llm config show` | Display configuration |
| `ask-llm config test` | Test API connections |
| `ask-llm config init` | Create example config |

### Batch Processing

The `batch` command supports processing multiple tasks concurrently:

```bash
# Basic usage
ask-llm batch batch-examples/prompt-contents.yml

# With options
ask-llm batch config.yml -o results.json -f json --threads 10 --retries 5
```

See [docs/BATCH_USAGE.md](docs/BATCH_USAGE.md) for detailed batch processing documentation.

## Project Structure

```
ask_llm/
├── src/ask_llm/          # Main package
│   ├── cli.py            # CLI entry point
│   ├── core/             # Core logic
│   ├── config/           # Configuration
│   └── utils/            # Utilities
├── tests/                # Tests
├── docs/                 # Documentation
└── default_config.yml    # Unified configuration (run `ask-llm config init` to create)
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src/ask_llm

# Type checking
mypy src/ask_llm

# Linting
ruff check src/ask_llm
ruff format src/ask_llm
```

## Documentation

See [docs/README_ask_llm.md](docs/README_ask_llm.md) for detailed documentation.

## License

MIT License
