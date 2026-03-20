# Ask LLM - Agent Guide

> Agent-focused documentation for the Ask LLM project - a modern CLI tool for calling multiple LLM APIs.

## Project Overview

**Ask LLM v2.0** is a modern command-line tool for calling multiple LLM APIs (DeepSeek, Qwen, etc.) with an elegant interface. It is built with:

- **Typer** - Modern CLI framework with type hints
- **Pydantic** - Data validation and serialization
- **Rich** - Beautiful console output
- **Loguru** - Powerful logging
- **llm-api-engine** - External LLM engine dependency

## Project Structure

```
ask_llm/
├── src/ask_llm/              # Main package
│   ├── __init__.py           # Version info
│   ├── __main__.py           # python -m entry point
│   ├── cli.py                # Typer CLI implementation (main entry)
│   ├── core/                 # Core business logic
│   │   ├── models.py         # Pydantic data models
│   │   ├── processor.py      # Request processing
│   │   ├── chat.py           # Interactive chat session
│   │   ├── batch.py          # Batch processing
│   │   ├── translator.py     # Translation utilities
│   │   ├── text_splitter.py  # Text splitting logic
│   │   └── md_heading_formatter.py  # Markdown formatting
│   ├── config/               # Configuration management
│   │   ├── loader.py         # Config loading
│   │   └── manager.py        # Config management
│   └── utils/                # Utility modules
│       ├── console.py        # Rich console wrapper
│       ├── file_handler.py   # File I/O with progress bars
│       ├── token_counter.py  # Token counting
│       ├── batch_exporter.py # Batch result export
│       ├── batch_loader.py   # Batch config loading
│       ├── notebook_translator.py  # Jupyter notebook support
│       ├── trans_config_loader.py  # Translation config
│       └── translation_exporter.py # Translation export
├── tests/                    # Tests
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── conftest.py           # Pytest fixtures
├── docs/                     # Documentation
├── pyproject.toml            # Modern Python project config
├── requirements.txt          # Dependencies
└── providers.yml             # Provider configuration
```

## Coding Conventions

### Style Guide

- **Formatter**: Ruff (replaces Black)
- **Line length**: 100 characters
- **Target Python**: 3.8+
- **Quote style**: Double quotes
- **Import style**: Use `isort` compatible imports (handled by Ruff)

### Type Hints

- Use type hints for function signatures
- Use `Optional[Type]` for nullable values
- Use `Annotated[Type, ...]` for Typer CLI arguments
- Pydantic models for data validation

Example:
```python
from typing import Optional
from typing_extensions import Annotated

def process(
    content: str,
    model: Optional[str] = None,
    temperature: Annotated[float, typer.Option()] = 0.7
) -> ProcessingResult:
    ...
```

### Docstrings

Use **Google style** docstrings:

```python
def process_content(content: str, template: str) -> str:
    """Process content with template.

    Args:
        content: Input content to process.
        template: Template string with {content} placeholder.

    Returns:
        Processed content string.

    Raises:
        ValueError: If template is invalid.
    """
```

### Error Handling

- Use specific exceptions (`FileNotFoundError`, `ValueError`, `RuntimeError`)
- Convert exceptions to user-friendly messages via `console.print_error()`
- Exit with appropriate codes using `raise typer.Exit(code)`

### Logging

Use Loguru's `logger`:
```python
from loguru import logger

logger.debug("Debug info")
logger.info("Information")
logger.warning("Warning")
logger.error("Error occurred")
```

## Build & Test Commands

### Installation

```bash
# Development install
pip install -e ".[dev]"

# Or specific extras
pip install -e ".[lint]"
pip install -e ".[security]"
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ask_llm --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py -v

# Run integration tests
pytest tests/integration -v
```

### Code Quality

```bash
# Ruff - lint and format
ruff check src/ask_llm/
ruff check --fix src/ask_llm/
ruff format src/ask_llm/

# MyPy - type checking
mypy src/ask_llm --ignore-missing-imports

# Pydocstyle - docstring checking
pydocstyle src/ask_llm/

# Bandit - security scanning
bandit -r src/ask_llm/ -ll

# Safety - dependency vulnerability check
safety check

# Pre-commit hooks
pre-commit run --all-files
```

### One-Click Check Script

```bash
./scripts/check_code_quality.sh
```

## Key Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| typer | CLI framework | >=0.9.0 |
| rich | Console output | >=13.0.0 |
| pydantic | Data validation | >=2.0.0 |
| loguru | Logging | >=0.7.0 |
| openai | OpenAI API client | >=1.0.0 |
| llm-api-engine | External LLM engine | >=0.1.2 |
| tiktoken | Token counting | >=0.5.0 |
| tqdm | Progress bars | >=4.65.0 |
| pyyaml | YAML support | >=6.0.0 |
| nbformat | Jupyter notebook support | >=5.0.0 |

## Configuration

### Provider Configuration

The tool uses a YAML/JSON configuration file for API providers:

```yaml
# providers.yml
default_provider: deepseek
default_model: deepseek-chat
providers:
  deepseek:
    api_provider: deepseek
    api_key: sk-your-api-key
    api_base: https://api.deepseek.com/v1
    models:
      - deepseek-chat
      - deepseek-reasoner
    api_temperature: 0.7
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `ask-llm ask [INPUT]` | Process input with LLM |
| `ask-llm chat` | Start interactive chat |
| `ask-llm batch [CONFIG]` | Batch processing from YAML |
| `ask-llm config show` | Display configuration |
| `ask-llm config test` | Test API connections |

### Example Usage

```bash
# Process a file
ask-llm ask input.txt -o output.txt

# Direct text input
ask-llm ask "Translate to Chinese: Hello world"

# Interactive chat
ask-llm chat -i context.txt -s "You are a helpful assistant"

# Batch processing
ask-llm batch config.yml -o results.json --threads 10
```

## Architecture Patterns

### Provider Pattern

The project uses `llm-api-engine` as the LLM provider abstraction:

```python
from llm_engine import create_provider_adapter

provider = create_provider_adapter(provider_config, default_model="gpt-4")
response = provider.call(messages=[...])
```

### Config Management

```python
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager

config = ConfigLoader.load("config.json")
manager = ConfigManager(config)
manager.set_provider("deepseek")
manager.apply_overrides(model="gpt-4", temperature=0.5)
```

### Console Output

```python
from ask_llm.utils.console import console

console.print_success("Success message")
console.print_error("Error message")
console.print_info("Info message")
console.print_table(headers=["Name", "Value"], rows=[["k1", "v1"]])
```

## Testing Conventions

### Test Structure

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Fixtures in `tests/conftest.py`

### Test Naming

- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Pytest Markers

```python
import pytest

@pytest.mark.unit
def test_something():
    pass

@pytest.mark.integration
def test_api_call():
    pass

@pytest.mark.slow
def test_heavy_computation():
    pass
```

## Common Tasks

### Adding a New Command

1. Add command function in `src/ask_llm/cli.py` using `@app.command()`
2. Use type hints and `Annotated` for CLI arguments
3. Use `console.print_*` methods for output
4. Add tests in `tests/unit/` or `tests/integration/`

### Adding a New Provider

Providers are handled externally by `llm-api-engine`. Update configuration in `providers.yml`.

### Adding New Models

1. Add Pydantic model in `src/ask_llm/core/models.py`
2. Add tests in `tests/unit/test_models.py`
3. Update documentation

## Important Notes

- **Do not** commit API keys - use environment variables or `.env` files
- **Do not** modify files outside the working directory
- Run pre-commit hooks before committing: `pre-commit run --all-files`
- The project uses modern Python packaging with `pyproject.toml`
- All CLI output should go through `console` utility for consistent formatting
