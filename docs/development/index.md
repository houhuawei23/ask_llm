# Development

This page covers how to set up the development environment, run tests, and use the code quality tools configured in the project.

## Project Structure

```
ask_llm/
├── src/ask_llm/          # Main package
│   ├── cli/              # Typer CLI commands
│   ├── core/             # Core logic (processor, chat, batch, etc.)
│   ├── config/           # Configuration loading and management
│   └── utils/            # Utilities (console, file handler, token counter)
├── tests/                # Tests
│   ├── unit/
│   └── integration/
├── docs/                 # Documentation
├── pyproject.toml        # Project configuration
└── requirements.txt      # Runtime dependencies
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/ask_llm

# Unit tests only
pytest tests/unit -v

# Integration tests only
pytest tests/integration -v
```

## Linting and Formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check code
ruff check src/ask_llm/

# Auto-fix issues
ruff check --fix src/ask_llm/

# Format code
ruff format src/ask_llm/
```

## Type Checking

```bash
mypy src/ask_llm/
```

## Security Scanning

```bash
# Security vulnerabilities in code
bandit -r src/ask_llm/

# Dependency vulnerability scanning
safety check
```

## Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```
