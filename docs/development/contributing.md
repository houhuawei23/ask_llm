# Contributing

Thank you for your interest in improving Ask LLM!

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Install development dependencies: `pip install -e ".[dev]"`.
4. Create a new branch for your changes.

## Code Style

- Follow the existing code style.
- Run `ruff check --fix src/ask_llm/` and `ruff format src/ask_llm/` before committing.
- Ensure `mypy src/ask_llm/` passes without new errors in modified files.

## Testing

- Add tests for new features or bug fixes.
- Run the full test suite with `pytest` before opening a pull request.

## Pull Request Process

1. Push your branch to your fork.
2. Open a pull request against the `main` branch.
3. Describe what changed and why.
4. Ensure CI checks pass.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/houhuawei23/ask_llm/issues).
