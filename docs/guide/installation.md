# Installation

Ask LLM requires Python 3.8 or newer.

## From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/houhuawei23/ask_llm.git
cd ask_llm

# Install runtime dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

## Install Development Dependencies

To run tests, linting, type checking, and security scans:

```bash
# Install all dev dependencies
pip install -e ".[dev]"

# Or install subsets
pip install -e ".[lint]"
pip install -e ".[security]"
```

## Verify Installation

After installation, the following entry points should be available:

```bash
ask-llm --version
askllm --version
```

Both commands resolve to the same CLI.
