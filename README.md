# Ask LLM v2.3.0

A modern command-line tool for calling multiple LLM APIs (DeepSeek, Qwen, etc.) with an elegant interface.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/houhuawei23/ask_llm)

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

# Paper explanation (Markdown by headings, or arxiv2md-beta directory)
ask-llm paper -i paper.md --run all
ask-llm paper -i path/to/arxiv-paper-dir --run sections
```

## Commands

| Command                    | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| `ask-llm ask [INPUT]`      | Process input with LLM                                                    |
| `ask-llm chat`             | Start interactive chat                                                    |
| `ask-llm trans [FILES...]` | Translate files (supports directory and glob)                             |
| `ask-llm paper -i PATH`    | Explain a paper: outputs under `./explain/` next to the file or directory |
| `ask-llm batch [CONFIG]`   | Process batch tasks from YAML config                                      |
| `ask-llm config show`      | Display configuration                                                     |
| `ask-llm config test`      | Test API connections                                                      |
| `ask-llm config init`      | Create example config                                                     |

### Batch Processing

The `batch` command supports processing multiple tasks concurrently:

```bash
# Basic usage
ask-llm batch batch-examples/prompt-contents.yml

# With options
ask-llm batch config.yml -o results.json -f json --threads 10 --retries 5
```

See [docs/BATCH_USAGE.md](docs/BATCH_USAGE.md) for detailed batch processing documentation.

### Paper explanation (`paper`)

- **Input**: a single `.md` whose **level-2** headings (`## …`) delimit explain sections (subsections `###`+ stay inside the same job), **or** a directory produced by tools like arxiv2md-beta (`paper.yml`, main `*.md`, optional `*-References.md`, `*-Appendix.md`).
- **Runs**: `--run sections` (meta + each recognized section), `--run full` (whole document), `--run all` (both).
- **Output**: `<input_dir>/explain/` (or next to the `.md` file). Files are **numbered in document order**, e.g. `0-meta.explain.md`, `1-abstract.explain.md`, …, `N-full.explain.md`. Recognized **CS/AI-oriented** section titles (among others) map to dedicated prompts: e.g. **Related Work** → `section-related-work.md`, **Model Architecture** → `section-model-architecture.md`. Headings that still do not match any canonical key use `section-generic.md` (`extra:…` keys, filenames like `3-model-architecture.explain.md`). Large **Appendix** sidecars split by `##` use `d-appendices-<slug>.explain.md`.
- **Preamble**: each output file starts with a short **说明** block (source slice + prompt path + one-line summary of the analysis prompt).
- **Length & models**: `paper.max_output_tokens` is the requested completion cap; the CLI sets API `max_tokens` to **min(requested, `max_output.maximum`)** from `providers.yml` for that model. **DeepSeek** HTTP caps differ by model: **`deepseek-chat`** ≤8192, **`deepseek-reasoner`** ≤65536 (then `min` with YAML). The **full-document** job (`full`) uses `paper.full_model` (default `deepseek-reasoner`). When the API returns reasoning content, it is written under **推理过程（思维链）** before **正文解析**. On API errors, the log includes **`model=`** and **`max_tokens=`** (from `llm_engine`).
- **Concurrency**: section jobs use **`GlobalBatchProcessor.process_global_tasks`** (same pipeline as `ask-llm trans`): each job gets its own provider/HTTP client, with Rich per-task progress. Default `paper.concurrency` (e.g. `20`); override with `ask-llm paper -i ... -j 8`. Use `1` to force sequential calls.
- **Prompts**: canonical tree is `prompts/paper/` at the **repository root** (`ask_llm/prompts/…`). Templates default to **computer science / AI** papers (methodology, experiments, reproducibility, related-work positioning, multiview-style full-paper analysis). Under `src/ask_llm/` the `prompts` entry is a **symlink** to that tree so setuptools package-data stays valid. Override directory via `paper.prompt_dir` in `default_config.yml`.

## Project Structure

```
ask_llm/
├── prompts/              # Prompt templates (paper/, tech-paper-trans.md, …)
├── src/ask_llm/          # Main package (prompts → symlink to ../prompts)
│   ├── cli/              # Typer CLI (app.py, commands/, common.py, errors.py)
│   ├── core/             # Core logic (batch, processor, paper_explain, …)
│   ├── config/           # Configuration
│   └── utils/            # Utilities
├── tests/                # Tests
├── docs/                 # Documentation
└── default_config.yml    # Unified configuration (run `ask-llm config init` to create)
```

### Batch, translation, and paper internals

- `ask_llm.config.cli_session` — shared config load (`ConfigLoader` + `set_config`), `ConfigManager`, CLI overrides, and API key gate used by `trans`, `paper`, and `batch`.
- `ask_llm.core.global_batch_runner` — `run_global_batch_tasks` creates `GlobalBatchProcessor` and runs `process_global_tasks` (optional worker clamp vs. task count for paper).
- `ask_llm.core.tasks.builders` — factories such as `build_paper_explain_task` for typed `BatchTask` construction.
- `BatchTask.task_kind` — `translation_chunk` or `paper_explain`; legacy `paper_mode=True` still maps to `paper_explain`.

## Development

### CLI package layout (`ask_llm/cli/`)

The Typer entry point is `ask_llm.cli:run_cli` (see `pyproject.toml` scripts). The former monolithic `cli.py` is split as follows:

| Module          | Role                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| `cli/app.py`    | `Typer` app, global `--version` / `--debug` / `--quiet` callback, registers subcommands                      |
| `cli/commands/` | One module per command: `ask`, `chat`, `config`, `batch`, `trans`, `format_cmd` (CLI name `format`), `paper` |
| `cli/common.py` | Shared helpers (`_config_init`, `_resolve_trans_input_paths`, notebook translation helper)                   |
| `cli/errors.py` | `raise_unexpected_cli_error`, optional `cli_errors` context manager for consistent exit codes and logging    |

Public imports from `ask_llm.cli` remain `app`, `run_cli`, and `_resolve_trans_input_paths` (for tests and tooling).

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
