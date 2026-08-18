# Ask LLM - Agent Guide

> Agent-focused documentation for the Ask LLM project - a modern CLI tool for calling multiple LLM APIs.

## Project Overview

**Ask LLM** is a modern command-line tool for calling multiple LLM APIs (DeepSeek, Qwen, etc.) with an elegant interface. It is built with:

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
│   ├── cli/                  # Typer CLI package
│   │   ├── app.py            # Typer app assembly and global callback
│   │   ├── commands/         # Per-command modules (ask, chat, batch, trans, paper, format_cmd, config, diagnose)
│   │   ├── common.py         # Shared CLI helpers
│   │   └── errors.py         # CLI error mapping (cli_errors context manager, raise_unexpected_cli_error)
│   ├── core/                 # Core business logic
│   │   ├── models.py                  # Pydantic data models (ProviderConfig/SecretStr, AppConfig, RequestMetadata)
│   │   ├── processor.py               # RequestProcessor (prompt format + LLM call)
│   │   ├── chat.py                    # Interactive chat session (from_initial_context bootstrap)
│   │   ├── batch_models.py            # BatchTask, BatchResult, AttemptRecord, BatchStatistics, TaskStatus
│   │   ├── batch_processor.py         # GlobalBatchProcessor (thin orchestrator: escalation + pool sizing)
│   │   ├── task_executor.py           # Single-config attempt: rate-limit + adapter + stream + metadata
│   │   ├── stream_collector.py        # Streaming + token collection (pure)
│   │   ├── progress_presenter.py      # Per-worker Rich progress bars
│   │   ├── provider_manager.py        # Build provider adapter cache for a batch
│   │   ├── global_batch_runner.py     # run_global_batch_tasks entry point
│   │   ├── command_runner.py          # run_with_checkpoint (shared checkpoint lifecycle for batch/trans)
│   │   ├── concurrent.py              # BoundedRetryRunner (single-queue scheduler + retry heap + SIGINT)
│   │   ├── retry_policy.py            # RetryPolicy / DEFAULT_RETRY_POLICY (transient-error classification)
│   │   ├── checkpoint.py              # Generic atomic checkpoint base (tmp + os.replace)
│   │   ├── batch_checkpoint.py        # Concrete checkpoint for batch/translation tasks
│   │   ├── telemetry.py               # LogContext, bind_context, classify_error, should_fallback_for_error
│   │   ├── execution_report.py        # Structured execution reports (AttemptRecord projection)
│   │   ├── error_keywords.py          # Single (keyword -> category, transient) rule table
│   │   ├── response_parser.py         # unwrap_translation_payload (JSON / LaTeX-escape repair)
│   │   ├── translator.py              # Translation prompt assembly
│   │   ├── text_splitter.py           # TextChunk + base splitter (thin after P3.2)
│   │   ├── markdown_structure.py      # Single-pass parser: fences, frontmatter, heading spans
│   │   ├── binary_splitter.py         # Budget-pluggable splitter (TokenBudget: safety factor + prompt_overhead)
│   │   ├── markdown_token_splitter.py # Thin compat wrapper over BinarySplitter
│   │   ├── chunked_llm_job.py         # Shared orchestration base for Heading/Body formatters
│   │   ├── md_heading_formatter.py    # Heading format pipeline
│   │   ├── md_body_formatter.py       # Body format pipeline (frontmatter carve + position-aware reassembly)
│   │   ├── format_checkpoint.py       # Format checkpoint v2 (original_text + chunk_spans for lossless resume)
│   │   ├── format_markdown_file.py    # Single-file format workflow (title/body dispatch)
│   │   ├── paper_explain.py           # Paper explanation pipeline
│   │   ├── paper_explain_pipeline.py  # Paper pipeline domain model + YAML loader (moved from config/ in P2.6)
│   │   ├── protocols.py               # LLMProviderProtocol
│   │   └── constants.py               # APPROX_TOKEN_SAFETY_FACTOR, TaskKind, defaults
│   ├── services/            # Use-case / orchestration services
│   │   ├── ask_service.py              # Single-request incl. streaming iter_stream (0 typer, returns dataclass)
│   │   ├── batch_service.py            # Batch orchestration (run_batch_from_config + BatchService print/export)
│   │   ├── translation_service.py      # Translation aggregator (delegates to text/notebook collaborators)
│   │   ├── text_file_translator.py     # Per-file text/markdown translation (P4.5)
│   │   ├── notebook_file_translator.py # Per-notebook translation (P4.5)
│   │   ├── translation_options.py      # TranslationOptions / *JobResult / *SessionResult
│   │   ├── paper_service.py            # Paper explain orchestration (returns PaperSessionResult)
│   │   └── format_service.py           # Format orchestration
│   ├── config/              # Configuration management
│   │   ├── loader.py                   # ConfigLoader (single model_validate pass)
│   │   ├── env.py                      # ${VAR} + ASK_LLM_* overrides
│   │   ├── merge.py                    # _deep_merge + provenance (record_leaves)
│   │   ├── providers_catalog.py        # providers.yml runtime-field loader
│   │   ├── unified_config.py           # UnifiedConfig (single source: providers + behavior sections)
│   │   ├── context.py                  # service-locator: get_config / get_config_or_none (13 callers)
│   │   ├── manager.py                  # ConfigManager (provider/model overrides)
│   │   └── cli_session.py              # CLI bootstrap (resolve_and_prepare, gate_api_key_or_exit, bootstrap_command)
│   └── utils/              # Utility modules
│       ├── engine_facade.py            # SINGLE llm_engine import point (create_engine_adapter, EngineConfigView)
│       ├── provider_cache.py           # ProviderAdapterCache (process-wide LRU)
│       ├── fallback_chain.py           # build_fallback_chain (renamed from provider_router, P4.6b)
│       ├── model_limits.py             # DeepSeek max_tokens caps + ModelLimits (renamed from provider_specs, P4.6b)
│       ├── api_key_gate.py             # Pre-flight key checks + UnresolvedAPIKeyError
│       ├── rate_limiter.py             # GlobalRateLimiter (per-(provider,model) token bucket)
│       ├── token_counter.py            # TokenCounter (cl100k_base approximation for DeepSeek/Qwen)
│       ├── chunk_balance.py            # rebalance_translation_chunks (routes through BinarySplitter)
│       ├── console.py                  # Rich + loguru console wrapper
│       ├── file_handler.py             # File I/O with on_chunk progress callbacks
│       ├── batch_loader.py             # Batch YAML config loading (strict validation)
│       ├── batch_exporter.py           # Batch result export (streaming iterencode)
│       ├── translation_exporter.py     # Translation export
│       ├── export_formats.py           # detect_export_format (single extension table)
│       ├── pricing.py                  # Pricing lookup
│       ├── notebook_translator.py      # Jupyter notebook markdown-cell translation
│       ├── path_resolver.py            # _resolve_trans_input_paths / _is_directory_output (P4.3)
│       ├── prompt_resolver.py          # Prompt template loading (@ prefix)
│       ├── md_path_discovery.py        # Markdown path discovery for format
│       └── interactive_config.py       # Interactive provider/key configuration
├── tests/                  # Tests
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Pytest fixtures
├── docs/                   # Documentation
├── prompts/                # Prompt templates (paper/, md-*-format, trans, ...) — symlinked into src/ask_llm/prompts
├── pyproject.toml          # Modern Python project config
├── requirements.txt        # Dependencies
├── providers.yml           # Provider runtime catalog (base_url, models, pricing, specs)
└── default_config.yml      # Unified configuration (run `ask-llm config init` to create)
```

## Coding Conventions

### Style Guide

- **Formatter**: Ruff (replaces Black)
- **Line length**: 100 characters
- **Target Python**: 3.10+
- **Quote style**: Double quotes
- **Import style**: Use `isort` compatible imports (handled by Ruff)

### Type Hints

- Use type hints for function signatures
- Use `X | None` for nullable values (Python 3.10+ syntax)
- Use `Annotated[Type, ...]` for Typer CLI arguments
- Pydantic models for data validation

Example:
```python
from typing import Annotated

def process(
    content: str,
    model: str | None = None,
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

### Unified Configuration (default_config.yml)

The tool uses a single `default_config.yml` for all settings. Run `ask-llm config init` to create a template.

Search order: `--config` > `./default_config.yml` > `~/.config/ask_llm/` > `/etc/ask_llm/` > package built-in.

Sections: `providers`, `general`, `translation`, `batch`, `file`, `format_heading`, `format_body`, `token`, `paper`, `rate_limits`, `project_root_markers`.
Use `${VAR}` in YAML for environment variable substitution.

Provider runtime configuration (base_url, api_key, models, max_output, etc.) lives in `providers.yml`.
Run `ask-llm config init` to generate both `default_config.yml` and `providers.yml` templates.

### CLI Commands

| Command | Description |
|---------|-------------|
| `ask-llm ask [INPUT]` | Process input with LLM |
| `ask-llm chat` | Start interactive chat |
| `ask-llm batch [CONFIG]` | Batch processing from YAML |
| `ask-llm config show` | Display configuration |
| `ask-llm config test` | Test API connections |
| `ask-llm config init` | Generate default_config.yml template |
| `ask-llm format [FILES]` | Format Markdown headings or body via LLM |

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

# Format Markdown
ask-llm format doc.md --type title
ask-llm format doc.md --type body
ask-llm format ./notes_dir --max-depth 1
ask-llm format doc.md --type body --resume doc.md.body_checkpoint.json
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
from ask_llm.config.context import set_config
from ask_llm.config.manager import ConfigManager

load_result = ConfigLoader.load()  # loads default_config.yml
set_config(load_result)  # required for modules using get_config()
manager = ConfigManager(load_result.app_config)
manager.set_provider("deepseek")
manager.apply_overrides(model="gpt-4", temperature=0.5)
```

### Service Layer

CLI commands are thin adapters. Heavy workflows (ask, batch, format, translation, paper explain)
are orchestrated by modules under `ask_llm.services.*`, which receive a prepared
`ConfigManager` / `RequestProcessor` and return structured results for the CLI to print/export.
Services must not call `typer.Exit`; they raise `ValueError`, `FileNotFoundError`, or
`RuntimeError` and let the CLI convert to user-facing messages and exit codes.

```python
from ask_llm.config.cli_session import (
    gate_api_key_or_exit,
    load_cli_session,
    resolve_and_prepare,
)
from ask_llm.services.ask_service import AskService

load_result, config_manager = load_cli_session(config_path)
provider, model = resolve_and_prepare(config_manager, cli_provider=provider)
gate_api_key_or_exit(config_manager, provider)
service = AskService(
    config_manager=config_manager,
    unified_config=load_result.unified_config,
    model=model,
)
# After API key checks:
service.set_processor(processor)
result = service.process_to_file(...)
```

### Bounded Concurrency

All I/O-bound batch work (batch, format, translation chunks) should go through the shared
`BoundedRetryRunner` so retries/backoff and in-flight limits live in one place:

```python
from ask_llm.core.concurrent import run_bounded_with_retries

results = run_bounded_with_retries(
    tasks,
    worker,
    max_workers=8,
    max_retries=3,
    retry_delay=1.0,
    retry_delay_max=10.0,
    is_failed=lambda r: r.status == TaskStatus.FAILED,
    error_message=lambda r: r.error or "",
    retry_count_from_result=lambda r: r.retry_count,
    order_key=lambda r: r.task_id,
)
```

### Rate Limiting

`GlobalBatchProcessor` reads `rate_limits` from the active config and caps its thread pool to
the smallest configured `burst_size` among the tasks, preventing workers from blocking on the
rate limiter waiting for tokens.

```python
from ask_llm.utils.rate_limiter import get_global_rate_limiter

limiter = get_global_rate_limiter(config.unified_config.rate_limits)
limiter.acquire("deepseek", "deepseek-chat", timeout=60.0)
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

## Observability and Performance

### Structured Logging

Use `LogContext` and `bind_context` to inject task/request correlation into Loguru logs:

```python
from ask_llm.core.telemetry import LogContext, bind_context

ctx = LogContext(task_id=task.task_id, provider="deepseek", model="deepseek-chat")
bind_context(ctx).info("Task started")
```

Enable JSON log output with the global `--log-format json` flag.

### Execution Reports

Batch, translation, and paper commands support `--report report.json` to export a
structured report. Inspect reports with:

```bash
ask-llm diagnose report.json
```

### Provider Adapter Cache

`ProviderAdapterCache` keeps HTTP clients warm across runs. Prefer it over creating
adapters directly:

```python
from ask_llm.utils.provider_cache import ProviderAdapterCache

adapter = ProviderAdapterCache.get(provider_config, default_model="gpt-4")
```

## Common Tasks

### Adding a New Command

1. Add command function in `src/ask_llm/cli/commands/` using `@app.command()`
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

## Contributors

- Designed and implemented with assistance from **kimi-code** (agent) and **kimi-k2.7** (model). \
  2.20.0 review & refactor with assistance from **ZCode** (agent) and **GLM-5.3** (model). \
  2.21.0 bug fixes & consolidation with assistance from **ZCode** (agent) and **GLM-5.3** (model).
