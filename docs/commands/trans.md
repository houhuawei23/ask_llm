# trans

Translate text files using an LLM API. Supports plain text (`.txt`), Markdown (`.md`), and Jupyter notebooks (`.ipynb`). For notebooks, only markdown cells are translated; code cells are preserved.

## Usage

```bash
ask-llm trans [FILES...] [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `FILES...` | Input file(s) to translate (supports glob patterns and directories) |

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file or directory path |
| `--config` | `-c` | Path to `default_config.yml` |
| `--target-lang` | `-t` | Target language code |
| `--source-lang` | `-s` | Source language code (default: auto-detect) |
| `--threads` | `-T` | Per-file concurrent API calls |
| `--max-parallel-files` | | Max files to process in parallel when translating a directory (default: 3) |
| `--retries` | `-r` | Maximum number of retries |
| `--provider` | `-a` | API provider to use |
| `--model` | `-m` | Model name to use |
| `--force` | `-f` | Overwrite existing output file |
| `--preserve-format` / `--no-preserve-format` | | Preserve original formatting (default: preserve) |
| `--stream` | | Stream translation progress to console |
| `--prompt` | `-p` | Path to prompt template file (supports `@` prefix for project-relative paths) |
| `--providers-pricing` | | Path to `providers.yml` for cost estimates |
| `--no-balance-chunks` | | Disable token-based chunk rebalancing |
| `--max-chunk-tokens` | | Max estimated body tokens per chunk after rebalance |
| `--skip-api-key-check` | | Skip API key presence check |
| `--glossary` | `-g` | Path to glossary file (YAML map or JSONL `{src,tgt}`) |

## Examples

```bash
# Single file
ask-llm trans document.md

# Directory output
ask-llm trans /path/to/dir/ -o translated/

# Glob pattern with parallel files
ask-llm trans *.md --max-parallel-files 5

# Use a glossary for consistent terminology
ask-llm trans paper.md --glossary glossary.yml

# Custom prompt template
ask-llm trans paper.md -p @prompts/tech-paper-trans.md

# Target language and threads
ask-llm trans file.txt -t en -s zh --threads 10
```
