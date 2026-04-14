# batch

Process batch tasks from a YAML configuration file. Supports running the same prompt across multiple contents or multiple prompt-content pairs across one or more models.

## Usage

```bash
ask-llm batch <CONFIG_FILE> [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `CONFIG_FILE` | Batch configuration file path (YAML format) |

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file or directory path |
| `--format` | `-f` | Output format: `json`, `yaml`, `csv`, or `markdown` |
| `--threads` | `-t` | Number of concurrent threads |
| `--retries` | `-r` | Maximum number of retries |
| `--config` | `-c` | Configuration file path |
| `--separate-files` | | Save results in separate files per model |
| `--split` | | Split results into separate files (one per task, content only) |
| `--skip-api-key-check` | | Skip API key presence check |

## Configuration Formats

### Prompt + Contents

Use the same prompt for multiple contents:

```yaml
provider-models:
  - provider: deepseek
    models:
      - model: deepseek-chat
      - model: deepseek-reasoner
        temperature: 1.0
prompt: You are a helpful assistant.
contents:
  - "What is the capital of France?"
  - "What is the capital of Germany?"
```

### Prompt-Content Pairs

Use YAML multi-documents for different prompts per task:

```yaml
---
prompt: Translate to French
content: Hello world
---
prompt: Summarize this
content: The quick brown fox...
```

## Examples

```bash
# Basic usage
ask-llm batch batch-examples/prompt-contents.yml

# Custom output format
ask-llm batch config.yml -o results.json -f json

# More threads and retries
ask-llm batch config.yml --threads 10 --retries 5

# Separate files per model
ask-llm batch config.yml --separate-files -o results/
```
