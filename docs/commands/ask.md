# ask

Send a single request to an LLM API with file or direct text input.

## Usage

```bash
ask-llm ask [INPUT] [OPTIONS]
```

The `ask` command can also be invoked without the explicit `ask` subcommand when providing a positional argument:

```bash
ask-llm "What is the capital of France?"
ask-llm input.txt
```

## Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Input file path or direct text |

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input file path (alternative to positional argument) |
| `--output` | `-o` | Output file path (default: auto-generated) |
| `--prompt` | `-p` | Prompt template file or text (use `{content}` placeholder) |
| `--provider` | `-a` | API provider to use |
| `--model` | `-m` | Model name to use |
| `--temperature` | `-t` | Sampling temperature (0.0–2.0) |
| `--config` | `-c` | Configuration file path |
| `--force` | `-f` | Overwrite existing output file |
| `--metadata` | | Include metadata in output |
| `--stream` / `--no-stream` | | Stream response to console (default: stream) |
| `--skip-api-key-check` | | Skip API key presence check |
| `--system` | `-s` | System prompt for the LLM |
| `--include-reasoning` | | Include reasoning content from reasoner models |
| `--dry-run` | `-n` | Preview prompt and token estimate without making an API call |

## Examples

```bash
# Direct text
ask-llm "Translate to Chinese: Hello world"

# File input
ask-llm input.txt

# Custom output and model
ask-llm input.md -o output.md -m gpt-4

# System prompt for one-shot behavior control
ask-llm "What are you?" --system "You are a pirate. Respond in pirate dialect."

# Show reasoning from a reasoner model
ask-llm "Solve: 12*15" --include-reasoning -m deepseek-reasoner

# Dry-run to preview the prompt and estimated tokens
ask-llm input.md --dry-run
```
