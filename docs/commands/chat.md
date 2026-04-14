# chat

Start an interactive chat session with an LLM.

## Usage

```bash
ask-llm chat [OPTIONS]
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input file for initial context |
| `--prompt` | `-p` | Prompt template for initial context |
| `--system` | `-s` | System prompt |
| `--provider` | `-a` | API provider to use |
| `--model` | `-m` | Model name to use |
| `--temperature` | `-t` | Sampling temperature (0.0–2.0) |
| `--config` | `-c` | Configuration file path |
| `--skip-api-key-check` | | Skip API key presence check |

## Meta Commands

While in chat mode, type any of the following commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/info` | Show session information |
| `/config` | Show current configuration |
| `/providers` | List available providers |
| `/models` | List models for the current provider |
| `/model <name>` | Switch to a different model |
| `/history` | Show conversation history summary |
| `/save <file>` | Save history to a JSON file |
| `/search <pattern>` | Search message history |
| `/export <file> [json\|md\|txt]` | Export history |
| `/clear` | Clear conversation history |
| `/system <text>` | Show or set the system prompt |
| `/clear-system` | Clear the system prompt |
| `!command` | Execute a shell command |
| `!!` | Repeat the last shell command |

## Examples

```bash
# Start an interactive session
ask-llm chat

# With initial context
ask-llm chat -i context.txt

# With a system prompt
ask-llm chat -s "You are a helpful coding assistant"

# Using a specific model
ask-llm chat -m deepseek-reasoner
```
