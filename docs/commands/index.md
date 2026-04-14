# Commands Overview

Ask LLM provides the following commands. Each command is implemented as a Typer subcommand.

| Command | Description |
|---------|-------------|
| [`ask`](./ask) | Send a single prompt to an LLM with file or text input |
| [`chat`](./chat) | Start an interactive chat session |
| [`trans`](./trans) | Translate files using an LLM (Markdown, text, notebooks) |
| [`batch`](./batch) | Process multiple tasks from a YAML configuration |
| [`paper`](./paper) | Explain academic papers section by section |
| [`format`](./format) | Normalize Markdown heading levels using an LLM |
| [`config`](./config) | Manage configuration and test API connections |

## Global Options

All commands inherit these global options from the root CLI:

- `--version`, `-v` — Show version and exit
- `--debug`, `-d` — Enable debug logging
- `--quiet`, `-q` — Suppress non-error output
