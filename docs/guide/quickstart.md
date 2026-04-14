# Quick Start

This page covers common workflows to get you productive with Ask LLM quickly.

## Ask a Simple Question

```bash
# Direct text input
ask-llm "What is the capital of France?"

# Process a file
ask-llm input.txt

# Save output to a file
ask-llm input.txt -o output.txt

# Use a custom system prompt
ask-llm "Solve: 12*15" --system "You are a math tutor." -m deepseek-reasoner --include-reasoning
```

## Interactive Chat

```bash
# Start a chat session
ask-llm chat

# With initial context and system prompt
ask-llm chat -i context.txt -s "You are a helpful coding assistant"
```

Inside chat mode, you can use meta commands such as `/help`, `/models`, `/save`, `/export`, and `!command` to execute shell commands.

## Translate Documents

```bash
# Single file
ask-llm trans document.md

# Entire directory
ask-llm trans ./posts/ -o ./translated/

# With glossary for consistent terminology
ask-llm trans paper.md --glossary glossary.yml
```

## Explain a Paper

```bash
# Section-by-section explanation
ask-llm paper -i paper.md --run all

# Dry-run to preview sections and token estimates
ask-llm paper -i paper.md --dry-run

# Resume an interrupted run
ask-llm paper -i paper.md --resume
```

## Batch Processing

```bash
# Run a batch configuration
ask-llm batch batch-examples/prompt-contents.yml

# With custom output and concurrency
ask-llm batch config.yml -o results.json -f json --threads 10 --retries 5
```

## Format Markdown Headings

```bash
# Format a single file
ask-llm format document.md

# Format all Markdown files in a directory
ask-llm format ./docs/ -o ./formatted/
```
