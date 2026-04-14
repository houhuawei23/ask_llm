# format

Use an LLM to normalize Markdown heading levels. Supports single files, globs, and directories with parallel processing.

## Usage

```bash
ask-llm format [FILES...] [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `FILES...` | Markdown files, directories, or glob patterns |

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file or directory |
| `--config` | `-c` | Configuration file path |
| `--provider` | `-a` | API provider |
| `--model` | `-m` | Model name |
| `--temperature` | `-t` | Sampling temperature (0.0–2.0) |
| `--force` | `-f` | Overwrite existing output files |
| `--inplace` | `-i` | Overwrite source files in place |
| `--heading-batch-size` | | Max headings formatted in a single API call |
| `--heading-concurrency` | | Concurrent API calls per file for heading batches |
| `--prompt` | `-p` | Prompt template file (supports `@` prefix) |
| `--recursive` / `--no-recursive` | | Recurse into subdirectories (default: recursive) |
| `--workers` | `-j` | Parallel file workers (default: based on CPU count) |

## Examples

```bash
# Format a single file
ask-llm format document.md

# Format a directory
ask-llm format ./notes_dir

# Parallel batch with output directory
ask-llm format ./notes_dir -j 8 -o ./formatted_out/

# In-place edit
ask-llm format doc.md --inplace

# Custom prompt
ask-llm format doc.md -p @prompts/custom-heading-format.md
```
