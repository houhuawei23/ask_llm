# format

Use an LLM to format Markdown documents. Supports two formatting modes:
- **title** (default): Normalize heading levels
- **body**: Optimize body text formatting (punctuation, whitespace, structure)

Supports single files, globs, and directories with parallel processing.

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
| `--type` | `-T` | Format type: `title` or `body` (default: `title`) |
| `--output` | `-o` | Output file or directory |
| `--config` | `-c` | Configuration file path |
| `--provider` | `-a` | API provider |
| `--model` | `-m` | Model name |
| `--temperature` | `-t` | Sampling temperature (0.0–2.0) |
| `--force` | `-f` | Overwrite existing output files |
| `--inplace` | `-i` | Overwrite source files in place |
| `--heading-batch-size` | | Max headings formatted in a single API call (title only) |
| `--heading-concurrency` | | Concurrent API calls per file for heading batches (title only) |
| `--body-max-chunk-tokens` | | Max tokens per chunk for body formatting (body only) |
| `--body-concurrency` | | Concurrent API calls per file for body chunks (body only) |
| `--prompt` | `-p` | Prompt template file (supports `@` prefix) |
| `--recursive` / `--no-recursive` | | Recurse into subdirectories (default: recursive) |
| `--workers` | `-j` | Parallel file workers (default: based on CPU count) |

## Format Types

### Title Formatting (`--type title`)

Extracts all headings from the Markdown file and uses an LLM to normalize their hierarchy:
- First heading becomes H1 (`#`)
- Numbered headings are assigned appropriate levels based on their numbering depth
- Context-aware batch processing maintains consistency across batch boundaries

### Body Formatting (`--type body`)

Splits the Markdown document into semantic chunks using heading-aware token-based splitting, then uses an LLM to optimize each chunk's formatting:
- Fixes punctuation (mixed Chinese/English punctuation)
- Normalizes whitespace and line breaks
- Preserves all headings, code blocks, math, tables, and lists
- Does not add or remove any substantive content

Chunks are processed concurrently for efficiency, then merged back in original order.

## Examples

```bash
# Format headings in a single file (default)
ask-llm format document.md
ask-llm format document.md --type title

# Format body text
ask-llm format document.md --type body

# Format a directory
ask-llm format ./notes_dir

# Parallel batch with output directory
ask-llm format ./notes_dir -j 8 -o ./formatted_out/

# In-place edit
ask-llm format doc.md --inplace

# Custom prompt for heading format
ask-llm format doc.md -p @prompts/custom-heading-format.md

# Body format with custom chunk size and concurrency
ask-llm format long_doc.md --type body --body-max-chunk-tokens 3000 --body-concurrency 8
```
