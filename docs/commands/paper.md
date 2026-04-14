# paper

Explain an academic paper by splitting Markdown by headings (or loading an arxiv2md-beta directory) and calling an LLM per section. Results are written to an `explain/` subdirectory next to the input.

## Usage

```bash
ask-llm paper -i <PATH> [OPTIONS]
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Path to a paper `.md` file or an arxiv2md-beta output directory (required) |
| `--run` | `-r` | `sections` (per-section + meta), `full` (whole paper), or `all` (default: `all`) |
| `--sections` | `-s` | Comma-separated section keys to run |
| `--provider` | `-a` | API provider |
| `--model` | `-m` | Model name |
| `--temperature` | `-t` | Sampling temperature (0.0–2.0) |
| `--config` | `-c` | Configuration file path |
| `--force` | `-f` | Overwrite existing `explain/*.md` files |
| `--metadata` | | Include token/latency metadata in each output file |
| `--concurrency` | `-j` | Parallel LLM calls (default: `paper.concurrency` in config) |
| `--dry-run` | `-n` | Show detected sections and token estimates without making API calls |
| `--resume` | | Skip sections whose output already exists |
| `--pipeline` | | Path to `paper-explain-pipeline.yml` |
| `--providers-pricing` | | Path to `providers.yml` for cost estimates |

## Examples

```bash
# Run everything
ask-llm paper -i paper.md --run all

# Only section-level explanations
ask-llm paper -i paper.md --run sections

# Dry-run to preview
ask-llm paper -i paper.md --dry-run

# Resume an interrupted run
ask-llm paper -i paper.md --resume

# Limit concurrency
ask-llm paper -i paper.md -j 8
```

## Output

Results are saved under `<input_dir>/explain/` or next to the `.md` file. Files are numbered in document order, for example:

- `0-meta.explain.md`
- `1-abstract.explain.md`
- `N-full.explain.md`
