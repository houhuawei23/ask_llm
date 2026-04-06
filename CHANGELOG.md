# Changelog

## 2.4.0 (2026-04-06)

### New Features

- **`ask --system`** - Add system prompt support for one-shot queries. Inject a system message to guide LLM behavior.
  ```bash
  ask-llm ask "What are you?" --system "You are a pirate. Respond in pirate dialect."
  ```

- **`ask --include-reasoning`** - Surface chain-of-thought reasoning from reasoner models (e.g., DeepSeek).
  ```bash
  ask-llm ask "Solve: 12*15" --include-reasoning -m deepseek-reasoner
  ```

- **`ask --dry-run`** - Preview prompt and token/cost estimate without making API calls.
  ```bash
  ask-llm ask input.md --dry-run
  ```

- **`paper --dry-run`** - Preview detected sections with token counts before expensive multi-job runs.
  ```bash
  ask-llm paper -i paper.md --dry-run
  ```

- **`paper --resume`** - Skip already-completed sections when resuming interrupted runs.
  ```bash
  ask-llm paper -i paper.md --resume
  ```

- **`trans --glossary`** - Inject terminology pairs for domain-specific translations.
  ```bash
  ask-llm trans paper.md --glossary glossary.yml
  ```
  Supports YAML (`{src: tgt}` or `[{src: ..., tgt: ...}]`) and JSONL formats.

- **`chat /search`** - Search message history with regex pattern.
  ```
  /search attention
  ```

- **`chat /export`** - Export conversation history to JSON, Markdown, or plain text.
  ```
  /export session.md
  /export session.txt txt
  /export session.json json
  ```

### Implementation Details

- Added `system_prompt` parameter to `process_with_metadata()` and `process()` in `processor.py`
- Added `return_reasoning` support for streaming responses via `iter_process_raw_stream()`
- Added `glossary_pairs` support to `Translator` class with `load_glossary()` static method
- Enhanced chat meta-commands with `/search` and `/export` functionality

## 2.3.0 (2026-04-04)

### Performance Improvements

- **Fixed O(N²) token counting in streaming loops**: `BatchProcessor` and `GlobalBatchProcessor` now count tokens incrementally per chunk instead of re-encoding the entire accumulated response on every stream chunk. This eliminates severe slowdowns on long LLM outputs.
- **Reused provider adapter instances across tasks**: `GlobalBatchProcessor` pre-builds a provider cache before entering the thread pool, cutting per-task connection setup overhead.
- **Improved progress bar UX for large batches**: Replaced "one progress bar per task" with a single overall progress bar, keeping the terminal usable for 100+ task batches.

### Architecture & Quality Improvements

- **Eliminated thread-safety hazard in `GlobalBatchProcessor`**: Worker threads no longer mutate the shared `ConfigManager`. Provider configs are resolved upfront in single-threaded code.
- **Replaced brittle reasoning tuple heuristic with explicit `ReasoningChunk`**: The streaming pipeline now uses a `NamedTuple` for reasoning content, improving type safety and readability.
- **Refactored `batch_processor.py` streaming logic**: Extracted `_stream_and_collect()` helper to remove ~80 lines of near-duplicate code between translation and paper-explain handlers.
- **Fixed provider deep-merge behavior**: User config can now override fields for a single provider without losing all default providers from the package config.
- **Removed dangerous fallback `base_url`**: `loader.py` now raises a clear `ValueError` instead of silently falling back to `https://api.example.com/v1`.
- **Migrated `UnifiedConfig` to native Pydantic parsing**: Removed manual `from_dict()` in favor of `UnifiedConfig.model_validate()`.
- **Hardened `BatchResultExporter` filename sanitization** and fixed a divide-by-zero risk in Markdown statistics.

### Contributors

- Code review and improvements by Claude Code.
