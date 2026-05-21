# Changelog

## 2.7.1 (2026-05-21)

### Features

- **Auto-load providers from `providers.yml`**: Provider runtime configuration (base_url, api_key, models, default_model, timeout, etc.) is now automatically loaded from `providers.yml`, eliminating the need to duplicate provider settings in `default_config.yml`.
  - New `_load_providers_yml()` in `loader.py` reads provider runtime fields from the first available `providers.yml` (search order: `ASK_LLM_PROVIDERS_YML` > `./providers.yml` > package root > `~/.config/ask_llm/`).
  - `providers.yml` acts as the single source of truth for provider specs; `default_config.yml` only needs to contain overrides or non-provider settings.
  - Added `siliconflow` provider to `providers.yml` with `deepseek-ai/DeepSeek-V4-Flash` model.

### Fixes

- **Provider-specific default_model resolution**: `ConfigManager.get_default_model()` now correctly returns the current provider's own `default_model` (stored as `models[0]`) instead of always falling back to the global `default_model`. This fixes the bug where `--provider siliconflow` would incorrectly use `deepseek-chat` instead of `deepseek-ai/DeepSeek-V4-Flash`.

### Refactors

- **`src/ask_llm/config/default_config.yml`**: Removed all built-in provider definitions (deepseek, kimi-code, ollama). Now contains only general defaults (translation, batch, file, formatting, paper, etc.).
- **`docs/default_config.example.yml`**: Updated to reflect the new configuration architecture and added documentation on provider auto-loading.
- **`providers.yml`**: Unified provider ID naming (`kimi` → `kimi-code`) and added missing `timeout: 120.0` for kimi-code.

### Contributors

- Feature designed and implemented with assistance from **Kimi CLI** (agent) and **kimi-k2.6** (model).


## 2.7.0 (2026-05-21)

### Features

- **Ollama provider support**: Add full Ollama API integration for local LLM inference.
  - New `ollama` provider in `providers.yml` and `default_config.yml` with `qwen3.6` and `qwen3.5` models.
  - No API key required — Ollama provider skips all API key validation gates.
  - `create_provider_from_config()` factory now correctly routes `"ollama"` to `OllamaProvider`.
  - Fix `base_url` for OpenAI SDK sync path: `/v1` suffix needed for `/v1/chat/completions`.
  - Separate `_get_litellm_api_base()` override returning plain host for LiteLLM async path.
  - `ProviderConfig.api_key` relaxed from required to optional (empty string OK) for keyless providers.
  - `loader.py`: handle YAML `api_key: null` → empty string conversion.
  - Interactive config, `config test`, and API key gate all skip Ollama.
  - Dependencies: llm-engine `>=0.2.2` (factory routing and `_get_litellm_api_base()` fixes).

### Contributors

- Feature designed and implemented with assistance from **Claude Code** (agent) and **deepseek-v4-pro** (model).
- llm-engine routing fixes co-authored with Claude Code.

## 2.6.2 (2026-05-19)

### Features

- **`ask-llm format --type body`**: Add markdown body formatting support via LLM. Splits large markdown files into heading-aware chunks, formats each chunk concurrently, and merges results. Supports configurable `max_chunk_tokens` and `concurrency`.

### Fixes

- **body format chunk merging**: Fix markdown elements (headings, images, blockquotes) being glued to preceding content when multiple chunks are merged. The root cause was a combination of paragraph-splitter boundary stripping, per-chunk `.strip()` on LLM output, and empty-separator join. Fixed by:
  - Replacing `.strip()` with `.rstrip()` in `_process_chunk` to preserve leading whitespace (e.g., code block indentation).
  - Introducing `_join_chunks()` which normalizes newlines and ensures a blank line (`\n\n`) between adjacent chunks.

### Contributors

- Feature designed and bug fixed with assistance from **Kimi CLI** (agent) and **kimi-k2.6** (model).

## 2.6.1 (2026-04-17)

### Fixes

- **DeepSeek JSON mode detection**: Fix incorrect `response_format` injection when prompts contain "JSON" in a negative context (e.g., "不允许直接以整段 JSON 呈现主体内容"). The naive `"json" in content.lower()` check matched any mention of JSON, causing DeepSeek Reasoner models to emit raw JSON with chain-of-thought text instead of Markdown. Fixed in `llm-engine` (v0.2.1) by using explicit pattern matching with negation guards and positive request patterns.

- **translation prompt**: Simplify `tech-paper-trans-compact.md` output format instruction by removing verbose JSON prohibition text.

### Contributors

- Fix implemented with assistance from Claude Code (Anthropic).

## 2.6.0 (2026-04-16)

### Fixes

- **progress bar**: Fix overlapping/conflicting progress display when multiple API calls run concurrently.
  - **Root cause**: All concurrent workers shared a single `main_task` progress bar ID. When workers called `progress.update(main_task, description=...)` simultaneously, their updates raced and overwrote each other, causing garbled/overlapping display.
  - **Fix**: Each worker now gets its own progress task ID via `task_to_progress_id` dict mapping. Each concurrent task has a separate progress bar row that updates independently without conflicts.
  - Added `estimate_output_tokens()` helper function to estimate expected output tokens based on task type and input tokens (`paper_explain` → 2x input, `translation` → 1.1x input).
  - Progress bars now show meaningful percentage based on actual output tokens vs estimated total, with Rich `BarColumn`, `TimeElapsedColumn`, and `TimeRemainingColumn`.

### Contributors

- Fix implemented with assistance from Claude Code (Anthropic).

## 2.5.2 (2026-04-15)

### Fixes

- **trans**: Fix missing/corrupted content in translated markdown files with LaTeX.
  - **Root cause 1**: Translation prompt templates did not instruct LLMs to preserve LaTeX `$...$`/`$$...$$` delimiters or to never omit content. LLMs frequently dropped section headers, image link `!` prefixes, and `$` delimiters from math expressions (e.g. `$\mathcal{V}$` → `{V}$`).
  - **Root cause 2**: Markdown splitter and chunk balancer could split at `$$` display-math boundaries, leaving orphaned equations at the start of chunks that LLMs then garbled.
  - **Root cause 3**: When LLMs ignored "no JSON" instructions and wrapped output in `{"translation": "..."}`, the JSON unwrapper failed to parse it because LaTeX backslashes (`\mathcal`, `\beta`) are invalid JSON escapes — raw JSON leaked into the output file.
  - Added `_PRESERVE_INSTRUCTION` to all built-in prompt templates: explicit LaTeX preservation and no-omission rules.
  - Updated `tech-paper-trans-compact.md` and `tech-paper-trans.md` prompt files with same instructions.
  - `MarkdownTokenSplitter._split_by_paragraphs_binary()`: merge `$$` display-math blocks with their preceding paragraph.
  - `chunk_balance._split_by_token_budget()`: same `$$` merge logic.
  - `translation_exporter._unwrap_translation_payload()`: new fallback `_try_parse_json_with_latex_escapes()` that fixes invalid backslash escapes and literal control characters in JSON string values before re-parsing.

## 2.5.1 (2026-04-14)

### Fixes

- **paper explain**: `resolved_section_prompts` now applies `key_resolution` rules before looking up `prompt_files`. This fixes `KeyError: 'No prompt_files entry for section: extra:contents'` (and similar `extra:*` / `appendices:h2:*` keys) when non-standard headings are processed.

## 2.5.0 (2026-04-07)

### Paper explain pipeline

- **Configurable pipeline** (`paper-explain-pipeline.yml` + bundled `paper-explain-pipeline.defaults.yml`): project file is **merged** with defaults; omit keys to inherit. Single source of truth for `prompt_files`, `section_labels_zh`, `heading_match`, `full_prompts`, and `key_resolution`.
- **Multi-template sections** via `section_prompts` (job keys `section:tpl-stem`) and **merged sections** via `section_combos` (job keys `combo:id:tpl-stem`, optional `output_stem` for filenames).
- **CLI** `--pipeline` and `paper.pipeline_config`; `--sections` filters extended for `combo:` and multi-template keys.
- **`src/ask_llm/prompts`** symlink to repo `prompts/` so packaged defaults resolve in dev and wheels.

### Other

- Markdown formatting and path discovery utilities (`format` command-related); tests and config wiring as in repository history for this release.

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
