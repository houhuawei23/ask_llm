# Changelog

## 2.16.12 (2026-07-15)

P2 continuation — complete `get_config()` de-globalization. All remaining raising `get_config()` call sites now fall back to built-in defaults, so the core library can be used without an active CLI config. Also fixed a latent B10 infinite-loop bug exposed by the fallback tests.

### Changed

- **P2 — remaining `get_config()` sites migrated to `get_config_or_none()` with defaults.** Replaced raising calls in:
  - `utils.file_handler` (`chunk_size`, `tqdm_ncols`, `default_output_suffix`)
  - `core.format_markdown_file` (`formatted_suffix`)
  - `core.md_heading_formatter` (`format_heading` defaults)
  - `core.processor` (`default_prompt_template`)
  - `core.md_body_formatter` (`format_body` defaults)
  - `core.text_splitter` (`max_chunk_size`)
  - `services.format_service` (`formatted_suffix`)
  - `core.paper_explain_pipeline`, `core.paper_explain`, `utils.prompt_resolver` (`project_root_markers`)
  Each module now carries private `_DEFAULT_*` constants matching `default_config.yml`, so the code works as a library even when `set_config()` was never called. `config.context.get_config()` is retained for callers that truly require an active config.

### Fixed

- **B10 — latent infinite loop in `_write_with_progress` for multibyte text.** The previous byte-total / char-increment mismatch could make `written` (chars) lag `total` (bytes) and slice an empty chunk forever. The writer now slices by characters (preserving UTF-8 boundaries) while advancing a separate byte counter, so the loop always terminates and the progress bar stays accurate. Added safety `break` on empty chunk.
- **Pre-existing RUF012 in `text_splitter.py`** — `HEADING_LEVELS` is now annotated `ClassVar[list[int]]`.

### Tests

- Updated `tests/unit/test_format_service.py` patch targets from `get_config` to `get_config_or_none`. Full suite: 415 passed, 1 skipped.

### Version

- 2.16.11 → 2.16.12

## 2.16.11 (2026-07-14)

B10 + B11 — last two correctness bugs from ARCHITECTURE_REVIEW.md §5. **All 11 load-bearing bugs are now fixed.**

### Fixed

- **B10 — write-progress bar overshot 100% on multibyte text.** `FileHandler._write_with_progress` set the bar `total` to the *character* count but incremented by UTF-8 *byte* length, so CJK text drove the bar past 100%. Total is now the byte length (`len(content.encode("utf-8"))`), matching the increments. (The read path was already correct — byte total from `stat().st_size`.) §4.5.7 / B10.
- **B11 — silent checkpoint residue on full success.** `format_service.resume_from_checkpoint` wrapped `os.remove(checkpoint)` in `except OSError: pass`, so a failed removal was invisible while the user saw "全部完成". It now emits a warning naming the leftover checkpoint path (and that it can be deleted manually). §5 / B11.

### Tests

- Added `test_write_progress_total_is_bytes_for_multibyte` (B10) and `test_resume_body_checkpoint_remove_failure_warns` (B11). 415 passed, 1 skipped.

### Version

- 2.16.10 → 2.16.11

## 2.16.10 (2026-07-14)

P2.6 — relocate `paper_explain_pipeline` out of `config/`. Internal module move; no CLI surface change.

### Changed

- **P2.6 — `paper_explain_pipeline.py` moved `config/` → `core/`.** The 453-LOC module is paper-explain domain logic (7 nested pydantic models, pipeline parsing), not configuration — it lived in `config/` only because it reads a YAML at startup. Now at `ask_llm.core.paper_explain_pipeline`. Its lazy `get_config()` read is unchanged. Importers (`core/paper_explain`, `services/paper_service`, two tests) retargeted. ARCHITECTURE_REVIEW.md §4.2.6.

### Version

- 2.16.9 → 2.16.10

## 2.16.9 (2026-07-14)

P2 (config de-globalization, incremental) — `TokenCounter` no longer requires a loaded config. Library/embedding hardening.

### Fixed

- **`TokenCounter._get_encoding` no longer crashes without a config.** The hot path called `get_config()` (which raises `RuntimeError` when no config is loaded) for empty / unknown model names. It now uses `get_config_or_none()` and falls back to `cl100k_base` (the project's common default across `ENCODING_MAP`), so token counting is usable in programmatic contexts that never call `set_config`. ARCHITECTURE_REVIEW.md §4.2.3.

### Tests

- Added `test_get_encoding_falls_back_when_no_config`. Fixed a pre-existing `C416` lint in the B2 approximate-token test. 413 passed, 1 skipped.

### Version

- 2.16.8 → 2.16.9

## 2.16.8 (2026-07-14)

P2.7 — surface silent duplicate env-var mappings. First P2 (config) step. No CLI surface change.

### Fixed

- **P2.7 — conflicting env overrides now warn.** Two env vars can map to the same config key (e.g. `ASK_LLM_TRANSLATION_THREADS` and `ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS` both → `translation.max_concurrent_api_calls`). Previously the apply loop overwrote silently in dict-iteration order, so which one won was accidental. `_apply_env_overrides` now detects when multiple *set* env vars target the same key and logs a warning naming the winner (last in `ENV_TO_CONFIG` order). Behaviour is otherwise unchanged (last still wins); the warning just makes it visible.

### Added

- `ask_llm.config.loader._duplicate_env_targets` and `_warn_conflicting_env_overrides`.

### Tests

- Added `test_conflicting_env_overrides_warns_and_last_wins` and `test_single_env_override_does_not_warn_conflict`. 412 passed, 1 skipped.

### Version

- 2.16.7 → 2.16.8

## 2.16.7 (2026-07-14)

P1.3 (substantial) — extract `TaskExecutor`. Internal refactor; no CLI surface change. Third and largest step of the `GlobalBatchProcessor` god-class split.

### Added

- `ask_llm.core.task_executor.TaskExecutor` — executes a single provider/model attempt: rate-limit acquire, adapter lookup, streaming collection (via `stream_and_collect`), `RequestMetadata` construction, progress updates, and the batch-wide auth-error log de-duplication. Holds the `verbose` / `stream_api` / auth-error state previously on the god class.
- `paper_request_timeout_seconds` and `update_global_task_progress_failed` moved into `task_executor` (consumed there; the former is also used by the provider cache) to avoid a circular import.

### Changed

- `GlobalBatchProcessor` shrank to a lean coordinator: 720 → 347 LOC (834 → 347 across all three P1.3 steps). `_process_single_global_task` (the B1 escalation) now delegates one-config attempts to `self._task_executor.try_run_with_config`. A pass-through `_auth_error_logged` property preserves the `translation_service` inspection.
- `provider_manager` now lazy-imports `paper_request_timeout_seconds` from `task_executor`.

### Tests

- Added `test_task_executor.py` (rate-limit-timeout failure, auth-error dedup flag, missing-provider failure). Retargeted `test_batch_processor` patches to `ask_llm.core.task_executor` (the moved code). 410 passed, 1 skipped.

### Version

- 2.16.6 → 2.16.7

## 2.16.6 (2026-07-14)

P1.3 (partial) — extract `ProgressPresenter`. Internal refactor; no CLI surface change. Second step of the `GlobalBatchProcessor` god-class split.

### Added

- `ask_llm.core.progress_presenter.ProgressPresenter` (+ `NullProgressPresenter`) — owns the `rich.Progress` instance, the per-task display metadata, and the per-worker-slot bar pool (B6). The worker acquires a slot (relabels the bar), runs, and releases it. `NullProgressPresenter` is a no-op used when progress display is disabled.

### Changed

- `GlobalBatchProcessor.process_global_tasks` no longer builds the `Progress` / slot pool inline; it constructs a presenter and calls `acquire`/`release`/`start`/`stop`. God class shrank 771 → 720 LOC (834 → 720 across both P1.3 steps). Dropped now-unused imports (`queue`, the inline `rich.progress`/`rich.console` lazy imports).

### Tests

- Added `test_progress_presenter.py` (one bar per slot, acquire relabels + release returns slot, null presenter no-op). Updated the B6 + B1 integration tests to patch `ask_llm.core.progress_presenter.Progress` (the construction moved out of `batch_processor`). 410 passed, 1 skipped.

### Version

- 2.16.5 → 2.16.6

## 2.16.5 (2026-07-14)

P1.3 (partial) — extract `StreamCollector`. Internal refactor; no CLI surface change. First step of the `GlobalBatchProcessor` god-class split (ARCHITECTURE_REVIEW.md §7.2 / P1.3).

### Added

- `ask_llm.core.stream_collector.stream_and_collect(...)` — the single streaming + token-collection implementation, extracted out of `GlobalBatchProcessor._stream_and_collect`. Pure function (no instance state); independently unit-testable. The unused `task` parameter was dropped.

### Changed

- `GlobalBatchProcessor` shrank 834 → 771 LOC. The paper and translation-chunk runners now call the module-level `stream_and_collect`. Dropped now-unused imports (`time`, `PROGRESS_UPDATE_INTERVAL`, `Iterator`, `ReasoningChunk`).

### Tests

- Added `test_stream_collector.py` (plain-text concatenation, reasoning separation, progress throttling). 404 passed, 1 skipped.

### Version

- 2.16.4 → 2.16.5

## 2.16.4 (2026-07-14)

P1.7 / B5 — checkpoint survives Ctrl-C. Fixes the "resumable on interrupt" gap (ARCHITECTURE_REVIEW.md §4.1.4 / B5).

### Fixed

- **B5 — Ctrl-C no longer discards all batch progress.** `BoundedRetryRunner` now installs a SIGINT handler (main thread only): on the first Ctrl-C it sets an interrupt flag, stops scheduling new tasks, drains the in-flight tasks to completion, and returns the partial results instead of re-raising `KeyboardInterrupt`. The prior handler is restored immediately, so a second Ctrl-C hard-interrupts. New `RunMetrics.interrupted` flag signals the partial run.
- **`batch` and `trans` services keep the checkpoint on interrupt.** Previously the post-run unlink (`if not new_failed: unlink`) could delete the checkpoint after a partial interrupted run. Both services now check `last_metrics.interrupted`: on interrupt the checkpoint is kept and a resume hint is printed; unlink happens only on a clean, fully-successful run. Existing resume logic already handles partial result sets, so interrupted tasks are re-run on `--resume`.

### Tests

- Added `test_sigint_returns_partial_results_and_drains_inflight` (simulates Ctrl-C mid-run, asserts `interrupted` + partial results + drained in-flight) and `test_normal_run_not_marked_interrupted`. 401 passed, 1 skipped.

### Version

- 2.16.3 → 2.16.4

## 2.16.3 (2026-07-14)

P1.6 — unify `RequestMetadata` construction (B8 root-cause finish). Internal refactor; no CLI surface change.

### Added

- `RequestMetadata.from_execution(...)` classmethod — single factory for the per-request metadata that `batch_processor` (paper + translation-chunk paths) and `processor` (single `ask`) build on every successful call.

### Changed

- **P1.6 — collapsed three duplicate `RequestMetadata(...)` construction sites.** The temperature-resolution ternary (`temperature if temperature is not None else provider.config.api_temperature`) — the exact code path that caused the v2.15.1 adapter dict-vs-object crash — now lives in one place. Per-site differences (how output words/tokens are computed) remain at the call sites.

### Version

- 2.16.2 → 2.16.3

## 2.16.2 (2026-07-14)

P1.1 / B1 — unify retry × fallback into a single shared-budget escalation. Fixes the highest-severity design defect in `docs/ARCHITECTURE_REVIEW.md` (§4.1.3).

### Fixed

- **B1 — retry × fallback call amplification.** The bounded runner and the in-worker fallback chain were two uncoordinated retry layers: when the runner retried a task, the worker re-walked the *entire* fallback chain from the primary, so each task could make up to `(max_retries + 1) × len(fallback_chain)` API calls (e.g. `max_retries=3` × a 3-config chain → up to 12 calls). The fallback chain and the retry budget now share a single escalation of at most `max_retries + 1` attempts: attempt *k* uses `configs[min(k, len-1)]` — a transient failure advances to the next config (or re-tries the last when the chain is shorter than the budget). Per-task API calls are bounded by the retry budget regardless of chain length. Terminal errors (auth, content filter, validation) still short-circuit immediately.

### Changed

- `GlobalBatchProcessor._process_single_global_task` now performs exactly one config attempt per call (previously walked the whole chain). Flat attempt records are threaded across runner retries via a per-run side dict (`attempt_history_by_task`).
- Terminal failures now saturate `result.retry_count = max_retries` so the runner declines to schedule another attempt (replaces the in-worker `break`).

### Behaviour note

- Multi-config fallback tasks now advance the chain on transient failure instead of re-trying the same provider. **Single-config tasks are unchanged** — they retry the same provider up to `max_retries + 1` times, exactly as before. Net effect: lower API cost and gentler rate-limit pressure for multi-provider batches.

### Tests

- Rewrote `test_fallback_succeeds_when_primary_fails` and `test_all_configs_fail_returns_failed` to drive the new escalation via a `_escalate` helper. Added `test_single_config_retries_same_provider_within_budget` and a runner-level `test_process_global_tasks_bounded_calls_with_fallback_chain` asserting `total_calls <= n_tasks × (max_retries + 1)` (the review's P1 acceptance criterion). 397 passed, 1 skipped.

### Version

- 2.16.1 → 2.16.2

## 2.16.1 (2026-07-14)

P1 execution-engine cleanup (internal refactor; no CLI surface change). Begins the P1 phase of `docs/ARCHITECTURE_REVIEW.md`.

### Added

- `BatchStatistics.from_results(results)` classmethod — single source of truth for per-`(provider, model)` batch-statistics aggregation (P1.5).

### Changed

- **P1.5 — collapsed duplicate statistics aggregators.** `batch_service._calculate_statistics` (a byte-duplicate of `batch_processor.calculate_statistics_by_model`) is deleted; both paths now route through `BatchStatistics.from_results`. `calculate_statistics_by_model` is reduced to a thin delegate for import compatibility.
- **P1.7 — removed a duplicate `TYPE_CHECKING` block** in `batch_processor.py` (the `RateLimitConfig` import was declared twice).

### Notes

- P1.7 dead-code audit: `ProviderRetryRegistry.set`, `BoundedRetryRunner.run`, and the `core/batch.py` re-export shim are each retained — all have callers (`.set` and `.run` are exercised by unit tests; `.run` supports reusing a runner instance across batches; the shim is imported by ~20 modules). `ProviderRetryRegistry.set` will be wired into the runner when the `EscalationPolicy` (P1.1) lands.

### Version

- 2.16.0 → 2.16.1

## 2.16.0 (2026-07-14)

Architecture-review P0 stopgap release — fixes seven load-bearing bugs identified in `docs/ARCHITECTURE_REVIEW.md` and tightens the API-key boundary. No CLI surface changes; one internal data-model type change (`BatchResult.attempt_history`).

### Fixed

- **B2 — CJK provider token counts were silently approximate.** `TokenCounter` mapped DeepSeek/Qwen to `cl100k_base`, which undercounts CJK text; chunk sizing against those context windows could overflow. Now warns once per model and applies a configurable safety factor (`APPROX_TOKEN_SAFETY_FACTOR = 0.85`) in `split_hard_by_max_tokens`. `truncate_to_tokens` fallback unified to word-count.
- **B3 — unresolved `${VAR}` API-key placeholders were silent.** `ProviderConfig.validate_api_key` now warns loudly when an `api_key` still carries an unresolved `${...}` placeholder after YAML load. The run-boundary gate (`api_key_gate`) already blocked this on ask/chat/batch; it is now also wired into `run_global_batch_tasks` (covers trans/paper) via a new pure `ensure_resolved_provider_keys` chokepoint.
- **B4 — body splitter cut fenced code blocks mid-fence.** `MarkdownTokenSplitter` is now code-fence aware: headings inside a fence are not used as split points, and long paragraphs containing fences are split at fence boundaries (each fenced block stays atomic). The full `MarkdownStructure` parser lands in P3.
- **B6 — N tasks produced N live progress bars.** `GlobalBatchProcessor` now uses a pool of `min(max_workers, num_tasks)` per-worker-slot bars instead of one bar per pending task. O(workers) bars regardless of batch size.
- **B7 — `BatchResult.attempt_history` was self-referential.** Changed to `list[AttemptRecord]` (flat, acyclic by construction); the v2.15.1 circular-reference crash class is now structurally impossible. New `AttemptRecord.from_result` factory; the manual slice/cycle-guard hacks were removed.
- **B8 — `ProviderAdapterCache.get` was untyped and accepted dicts.** Typed `(config: ProviderConfig | dict) -> LLMProviderProtocol`; the dict path (root of the v2.15.1 adapter dict-vs-object crash) is rebuilt into a `ProviderConfig` and emits `DeprecationWarning`; bad inputs raise `TypeError`.
- **B9 — rate-limit acquire timeout was hardcoded 60s.** New `ProviderRateLimitConfig.acquire_timeout_seconds` (per provider/model); the failure message points at the knob.
- **Secrets — stale credentials lingered after key rotation.** The interactive gate now calls `ProviderAdapterCache.clear()` after applying a rotated key. Full `SecretStr` migration deferred to P2.

### Added

- `APPROX_TOKEN_SAFETY_FACTOR` constant.
- `ProviderRateLimitConfig.acquire_timeout_seconds` field (+ `default_config.yml` docs).
- `ask_llm.utils.api_key_gate.UnresolvedAPIKeyError` and `ensure_resolved_provider_keys`.
- `ask_llm.core.batch_models.AttemptRecord` (moved from `execution_report`; re-exported for compatibility).
- Tests: `test_api_key_gate.py`, `test_markdown_token_splitter.py`; expanded `test_models.py`, `test_utils.py`, `test_rate_limiter.py`, `test_provider_cache.py`, `test_batch_processor.py`, `test_execution_report.py`.

### Changed

- `BatchResult.attempt_history` type changed from `list[BatchResult]` to `list[AttemptRecord]`. External code reading it as `BatchResult`s must switch to the flat fields (`provider`, `model`, `status`, …).

### Version

- 2.15.1 → 2.16.0

### Contributors

- Refactored with assistance from **Claude** (agent).

## 2.15.1 (2026-06-24)

### Fixed

- **`ask-llm trans` 崩溃：`'dict' object has no attribute 'api_temperature'`**。
  - 根因：`ProviderAdapterCache` 为了复用连接，把配置拆成原始字段后重新拼成普通 `dict` 传给 `llm-engine`，但返回的 adapter 的 `config` 也变成了 `dict`；而 `BatchProcessor` / `GlobalBatchProcessor` / `RequestProcessor` 在构造 `RequestMetadata` 时按对象属性访问 `provider.config.api_temperature`，导致翻译/批处理在 API 成功返回后崩溃。
  - 修复：`ProviderAdapterCache` 在创建 adapter 前先组装成真正的 `ProviderConfig` 对象，再传给 `create_provider_adapter`，使 `adapter.config` 保持对象语义，与代码其它路径一致。
- **`ask-llm trans` 保存 checkpoint 时崩溃：`Circular reference detected (id repeated)`**。
  - 根因：`GlobalBatchProcessor._process_single_global_task()` 在设置成功/失败结果的 `attempt_history` 时，把包含结果自身的列表直接赋给 `attempt_history`，形成循环引用；Pydantic `model_dump(mode="json")` 在序列化 checkpoint 时拒绝这种结构。
  - 修复：`attempt_history` 只记录**当前结果之前的尝试**；成功结果保存前面失败的尝试，最终失败结果 likewise 排除自身，保证对象图无环。

### Tests

- 更新 `tests/unit/test_provider_cache.py`：测试配置改用字典，避免 `MagicMock` 对象触发 `ProviderConfig` 校验失败。
- 更新 `tests/unit/test_batch_processor.py::test_build_provider_cache_includes_fallbacks`：使用真实的 `ProviderConfig` 作为 `get_provider_config` 的返回值。

### Version

- 2.15.0 → 2.15.1

### Contributors

- Fixed with assistance from **kimi-code** (agent) and **kimi-k2.7** (model).

## 2.15.0 (2026-06-24)

### Features

- **Performance optimizations (Phase G)**:
  - New `ask_llm.utils.provider_cache.ProviderAdapterCache`: process-wide LRU cache for llm-engine provider adapters so repeated batch/trans/paper runs reuse the same HTTP client and warm connections.
  - `GlobalBatchProcessor` and batch model validation now use the shared adapter cache.
  - JSON batch exports use `json.JSONEncoder.iterencode()` and stream directly to disk, reducing peak memory for large result sets.
  - Stream collection in `GlobalBatchProcessor` avoids an unconditional `str(chunk)` conversion when chunks are already strings.

### Tests

- Added `tests/unit/test_provider_cache.py` for cache hit/miss, clear, and dict/object config support.
- Added `tests/benchmarks/test_performance.py` with benchmarks for provider-adapter cache access and JSON export (streaming vs `json.dumps`).
- Updated existing tests to patch the cached adapter creation path.

### Documentation

- Updated `README.md` to v2.15.0, added observability/fallback/checkpoint features, `diagnose` command examples, and Contributors section.
- Updated `AGENTS.md` project structure to include `telemetry.py`, `execution_report.py`, `provider_cache.py`, and `diagnose` command; added Observability and Performance section.

### Version

- 2.14.0 → 2.15.0

### Contributors

- Designed and implemented with assistance from **kimi-code** (agent) and **kimi-k2.7** (model).

## 2.14.0 (2026-06-24)

### Features

- **Structured observability (Phase D)**:
  - New `ask_llm.core.telemetry` module with `LogContext`, `ErrorCategory`, `classify_error()`, and `should_fallback_for_error()`.
  - `GlobalBatchProcessor` and `BatchProcessor` now inject structured context into all task logs.
  - `BatchResult` gains `error_category` and `attempt_history` for transparent fallback tracking.
  - Fallback chains stop early for terminal error categories (`authentication`, `content_filter`, `validation_error`).
  - New global `--log-format json` option outputs machine-parseable Loguru JSON logs.
- **Execution reports (`--report`)**:
  - `ask-llm batch --report report.json` exports a structured JSON report.
  - `ask-llm trans --report report.json` exports per-chunk attempt histories.
  - `ask-llm paper --report report.json` exports per-section attempt histories.
  - New `ask_llm.core.execution_report` module with `ExecutionReport`, `TaskRecord`, and `AttemptRecord`.
- **New `ask-llm diagnose` command**:
  - Summarizes any execution report: success rate, token usage, provider/model breakdown, failure categories, and failed task details.
  - Warns when failures are caused by terminal error categories where fallback cannot help.

### Changed

- `BatchService`, `TranslationService`, and `PaperService` all support `export_report()`.
- `NotebookTranslator` exposes `last_results` so translation reports can include notebook chunk attempts.
- `console.setup()` accepts a `log_format` parameter for text or JSON log sinks.

### Tests

- Added `tests/unit/test_telemetry.py` for error classification and log context.
- Added `tests/unit/test_execution_report.py` for report generation and serialization.
- Added `tests/unit/test_diagnose_cli.py` for the diagnose command.
- Extended `tests/unit/test_batch_processor.py` with authentication-error fallback termination.

### Version

- 2.13.0 → 2.14.0

### Contributors

- Designed and implemented with assistance from **kimi-code** (agent) and **kimi-k2.7** (model).

## 2.13.0 (2026-06-24)

### Features

- **Provider fallback chain**: tasks can now retry with alternate providers/models when the primary fails.
  - `BatchTask` gained `fallback_model_configs` for an ordered fallback chain.
  - `ProviderConfig` gained `fallback_to` with `FallbackConfig` entries.
  - New `ask_llm.utils.provider_router.build_fallback_chain()` resolves fallbacks from app config.
- **`GlobalBatchProcessor` fallback execution**: `_process_single_global_task` now tries the primary config and each fallback until one succeeds.
- **`--fallback/--no-fallback` CLI flag**: added to `ask-llm batch`, `ask-llm trans`, and `ask-llm paper` (default: enabled).

### Changed

- `BatchService.run_batch_from_config` accepts `use_fallback` and populates task fallback chains.
- `TranslationService` and `PaperService` accept `app_config` and populate fallback chains for text/markdown and notebook translation tasks.
- `NotebookTranslator` accepts `fallback_configs` and applies them to notebook chunk tasks.
- `build_paper_explain_task` accepts `fallback_model_configs`.

### Fixed

- Resolved a top-level circular import between `translation_service.py` and `cli.app` by moving CLI helper imports into `translate_files()`.

### Tests

- Added `tests/unit/test_batch_processor.py` for fallback execution paths.
- Added `tests/unit/test_translation_service.py` for translation fallback wiring.
- Extended `tests/unit/test_batch_service.py` with fallback chain application tests.

### Version

- 2.12.0 → 2.13.0

### Contributors

- Designed and implemented with assistance from **kimi-code** (agent) and **kimi-k2.7** (model).

## 2.12.0 (2026-06-24)

### Features

- **Generic checkpoint framework**: added `BaseCheckpoint` and `BatchCheckpoint` for resumable batch/translation workflows.
- **`ask-llm batch --resume`**: batch runs now persist a checkpoint (`<config>.checkpoint.json`) and can resume from it, skipping completed tasks.
- **`ask-llm trans --resume`**: per-file translation checkpoints (`<output>.trans_checkpoint.json`) allow resuming long document translations from failed chunks.

### Changed

- `BatchService` manages checkpoint save/load/merge during batch runs.
- `TranslationService` manages per-file checkpoints during text/markdown translation.
- Checkpoints are saved atomically via a temporary file and `replace()`.

### Tests

- Added `tests/unit/test_checkpoint.py` for generic checkpoint persistence.
- Added `tests/integration/test_batch_checkpoint.py` for batch resume behavior.
- Added `tests/integration/test_trans_checkpoint.py` for translation resume behavior.

### Version

- 2.11.0 → 2.12.0

### Contributors

- Designed and implemented with assistance from **kimi-code** (agent) and **kimi-k2.7** (model).

## 2.11.0 (2026-06-24)

### Features

- **Complete Service layer**: added `AskService`, `BatchService`, and `FormatService` under `ask_llm.services/`, moving single-request, batch export/statistics, and format resume logic out of the CLI commands.

### Changed

- `ask.py` business logic moved to `AskService`; CLI now focuses on argument parsing, streaming UX, and exit codes.
- `batch.py` export/statistics printing moved to `BatchService` (`BatchExportResult`, `print_statistics`, `export_results`).
- `format_cmd.py` `--resume` handling moved to `FormatService.resume_from_checkpoint()`.
- Services no longer call `typer.Exit`; they raise `ValueError` / `RuntimeError` and let CLI commands convert to user-facing messages and exit codes.
- Updated `AGENTS.md` Service layer documentation with `AskService` example and exception contract.

### Tests

- Added `tests/unit/test_ask_service.py` covering input loading, prompt normalization, dry-run, and output path resolution.
- Added `tests/unit/test_batch_service.py` covering statistics printing and single/separate/split export modes.
- Added `tests/unit/test_format_service.py` covering body checkpoint resume and title-mode rejection.

### Version

- 2.10.0 → 2.11.0

### Contributors

- Designed and implemented with assistance from **kimi-code** (agent) and **kimi-k2.7** (model).

## 2.10.0 (2026-06-24)

### Features

- **Service layer extraction**: introduced `TranslationService` and `PaperService` under `ask_llm.services/`, moving core orchestration out of `trans.py` and `paper.py` CLI modules.
- **Unified provider/model resolution**: added `resolve_provider_and_model_or_exit()` in `ask_llm.config.cli_session`, used by `ask`, `chat`, `trans`, and `paper` commands.
- **Runner metrics**: `BoundedRetryRunner.run_with_metrics()` returns `RunMetrics` (total/successful/failed/retried, latency), surfaced through `GlobalBatchProcessor.last_metrics` and translation session totals.
- **Rate-limiter visibility**: `GlobalRateLimiter.acquire()` now logs a warning when it waits for a token, including configured RPM/burst and a tuning hint.

### Changed

- `trans.py` reduced from ~696 lines to ~175 lines; `paper.py` reduced from ~574 lines to ~175 lines.
- `ASK_LLM_TRANSLATION_THREADS` now maps directly to `translation.max_concurrent_api_calls`, the field actually used by `trans`.
- `mypy` is now a required CI gate (`continue-on-error: true` removed).

### Fixed

- Resolved 14 remaining `mypy` errors across `utils/console.py`, `config/loader.py`, `utils/token_counter.py`, `config/paper_explain_pipeline.py`, `core/chat.py`, `cli/commands/ask.py`, `core/batch_processor.py`, and `core/paper_explain.py`.
- Fixed `ReasoningChunk` handling in streaming paths so token counting and output joining receive `str` content.

### Tests

- Added unit tests for `resolve_provider_and_model_or_exit()`.
- Added `RunMetrics` unit tests covering retries and failures.

### Documentation

- Updated `AGENTS.md` with the Service layer and updated project structure.

### Version

- 2.9.0 → 2.10.0

### Contributors

- Designed and implemented with assistance from **kimi-code** (agent) and **kimi-k2.7** (model).

## 2.9.0 (2026-06-22)

### Features

- **`ask-llm trans` directory output fix**: `-o <dir>` now correctly creates the directory and writes each translated file into it, instead of overwriting the directory path as a single file.
- **`ask-llm trans` cross-file + cross-chunk parallelism**: text and Markdown files are now translated with file-level parallelism (`--max-parallel-files`) while each file keeps chunk-level parallelism (`--threads`).
- **Failure isolation and early per-file export**: each file is saved as soon as its own translation finishes. A failure in one file no longer blocks or invalidates other files; successful files are still written to the output directory.

### Changed

- `--threads` help text now reads "Max concurrent API calls per file".
- `--max-parallel-files` help text now reads "Max files to translate in parallel".
- Internal refactor of `trans.py` into `_prepare_text_file`, `_translate_and_export_text_file`, and `_export_text_file` helpers for clearer per-file lifecycle management.
- Added `_is_directory_output` and `_offset_task_ids` helpers in `ask_llm.cli.common`.

### Tests

- Added integration tests for directory-output heuristics (`_is_directory_output`).
- Added integration tests for task/chunk ID offsetting (`_offset_task_ids`).
- Added `TestTransPerFileBatching` to verify each file gets its own batch call and that a failing file does not block others.

### Version

- 2.8.0 → 2.9.0

### Contributors

- Designed and implemented with assistance from **kimi-code** (agent) and **kimi-k2.7** (model).

## 2.8.0 (2026-06-19)

### Features

- **Optional non-streaming API for translation**: new `--stream-api / --no-stream-api` flag on `ask-llm trans`. Non-streaming mode reduces per-token overhead and is recommended for batch throughput; streaming remains the default for interactive use.
- **tiktoken encoding cache**: `TokenCounter` now caches `Encoding` objects by encoding name, eliminating repeated `tiktoken.get_encoding()` calls during splitting, rebalancing, and statistics.
- **Global per-provider rate limiter**: new `GlobalRateLimiter` limits concurrent API requests per provider/model across files and `GlobalBatchProcessor` instances, reducing 429 rate-limit retries and improving tail latency.

### Changed

- **Default translation concurrency reduced**: `translation.threads` and `translation.max_concurrent_api_calls` lowered from `32` to `12` to match the new rate limiter and avoid provider throttling.
- **Version**: 2.7.5 → 2.8.0

### Contributors

- Designed and implemented with assistance from **Kimi CLI** (agent) and **kimi-k2.7** (model).

## 2.7.5 (2026-05-29)

### Fixes

- **`ask-llm format --type title` no longer formats headings inside code blocks**.
  - `HeadingExtractor` now detects Markdown code fences (`` ``` `` and `~~~`) and skips any headings that appear within code block ranges.
  - This prevents LLM from reformatting example Markdown headings inside `` ```md `` or other fenced code blocks, which was an unintended side effect.
  - Handles both closed and unclosed code blocks (unclosed blocks are treated as code until end of file).

### Contributors

- Fix designed and implemented with assistance from **Kimi CLI** (agent) and **kimi-k2.6** (model).

## 2.7.4 (2026-05-29)

### Features

- **New English prompt templates**: Added `md-heading-format_en.md` and `md-body-format_en.md` for English Markdown formatting workflows.
- **Enhanced Chinese prompt templates**:
  - `md-body-format.md`: Added explicit rules for Chinese punctuation, paragraph splitting for long content, LaTeX symbol fixes, and math formula spacing in tables (`|x|` → `\vert x \vert`).
  - `tech-paper-trans.md` & `tech-paper-trans-compact.md`: Added instructions to fix malformed formulas and enforce spacing around inline math in Chinese text.
- **Provider model updates (`providers.yml`)**:
  - Added `deepseek-v4-flash` (1M context, 384K output) and `deepseek-v4-pro` (1M context, 384K output) with pricing and capability metadata.
  - Improved numeric readability with underscore separators (`128_000` instead of `128000`).
- **Default configuration tuning (`default_config.yml`)**:
  - Increased translation concurrency: `threads` / `max_concurrent_api_calls` from `20` → `32`.
  - Increased translation chunk size: `max_chunk_tokens` from `3000` → `6000`, `min_chunk_merge_tokens` from `1500` → `3000`.
  - Increased formatting throughput: `format_heading.batch_size` from `80` → `160`, `format_body.concurrency` from `8` → `32`.
  - Increased paper explain concurrency: `paper.concurrency` from `20` → `32`.

### Contributors

- Feature designed and implemented with assistance from **Kimi CLI** (agent) and **kimi-k2.6** (model).

## 2.7.3 (2026-05-21)

### Features

- **Format command retry and checkpoint recovery**: `ask-llm format` now gracefully handles API timeouts and failures.
  - **Per-chunk/batch retry**: Failed API calls are automatically retried with exponential backoff (`retry_delay * 2^attempt`, capped at `retry_delay_max`). Configurable via `--retries` / `--retry-delay` / `--retry-delay-max` CLI flags or `default_config.yml`.
  - **Failure fallback**: After all retries are exhausted, failed chunks retain their original content while successful chunks keep their formatted results. The merged output is still saved.
  - **Checkpoint persistence**: When chunks fail, a checkpoint file (`.body_checkpoint.json` / `.title_checkpoint.json`) is saved next to the source file, containing the full context needed to resume.
  - **Resume support**: `ask-llm format --resume <checkpoint.json>` re-processes only the failed chunks and merges with previous successful results. Automatically deletes the checkpoint when all chunks succeed.
  - **New CLI flags**: `--retries`, `--retry-delay`, `--retry-delay-max`, `--resume`.
  - **New config fields**: `format_body.retries`, `format_body.retry_delay`, `format_body.retry_delay_max`, `format_heading.retries`, `format_heading.retry_delay`, `format_heading.retry_delay_max`.

- **Directory recursion depth control**: `ask-llm format ./dir --max-depth N` limits how many levels of subdirectories to scan for Markdown files. `0` means only the top-level directory.

### Refactors

- **`BodyFormatter.format_body()`**: Now returns `BodyFormatResult` (dataclass with `text`, `stats`, `failed_chunks`, `checkpoint_path`) instead of `(str, stats)` tuple.
- **`HeadingFormatter.format_headings()`**: Now returns `HeadingFormatResult` (dataclass with `formatted_headings`, `stats`, `failed_batches`, `checkpoint_path`) instead of `List[str]`.
- **New `format_checkpoint.py` module**: Centralized checkpoint serialization/deserialization with `FormatCheckpoint`, `FailedChunkInfo`, `SuccessfulChunkInfo` data classes.

### Contributors

- Feature designed and implemented with assistance from **Kimi CLI** (agent) and **kimi-k2.6** (model).

## 2.7.2 (2026-05-21)

### Features

- **Enhanced format body logging**: `ask-llm format --type body` now provides detailed per-chunk and aggregate token consumption information.
  - Start of formatting: logs model name, total document tokens, number of chunks, and concurrency.
  - Per-chunk start: logs chunk index, estimated tokens, and position.
  - Per-chunk completion: logs chunk index, input/output tokens, and latency.
  - Completion summary: logs total input/output tokens and total latency for all chunks.
  - CLI output: shows per-file token consumption and a final aggregate summary.

### Fixes

- **Eliminated spurious API key warnings**: Environment variable resolution for unused providers (e.g., `KIMI_CODE_API_KEY` when using `--provider aliyun`) no longer emits `WARNING` logs. Both `ask_llm` and `llm-engine` now log missing env vars at `DEBUG` level only.
- **Aliyun timeout increased**: Added `timeout: 300.0` to the `aliyun` provider in `providers.yml` to accommodate slower response times from Qwen models.

### Refactors

- **`BodyFormatter.format_body()`**: Now returns `(str, BodyFormatStats)` tuple instead of just `str`, enabling callers to aggregate token statistics across files.
- **`FormatMarkdownOutcome`**: Extended with `total_input_tokens`, `total_output_tokens`, and `total_latency` fields.

### Contributors

- Feature designed and implemented with assistance from **Kimi CLI** (agent) and **kimi-k2.6** (model).

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
