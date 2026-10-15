"""Unit tests for the extracted TaskExecutor (P1.3)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ask_llm.core.batch_models import BatchTask, ModelConfig, TaskStatus
from ask_llm.core.task_executor import TaskExecutor
from ask_llm.core.telemetry import ErrorCategory


def _task() -> BatchTask:
    return BatchTask(
        task_id=1,
        prompt="p",
        content="c",
        model_settings=ModelConfig(provider="p", model="m"),
    )


def _patch_tokens():
    return patch.multiple(
        "ask_llm.utils.token_counter.TokenCounter",
        estimate_tokens=lambda text, model: {"word_count": 1, "token_count": 1},
        count_words=lambda text: 1,
        get_encoding=lambda model: None,
        count_tokens=lambda text, model: 1,
    )


def test_rate_limit_acquire_timeout_yields_failed_result():
    """When the rate limiter times out, the attempt fails (not raises).

    Audit 3.3: the result is flagged ``throttled`` so the runner tail-requeues
    the task without consuming its retry budget, instead of re-blocking a
    worker on every backoff round.
    """
    executor = TaskExecutor()
    limiter = MagicMock()
    limiter.acquire_timeout.return_value = 5.0
    limiter.acquire.return_value = False  # token never acquired
    provider_cache = {"p/m": MagicMock()}

    with (
        patch("ask_llm.core.task_executor.get_global_rate_limiter", return_value=limiter),
        _patch_tokens(),
    ):
        result = executor.try_run_with_config(
            _task(),
            ModelConfig(provider="p", model="m"),
            provider_cache,
            0,
            None,
            None,
            None,
        )

    assert result.status == TaskStatus.FAILED
    assert "Rate limit wait timeout" in (result.error or "")
    assert result.throttled is True
    assert result.error_category == ErrorCategory.RATE_LIMIT


def test_auth_error_dedup_flag_flips_once():
    """The batch-wide auth-error flag flips exactly once across parallel workers."""
    executor = TaskExecutor()
    assert executor.auth_error_logged is False

    executor.log_task_failure(1, "p/m", "401 Unauthorized", ErrorCategory.AUTHENTICATION)
    assert executor.auth_error_logged is True

    # A second parallel auth failure does not re-trigger (flag stays True).
    executor.log_task_failure(2, "p/m", "401 Unauthorized", ErrorCategory.AUTHENTICATION)
    assert executor.auth_error_logged is True


def test_missing_provider_in_cache_yields_failed_result():
    """A missing cache entry surfaces as a failed result, not an exception."""
    executor = TaskExecutor()
    limiter = MagicMock()
    limiter.acquire_timeout.return_value = 5.0
    limiter.acquire.return_value = True

    with (
        patch("ask_llm.core.task_executor.get_global_rate_limiter", return_value=limiter),
        _patch_tokens(),
    ):
        result = executor.try_run_with_config(
            _task(),
            ModelConfig(provider="p", model="m"),
            {},  # provider_cache: no adapter cached
            0,
            None,
            None,
            None,
        )

    assert result.status == TaskStatus.FAILED
    assert "Provider not found" in (result.error or "")


def test_auth_error_dedup_is_per_provider():
    """M3: a second *different* provider's auth failure must log at ERROR —
    the old executor-wide flag silenced it after the first provider failed."""
    executor = TaskExecutor()
    executor.log_task_failure(1, "p/a", "401 Unauthorized", ErrorCategory.AUTHENTICATION)
    executor.log_task_failure(2, "p/b", "401 Unauthorized", ErrorCategory.AUTHENTICATION)
    # Both keys are recorded; a repeat of either one is deduped.
    assert executor._auth_error_logged_keys == {"p/a", "p/b"}
    assert executor.auth_error_logged is True


# --- retry-aware progress labels ---------------------------------------------


class _FakeProvider:
    name = "p"

    def __init__(self, chunks=("hello ", "world"), error: str | None = None):
        # ``config`` mirrors the real provider attribute (lowercase by protocol).
        self.config = SimpleNamespace(api_temperature=0.7)
        self._chunks = chunks
        self._error = error

    def call(self, **kwargs):
        if self._error:
            raise RuntimeError(self._error)
        return iter(self._chunks)


def _run(executor, provider, retry_count=0, progress=None):
    with (
        patch(
            "ask_llm.core.task_executor.get_global_rate_limiter",
            return_value=MagicMock(acquire_timeout=lambda *a: 1.0, acquire=lambda *a, **k: True),
        ),
        _patch_tokens(),
    ):
        return executor.try_run_with_config(
            _task(),
            ModelConfig(provider="p", model="m"),
            {"p/m": provider},
            retry_count,
            progress,
            0 if progress is not None else None,
            100,
        )


def test_transient_failure_label_is_retry_pending():
    """A budget-left transient failure must read as retry-pending, not dead."""
    executor = TaskExecutor(max_retries=3)
    progress = MagicMock()
    result = _run(
        executor,
        _FakeProvider(error="Error code: 503 - Service is too busy"),
        retry_count=0,
        progress=progress,
    )

    assert result.status == TaskStatus.FAILED
    description = progress.update.call_args.kwargs["description"]
    assert "↻" in description
    assert "第1/4次失败" in description


def test_terminal_failure_label_is_dead():
    executor = TaskExecutor(max_retries=3)
    progress = MagicMock()
    result = _run(
        executor,
        _FakeProvider(error="401 Unauthorized: invalid api key"),
        retry_count=0,
        progress=progress,
    )

    assert result.status == TaskStatus.FAILED
    description = progress.update.call_args.kwargs["description"]
    assert "✗ Failed" in description
    assert "↻" not in description


def test_exhausted_budget_label_is_dead():
    executor = TaskExecutor(max_retries=3)
    progress = MagicMock()
    result = _run(
        executor,
        _FakeProvider(error="Error code: 503 - Service is too busy"),
        retry_count=3,
        progress=progress,
    )

    assert result.status == TaskStatus.FAILED
    description = progress.update.call_args.kwargs["description"]
    assert "✗ Failed" in description


def test_retry_attempt_start_label_shows_attempt_number():
    """Attempt k+1 (retry_count=k) relabels the bar with 第k+1次尝试."""
    executor = TaskExecutor(max_retries=3)
    progress = MagicMock()
    result = _run(executor, _FakeProvider(), retry_count=2, progress=progress)

    assert result.status == TaskStatus.SUCCESS
    first_call = progress.update.call_args_list[0]
    assert "第3次尝试" in first_call.kwargs["description"]


def test_non_retryable_non_terminal_failure_label_is_dead():
    """billing/model-error failures are non-terminal for fallback yet never
    retried — the label must NOT promise a retry the runner will not make."""
    executor = TaskExecutor(max_retries=3)
    progress = MagicMock()
    result = _run(
        executor,
        _FakeProvider(error="Error code: 400 - bad request: invalid_request_error"),
        retry_count=0,
        progress=progress,
    )

    assert result.status == TaskStatus.FAILED
    description = progress.update.call_args.kwargs["description"]
    assert "✗ Failed" in description
    assert "将重试" not in description
