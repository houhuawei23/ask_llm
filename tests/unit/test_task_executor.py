"""Unit tests for the extracted TaskExecutor (P1.3)."""

from __future__ import annotations

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
    """When the rate limiter times out, the attempt fails (not raises)."""
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
    assert "Rate limit timeout" in (result.error or "")


def test_auth_error_dedup_flag_flips_once():
    """The batch-wide auth-error flag flips exactly once across parallel workers."""
    executor = TaskExecutor()
    assert executor._auth_error_logged is False

    executor.log_task_failure(1, "p/m", "401 Unauthorized", ErrorCategory.AUTHENTICATION)
    assert executor._auth_error_logged is True

    # A second parallel auth failure does not re-trigger (flag stays True).
    executor.log_task_failure(2, "p/m", "401 Unauthorized", ErrorCategory.AUTHENTICATION)
    assert executor._auth_error_logged is True


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
