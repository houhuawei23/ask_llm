"""Tests for error classification and structured logging context."""

import pytest

from ask_llm.core.telemetry import (
    ErrorCategory,
    LogContext,
    bind_context,
    classify_error,
    should_fallback_for_error,
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("401 Unauthorized", ErrorCategory.AUTHENTICATION),
        ("Authentication failed", ErrorCategory.AUTHENTICATION),
        ("invalid api key", ErrorCategory.AUTHENTICATION),
        ("429 Too Many Requests", ErrorCategory.RATE_LIMIT),
        ("rate limit exceeded", ErrorCategory.RATE_LIMIT),
        ("Request timed out", ErrorCategory.TIMEOUT),
        ("deadline exceeded", ErrorCategory.TIMEOUT),
        ("content filter triggered", ErrorCategory.CONTENT_FILTER),
        ("moderation policy violation", ErrorCategory.CONTENT_FILTER),
        ("model not found", ErrorCategory.MODEL_ERROR),
        ("context length too long", ErrorCategory.MODEL_ERROR),
        ("connection refused", ErrorCategory.NETWORK_ERROR),
        ("DNS lookup failed", ErrorCategory.NETWORK_ERROR),
        ("Provider not found in cache", ErrorCategory.VALIDATION_ERROR),
        ("some random failure", ErrorCategory.UNKNOWN),
        ("", ErrorCategory.UNKNOWN),
        (None, ErrorCategory.UNKNOWN),
    ],
)
def test_classify_error(message, expected):
    assert classify_error(message) == expected


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (ErrorCategory.AUTHENTICATION, False),
        (ErrorCategory.CONTENT_FILTER, False),
        (ErrorCategory.VALIDATION_ERROR, False),
        (ErrorCategory.RATE_LIMIT, True),
        (ErrorCategory.TIMEOUT, True),
        (ErrorCategory.MODEL_ERROR, True),
        (ErrorCategory.NETWORK_ERROR, True),
        (ErrorCategory.UNKNOWN, True),
    ],
)
def test_should_fallback_for_error(category, expected):
    assert should_fallback_for_error(category) == expected


def test_log_context_defaults():
    ctx = LogContext(task_id=1, provider="deepseek", model="deepseek-chat")
    assert ctx.task_id == 1
    assert ctx.provider == "deepseek"
    assert ctx.model == "deepseek-chat"
    assert ctx.attempt == 1
    assert ctx.request_id
    extra = ctx.to_extra()
    assert extra["task_id"] == 1
    assert extra["provider"] == "deepseek"


def test_log_context_with_attempt():
    ctx = LogContext(task_id=1).with_attempt(3)
    assert ctx.attempt == 3


def test_log_context_with_provider_model():
    ctx = LogContext(task_id=1).with_provider_model("qwen", "qwen-max")
    assert ctx.provider == "qwen"
    assert ctx.model == "qwen-max"


def test_bind_context_returns_bound_logger():
    logger = bind_context(LogContext(task_id=42))
    assert logger is not None
