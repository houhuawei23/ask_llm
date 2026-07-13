"""Unit tests for GlobalBatchProcessor fallback execution."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.batch_models import BatchResult, BatchTask, ModelConfig, TaskStatus
from ask_llm.core import batch_processor as bp_module
from ask_llm.core.batch_processor import GlobalBatchProcessor
from ask_llm.core.models import ProviderConfig
from ask_llm.core.provider_manager import ProviderManager
from ask_llm.core.telemetry import ErrorCategory


@pytest.fixture(autouse=True)
def _patch_paper_timeout():
    with patch.object(bp_module, "paper_request_timeout_seconds", return_value=600.0):
        yield


def _make_task(fallback_configs=None):
    return BatchTask(
        task_id=1,
        prompt="Translate: {content}",
        content="hello",
        output_filename="out.txt",
        model_settings=ModelConfig(provider="primary", model="model-a"),
        fallback_model_configs=fallback_configs or [],
    )


def _make_provider(provider: str, model: str) -> MagicMock:
    p = MagicMock()
    p.name = f"{provider}/{model}"
    p.config.api_temperature = 0.7
    return p


def _patch_token_helpers():
    return patch.multiple(
        "ask_llm.utils.token_counter.TokenCounter",
        estimate_tokens=lambda text, model: {"word_count": 1, "token_count": 1},
        count_words=lambda text: 1,
        get_encoding=lambda model: None,
        count_tokens=lambda text, model: 1,
    )


def _patch_rate_limiter():
    limiter = MagicMock()
    limiter.acquire.return_value = True
    return patch("ask_llm.core.batch_processor.get_global_rate_limiter", return_value=limiter)


def test_primary_succeeds_fallback_not_used():
    task = _make_task(fallback_configs=[ModelConfig(provider="fallback", model="model-b")])
    processor = GlobalBatchProcessor()
    primary = _make_provider("primary", "model-a")
    fallback = _make_provider("fallback", "model-b")
    provider_cache: dict[str, Any] = {"primary/model-a": primary}

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.batch_processor.RequestProcessor") as mock_rp,
    ):
        called = []

        def side_effect(provider):
            called.append(provider.name)
            proc = MagicMock()
            proc.provider = provider
            proc.process.return_value = iter(["success"])
            return proc

        mock_rp.side_effect = side_effect
        result = processor._process_single_global_task(task, provider_cache, retry_count=0)

    assert result.status == TaskStatus.SUCCESS
    assert result.response == "success"
    assert result.model_settings.provider == "primary"
    assert result.model_settings.model == "model-a"
    assert called == ["primary/model-a"]


def test_fallback_succeeds_when_primary_fails():
    task = _make_task(fallback_configs=[ModelConfig(provider="fallback", model="model-b")])
    processor = GlobalBatchProcessor()
    primary = _make_provider("primary", "model-a")
    fallback = _make_provider("fallback", "model-b")
    provider_cache: dict[str, Any] = {
        "primary/model-a": primary,
        "fallback/model-b": fallback,
    }

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.batch_processor.RequestProcessor") as mock_rp,
    ):
        called = []

        def side_effect(provider):
            called.append(provider.name)
            proc = MagicMock()
            proc.provider = provider
            if provider.name == "primary/model-a":
                proc.process.side_effect = RuntimeError("primary down")
            else:
                proc.process.return_value = iter(["fallback result"])
            return proc

        mock_rp.side_effect = side_effect
        result = processor._process_single_global_task(task, provider_cache, retry_count=0)

    assert result.status == TaskStatus.SUCCESS
    assert result.response == "fallback result"
    assert result.model_settings.provider == "fallback"
    assert result.model_settings.model == "model-b"
    assert called == ["primary/model-a", "fallback/model-b"]


def test_all_configs_fail_returns_failed():
    task = _make_task(fallback_configs=[ModelConfig(provider="fallback", model="model-b")])
    processor = GlobalBatchProcessor()
    primary = _make_provider("primary", "model-a")
    fallback = _make_provider("fallback", "model-b")
    provider_cache: dict[str, Any] = {
        "primary/model-a": primary,
        "fallback/model-b": fallback,
    }

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.batch_processor.RequestProcessor") as mock_rp,
    ):

        def side_effect(provider):
            proc = MagicMock()
            proc.provider = provider
            proc.process.side_effect = RuntimeError(f"{provider.name} down")
            return proc

        mock_rp.side_effect = side_effect
        result = processor._process_single_global_task(task, provider_cache, retry_count=0)

    assert result.status == TaskStatus.FAILED
    assert result.error is not None
    assert "fallback/model-b down" in result.error
    assert result.model_settings.provider == "fallback"
    assert result.model_settings.model == "model-b"
    assert result.error_category == ErrorCategory.UNKNOWN
    # attempt_history records the *preceding* attempts, not the final result itself.
    assert len(result.attempt_history) == 1
    assert result.attempt_history[0].model_settings.provider == "primary"
    assert all(r.error_category == ErrorCategory.UNKNOWN for r in result.attempt_history)
    # Must serialize without circular references.
    result.model_dump(mode="json")


def test_no_fallback_returns_failed_on_primary_failure():
    task = _make_task()
    processor = GlobalBatchProcessor()
    primary = _make_provider("primary", "model-a")
    provider_cache: dict[str, Any] = {"primary/model-a": primary}

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.batch_processor.RequestProcessor") as mock_rp,
    ):

        def side_effect(provider):
            proc = MagicMock()
            proc.provider = provider
            proc.process.side_effect = RuntimeError("primary down")
            return proc

        mock_rp.side_effect = side_effect
        result = processor._process_single_global_task(task, provider_cache, retry_count=0)

    assert result.status == TaskStatus.FAILED
    assert result.error is not None
    assert "primary down" in result.error
    assert result.error_category == ErrorCategory.UNKNOWN


def test_authentication_error_stops_fallback_chain():
    task = _make_task(
        fallback_configs=[
            ModelConfig(provider="fallback1", model="model-b"),
            ModelConfig(provider="fallback2", model="model-c"),
        ]
    )
    processor = GlobalBatchProcessor()
    primary = _make_provider("primary", "model-a")
    fallback1 = _make_provider("fallback1", "model-b")
    fallback2 = _make_provider("fallback2", "model-c")
    provider_cache: dict[str, Any] = {
        "primary/model-a": primary,
        "fallback1/model-b": fallback1,
        "fallback2/model-c": fallback2,
    }

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.batch_processor.RequestProcessor") as mock_rp,
    ):
        called = []

        def side_effect(provider):
            called.append(provider.name)
            proc = MagicMock()
            proc.provider = provider
            if provider.name == "primary/model-a":
                proc.process.side_effect = RuntimeError("401 Unauthorized")
            else:
                proc.process.return_value = iter(["should not reach"])
            return proc

        mock_rp.side_effect = side_effect
        result = processor._process_single_global_task(task, provider_cache, retry_count=0)

    assert result.status == TaskStatus.FAILED
    assert result.error_category == ErrorCategory.AUTHENTICATION
    # The first (and only) attempt is the result itself; there are no preceding attempts.
    assert len(result.attempt_history) == 0
    assert called == ["primary/model-a"]
    # Must serialize without circular references.
    result.model_dump(mode="json")


def test_build_provider_cache_includes_fallbacks():
    task = _make_task(fallback_configs=[ModelConfig(provider="fallback", model="model-b")])
    cm = MagicMock()
    base_cfg = ProviderConfig(
        api_provider="primary",
        api_base="https://api.primary.com/v1",
        api_key="sk-test",
        models=["model-a"],
    )
    cm.config.get_provider_config.return_value = base_cfg

    with patch("ask_llm.utils.provider_cache.create_provider_adapter") as mock_create:
        mock_create.return_value = MagicMock()
        cache = ProviderManager(cm).build_provider_cache([task])

    assert "primary/model-a" in cache
    assert "fallback/model-b" in cache
    assert mock_create.call_count == 2

    calls = [call.kwargs.get("default_model") for call in mock_create.call_args_list]
    assert "model-a" in calls
    assert "model-b" in calls
