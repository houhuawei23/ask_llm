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


def _escalate(processor, task, provider_cache, *, max_retries):
    """Drive the shared-budget escalation the way ``BoundedRetryRunner`` does.

    Each step attempts exactly one config; transient failures advance
    ``retry_count``, terminal failures stop at once (the worker saturates
    ``retry_count`` to ``max_retries``). Mirrors the runner's retry gate so unit
    tests can exercise a full escalation without spinning up the thread pool.
    """
    history: dict[int, list] = {}
    retry_count = 0
    while True:
        result = processor._process_single_global_task(
            task,
            provider_cache,
            retry_count=retry_count,
            attempt_history_by_task=history,
        )
        if result.status == TaskStatus.SUCCESS:
            return result
        # Runner gate: stop once the retry budget is exhausted. A terminal error
        # saturates result.retry_count to max_retries, so this also short-circuits.
        if result.retry_count >= max_retries:
            return result
        retry_count += 1


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
        # Shared-budget escalation: primary (attempt 0) fails transiently, then
        # fallback (attempt 1) succeeds.
        result = _escalate(processor, task, provider_cache, max_retries=processor.max_retries)

    assert result.status == TaskStatus.SUCCESS
    assert result.response == "fallback result"
    assert result.model_settings.provider == "fallback"
    assert result.model_settings.model == "model-b"
    assert called == ["primary/model-a", "fallback/model-b"]
    # The successful result's attempt_history holds the preceding primary failure.
    assert len(result.attempt_history) == 1
    assert result.attempt_history[0].provider == "primary"


def test_all_configs_fail_returns_failed():
    """B1 regression: shared budget bounds calls by retry budget, not x chain length.

    With ``max_retries=3`` and a 2-config fallback chain, the task must make at
    most ``max_retries + 1 == 4`` API calls (primary once, then the last config
    retried for the remaining budget) -- never ``(max_retries + 1) * len(chain)
    == 8`` as the old two-layer retry produced.
    """
    task = _make_task(fallback_configs=[ModelConfig(provider="fallback", model="model-b")])
    processor = GlobalBatchProcessor(max_retries=3)
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
            proc.process.side_effect = RuntimeError(f"{provider.name} down")
            return proc

        mock_rp.side_effect = side_effect
        result = _escalate(processor, task, provider_cache, max_retries=processor.max_retries)

    assert result.status == TaskStatus.FAILED
    assert result.error is not None
    assert "fallback/model-b down" in result.error
    assert result.model_settings.provider == "fallback"
    assert result.model_settings.model == "model-b"
    assert result.error_category == ErrorCategory.UNKNOWN
    # B1 invariant: 4 calls == max_retries + 1, NOT 8.
    assert len(called) == 4
    # attempt 0 = primary; attempts 1..3 retry the last config (fallback).
    assert called == [
        "primary/model-a",
        "fallback/model-b",
        "fallback/model-b",
        "fallback/model-b",
    ]
    # attempt_history records the *preceding* attempts (flat AttemptRecords), not
    # the final result itself -- 3 preceding attempts here.
    assert len(result.attempt_history) == 3
    assert result.attempt_history[0].provider == "primary"
    assert all(r.error_category == ErrorCategory.UNKNOWN for r in result.attempt_history)
    # Must serialize without circular references.
    result.model_dump(mode="json")


def test_single_config_retries_same_provider_within_budget():
    """B1: a single-config task retries the same provider, bounded by the budget.

    No fallback chain, so every attempt re-uses the primary. Total calls ==
    ``max_retries + 1`` (behaviour unchanged from before the escalation unify).
    """
    task = _make_task()  # no fallbacks
    processor = GlobalBatchProcessor(max_retries=2)
    primary = _make_provider("primary", "model-a")
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
            proc.process.side_effect = RuntimeError("primary down")
            return proc

        mock_rp.side_effect = side_effect
        result = _escalate(processor, task, provider_cache, max_retries=processor.max_retries)

    assert result.status == TaskStatus.FAILED
    assert "primary down" in (result.error or "")
    # 3 calls == max_retries + 1, all on the primary.
    assert called == ["primary/model-a", "primary/model-a", "primary/model-a"]


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


def test_process_global_tasks_creates_per_worker_progress_bars():
    """B6: progress bars scale with the worker count, not the task count.

    A 20-task run with max_workers=4 must create exactly 4 progress bars (one
    per worker slot), never 20 (one per task).
    """
    tasks = [
        BatchTask(
            task_id=i,
            prompt="p",
            content="c",
            model_settings=ModelConfig(provider="primary", model="model-a"),
        )
        for i in range(20)
    ]
    processor = GlobalBatchProcessor(max_workers=4)
    cm = MagicMock()
    cm.config.get_provider_config.return_value = ProviderConfig(
        api_provider="primary",
        api_base="https://api.primary.com/v1",
        api_key="sk-test",
        models=["model-a"],
    )
    primary = _make_provider("primary", "model-a")
    add_task_counter = {"n": 0}

    def fake_add_task(*args, **kwargs):
        add_task_counter["n"] += 1
        return add_task_counter["n"]  # unique TaskID per bar

    with (
        patch("ask_llm.core.progress_presenter.Progress") as mock_progress_cls,
        patch("ask_llm.utils.provider_cache.create_provider_adapter", return_value=primary),
        _patch_rate_limiter() as limiter_patch,
        _patch_token_helpers(),
        patch("ask_llm.core.batch_processor.RequestProcessor") as mock_rp,
    ):
        limiter_patch.return_value.burst_for.return_value = 100  # cap == max_workers (4)
        progress_instance = MagicMock()
        progress_instance.add_task.side_effect = fake_add_task
        mock_progress_cls.return_value = progress_instance

        proc = MagicMock()
        proc.process.return_value = iter(["ok"])
        mock_rp.return_value = proc

        results = processor.process_global_tasks(tasks, cm, show_progress=True)

    assert len(results) == 20
    # Bars created == max_workers (4), NOT task count (20).
    assert add_task_counter["n"] == 4


def test_process_global_tasks_bounded_calls_with_fallback_chain():
    """B1 (runner-level): a fallback chain must not multiply API calls by chain length.

    Each task has a 2-config fallback chain and always fails with a transient
    error. Total API calls must stay <= ``n_tasks * (max_retries + 1)`` — never
    ``n_tasks * (max_retries + 1) * len(chain)`` as the old two-layer retry did.
    This is the ARCHITECTURE_REVIEW.md P1 acceptance criterion.
    """
    max_retries = 2
    n_tasks = 5
    tasks = [
        BatchTask(
            task_id=i,
            prompt="p",
            content="c",
            model_settings=ModelConfig(provider="primary", model="model-a"),
            fallback_model_configs=[ModelConfig(provider="fallback", model="model-b")],
        )
        for i in range(n_tasks)
    ]
    processor = GlobalBatchProcessor(max_workers=4, max_retries=max_retries)
    cm = MagicMock()
    cm.config.get_provider_config.return_value = ProviderConfig(
        api_provider="primary",
        api_base="https://api.primary.com/v1",
        api_key="sk-test",
        models=["model-a"],
    )
    primary = _make_provider("primary", "model-a")
    call_counter = {"n": 0}

    with (
        patch("ask_llm.core.progress_presenter.Progress"),
        patch("ask_llm.utils.provider_cache.create_provider_adapter", return_value=primary),
        _patch_rate_limiter() as limiter_patch,
        _patch_token_helpers(),
        patch("ask_llm.core.batch_processor.RequestProcessor") as mock_rp,
    ):
        limiter_patch.return_value.burst_for.return_value = 100

        def side_effect(provider):
            call_counter["n"] += 1
            proc = MagicMock()
            proc.provider = provider
            proc.process.side_effect = RuntimeError("connection timeout")
            return proc

        mock_rp.side_effect = side_effect
        results = processor.process_global_tasks(tasks, cm, show_progress=True)

    assert len(results) == n_tasks
    assert all(r.status == TaskStatus.FAILED for r in results)
    # B1 invariant: <= n_tasks * (max_retries + 1) == 5 * 3 == 15.
    # Old two-layer retry would have made 5 * 3 * 2 == 30.
    assert call_counter["n"] <= n_tasks * (max_retries + 1)
