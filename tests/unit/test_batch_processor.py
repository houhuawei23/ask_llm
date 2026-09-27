"""Unit tests for GlobalBatchProcessor task execution."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.batch_models import BatchResult, BatchTask, ModelConfig, TaskStatus
from ask_llm.core.batch_processor import GlobalBatchProcessor
from ask_llm.core.models import ProviderConfig
from ask_llm.core.provider_manager import build_provider_cache
from ask_llm.config.unified_config import RateLimitConfig
from ask_llm.core.error_keywords import ErrorCategory


@pytest.fixture(autouse=True)
def _patch_paper_timeout():
    with patch("ask_llm.core.task_executor.paper_request_timeout_seconds", return_value=600.0):
        yield


def _make_task():
    return BatchTask(
        task_id=1,
        prompt="Translate: {content}",
        content="hello",
        output_filename="out.txt",
        model_settings=ModelConfig(provider="primary", model="model-a"),
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
    return patch("ask_llm.core.task_executor.get_global_rate_limiter", return_value=limiter)


def _escalate(processor, task, provider_cache, *, max_retries):
    """Drive retries the way ``BoundedRetryRunner`` does.

    Each step attempts exactly one config; the caller advances ``retry_count``
    until the budget is exhausted (transient errors). Mirrors the runner's
    retry loop so unit tests can exercise retries without a thread pool.
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
        # Runner gate: stop once the retry budget is exhausted.
        if result.retry_count >= max_retries:
            return result
        retry_count += 1


def test_primary_succeeds():
    task = _make_task()
    processor = GlobalBatchProcessor()
    primary = _make_provider("primary", "model-a")
    provider_cache: dict[str, Any] = {"primary/model-a": primary}

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.task_executor.RequestProcessor") as mock_rp,
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


def test_all_retries_fail_returns_failed():
    """B1 regression: retries are bounded by the retry budget.

    With ``max_retries=3`` the task must make at most ``max_retries + 1 == 4``
    API calls, all on the primary provider.
    """
    task = _make_task()
    processor = GlobalBatchProcessor(max_retries=3)
    primary = _make_provider("primary", "model-a")
    provider_cache: dict[str, Any] = {"primary/model-a": primary}

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.task_executor.RequestProcessor") as mock_rp,
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
    assert "primary/model-a down" in result.error
    assert result.model_settings.provider == "primary"
    assert result.model_settings.model == "model-a"
    assert result.error_category == ErrorCategory.UNKNOWN
    # B1 invariant: 4 calls == max_retries + 1.
    assert len(called) == 4
    assert called == ["primary/model-a"] * 4
    # attempt_history records the *preceding* attempts (flat AttemptRecords), not
    # the final result itself -- 3 preceding attempts here.
    assert len(result.attempt_history) == 3
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
        patch("ask_llm.core.task_executor.RequestProcessor") as mock_rp,
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


def test_primary_failure_returns_failed_result():
    task = _make_task()
    processor = GlobalBatchProcessor()
    primary = _make_provider("primary", "model-a")
    provider_cache: dict[str, Any] = {"primary/model-a": primary}

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.task_executor.RequestProcessor") as mock_rp,
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


def test_authentication_error_is_terminal_single_attempt():
    task = _make_task()
    processor = GlobalBatchProcessor()
    primary = _make_provider("primary", "model-a")
    provider_cache: dict[str, Any] = {"primary/model-a": primary}

    with (
        _patch_rate_limiter(),
        _patch_token_helpers(),
        patch("ask_llm.core.task_executor.RequestProcessor") as mock_rp,
    ):
        called = []

        def side_effect(provider):
            called.append(provider.name)
            proc = MagicMock()
            proc.provider = provider
            proc.process.side_effect = RuntimeError("401 Unauthorized")
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


def test_build_provider_cache_primary_only():
    task = _make_task()
    cm = MagicMock()
    base_cfg = ProviderConfig(
        api_provider="primary",
        api_base="https://api.primary.com/v1",
        api_key="sk-test",
        models=["model-a"],
    )
    cm.unified_config.get_provider_config.return_value = base_cfg

    with patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_create:
        mock_create.return_value = MagicMock()
        cache = build_provider_cache([task], cm)

    assert "primary/model-a" in cache
    assert mock_create.call_count == 1

    calls = [call.kwargs.get("default_model") for call in mock_create.call_args_list]
    assert "model-a" in calls


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
    cm.unified_config.get_provider_config.return_value = ProviderConfig(
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
        patch("ask_llm.utils.provider_cache.create_engine_adapter", return_value=primary),
        _patch_rate_limiter() as limiter_patch,
        _patch_token_helpers(),
        patch("ask_llm.core.task_executor.RequestProcessor") as mock_rp,
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


def test_process_global_tasks_bounded_calls():
    """B1 (runner-level): API calls are bounded by the retry budget.

    Every task always fails with a transient error. Total API calls must stay
    <= ``n_tasks * (max_retries + 1)``. This is the ARCHITECTURE_REVIEW.md P1
    acceptance criterion.
    """
    max_retries = 2
    n_tasks = 5
    tasks = [
        BatchTask(
            task_id=i,
            prompt="p",
            content="c",
            model_settings=ModelConfig(provider="primary", model="model-a"),
        )
        for i in range(n_tasks)
    ]
    processor = GlobalBatchProcessor(max_workers=4, max_retries=max_retries)
    cm = MagicMock()
    cm.unified_config.get_provider_config.return_value = ProviderConfig(
        api_provider="primary",
        api_base="https://api.primary.com/v1",
        api_key="sk-test",
        models=["model-a"],
    )
    primary = _make_provider("primary", "model-a")
    call_counter = {"n": 0}

    with (
        patch("ask_llm.core.progress_presenter.Progress"),
        patch("ask_llm.utils.provider_cache.create_engine_adapter", return_value=primary),
        _patch_rate_limiter() as limiter_patch,
        _patch_token_helpers(),
        patch("ask_llm.core.task_executor.RequestProcessor") as mock_rp,
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
    assert call_counter["n"] <= n_tasks * (max_retries + 1)


class TestAudit33LanePools:
    """Audit 3.3 (M6): per-(provider, model) lanes through process_global_tasks."""

    def _two_lane_tasks(self) -> list[BatchTask]:
        tasks = []
        for i in range(4):
            provider, model = ("primary", "model-a") if i % 2 == 0 else ("fallback", "model-b")
            tasks.append(
                BatchTask(
                    task_id=i,
                    prompt="p",
                    content="c",
                    model_settings=ModelConfig(provider=provider, model=model),
                )
            )
        return tasks

    def _config_manager(self) -> MagicMock:
        cm = MagicMock()
        cm.unified_config.get_provider_config.return_value = ProviderConfig(
            api_provider="primary",
            api_base="https://api.primary.com/v1",
            api_key="sk-test",
            models=["model-a", "model-b"],
        )
        return cm

    def test_multi_lane_run_returns_all_results_sorted(self):
        """Two providers run in concurrent lanes; every result is accounted for."""
        processor = GlobalBatchProcessor(max_workers=4)
        cm = self._config_manager()
        providers = {
            "primary/model-a": _make_provider("primary", "model-a"),
            "fallback/model-b": _make_provider("fallback", "model-b"),
        }

        with (
            patch("ask_llm.core.progress_presenter.Progress"),
            patch(
                "ask_llm.utils.provider_cache.create_engine_adapter",
                side_effect=lambda cfg, **kw: providers[f"{cfg.api_provider}/{cfg.models[0]}"],
            ),
            patch("ask_llm.core.batch_processor.get_global_rate_limiter") as lane_limiter,
            _patch_rate_limiter() as exec_limiter,
            _patch_token_helpers(),
            patch("ask_llm.core.task_executor.RequestProcessor") as mock_rp,
        ):
            lane_limiter.return_value.burst_for.return_value = 100
            exec_limiter.return_value.burst_for.return_value = 100

            proc = MagicMock()
            proc.process.return_value = iter(["ok"])
            mock_rp.return_value = proc

            results = processor.process_global_tasks(self._two_lane_tasks(), cm)

        assert [r.task_id for r in results] == [0, 1, 2, 3]
        assert all(r.status == TaskStatus.SUCCESS for r in results)
        # Two lanes => two per-lane pools, each <= max_workers.
        assert processor.last_metrics is not None
        assert processor.last_metrics.successful == 4

    def test_tight_lane_gets_fewer_slots_than_max_workers(self):
        """Lane sizing: burst=1 provider gets a 1-slot lane; others unaffected."""
        rate_config = RateLimitConfig(
            primary={"requests_per_minute": 60, "burst_size": 1},
            fallback={"requests_per_minute": 600, "burst_size": 8},
        )
        processor = GlobalBatchProcessor(max_workers=6, rate_limit_config=rate_config)
        tasks = self._two_lane_tasks()
        # primary burst 1, fallback burst 8.
        lanes = processor._build_lanes(tasks)
        assert lanes["primary:model-a"][0] == 1
        assert lanes["fallback:model-b"][0] == 6
