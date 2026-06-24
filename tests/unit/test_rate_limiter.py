"""Tests for the configurable rate limiter and GlobalBatchProcessor burst cap."""

from __future__ import annotations

import pytest

from ask_llm.config.unified_config import ProviderRateLimitConfig, RateLimitConfig
from ask_llm.core.batch_models import BatchTask, ModelConfig
from ask_llm.core.batch_processor import GlobalBatchProcessor
from ask_llm.utils.rate_limiter import GlobalRateLimiter, get_global_rate_limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter_singleton():
    """Reset the singleton's config so tests are isolated from run order."""
    limiter = get_global_rate_limiter()
    previous = limiter._config
    limiter.configure(None)
    yield
    limiter.configure(previous)


def test_rate_limit_default_fallback():
    config = RateLimitConfig()
    limits = config.get_limits("unknown")
    assert limits.requests_per_minute == 60
    assert limits.burst_size == 10


def test_rate_limit_provider_override():
    config = RateLimitConfig(
        default_limits=ProviderRateLimitConfig(requests_per_minute=10, burst_size=2),
        deepseek={"requests_per_minute": 100, "burst_size": 20},
    )
    assert config.get_limits("deepseek").burst_size == 20
    assert config.get_limits("deepseek", "deepseek-chat").burst_size == 20
    assert config.get_limits("unknown").burst_size == 2


def test_rate_limit_model_specific_override():
    config = RateLimitConfig(
        **{
            "deepseek": {"requests_per_minute": 100, "burst_size": 20},
            "deepseek/deepseek-reasoner": {"requests_per_minute": 50, "burst_size": 5},
        },
    )
    assert config.get_limits("deepseek", "deepseek-chat").burst_size == 20
    assert config.get_limits("deepseek", "deepseek-reasoner").burst_size == 5


def test_burst_for_uses_configured_limits():
    config = RateLimitConfig(
        deepseek={"requests_per_minute": 100, "burst_size": 7},
    )
    limiter = get_global_rate_limiter(config)
    assert limiter.burst_for("deepseek", "deepseek-chat") == 7


def test_effective_max_workers_capped_by_min_burst():
    rate_config = RateLimitConfig(
        deepseek={"requests_per_minute": 100, "burst_size": 3},
        qwen={"requests_per_minute": 300, "burst_size": 30},
    )
    processor = GlobalBatchProcessor(max_workers=20, rate_limit_config=rate_config)
    tasks = [
        BatchTask(
            task_id=1,
            prompt="p",
            content="c",
            task_model_config=ModelConfig(provider="deepseek", model="deepseek-chat"),
        ),
        BatchTask(
            task_id=2,
            prompt="p",
            content="c",
            task_model_config=ModelConfig(provider="qwen", model="qwen-max"),
        ),
    ]
    assert processor._effective_max_workers(tasks) == 3


def test_effective_max_workers_respects_user_max():
    rate_config = RateLimitConfig(
        qwen={"requests_per_minute": 300, "burst_size": 100},
    )
    processor = GlobalBatchProcessor(max_workers=5, rate_limit_config=rate_config)
    tasks = [
        BatchTask(
            task_id=1,
            prompt="p",
            content="c",
            task_model_config=ModelConfig(provider="qwen", model="qwen-max"),
        ),
    ]
    assert processor._effective_max_workers(tasks) == 5


def test_effective_max_workers_at_least_one():
    rate_config = RateLimitConfig(
        deepseek={"requests_per_minute": 100, "burst_size": 1},
    )
    processor = GlobalBatchProcessor(max_workers=10, rate_limit_config=rate_config)
    tasks = [
        BatchTask(
            task_id=1,
            prompt="p",
            content="c",
            task_model_config=ModelConfig(provider="deepseek", model="deepseek-chat"),
        ),
    ]
    assert processor._effective_max_workers(tasks) == 1


def test_effective_max_workers_without_model_config():
    processor = GlobalBatchProcessor(max_workers=10)
    tasks = [BatchTask(task_id=1, prompt="p", content="c")]
    assert processor._effective_max_workers(tasks) == 10
