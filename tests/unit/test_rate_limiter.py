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


def test_tight_provider_lane_does_not_throttle_other_lanes():
    """Audit 3.3 (M6): lanes are sized per (provider, model) burst.

    The old global min-burst cap let deepseek's burst=3 pin the whole batch
    (qwen included) to 3 workers. Now deepseek's lane is 3 and qwen's lane
    runs at the user's max_workers.
    """
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
            model_settings=ModelConfig(provider="deepseek", model="deepseek-chat"),
        ),
        BatchTask(
            task_id=2,
            prompt="p",
            content="c",
            model_settings=ModelConfig(provider="qwen", model="qwen-max"),
        ),
    ]
    lanes = processor._build_lanes(tasks)
    assert lanes["deepseek:deepseek-chat"][0] == 3
    assert lanes["qwen:qwen-max"][0] == 20  # capped by user's max_workers, not qwen's burst
    # Summary view: total slots across lanes.
    assert processor._effective_max_workers(tasks) == 23


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
            model_settings=ModelConfig(provider="qwen", model="qwen-max"),
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
            model_settings=ModelConfig(provider="deepseek", model="deepseek-chat"),
        ),
    ]
    assert processor._effective_max_workers(tasks) == 1


def test_effective_max_workers_without_model_config():
    processor = GlobalBatchProcessor(max_workers=10)
    tasks = [BatchTask(task_id=1, prompt="p", content="c")]
    assert processor._effective_max_workers(tasks) == 10


def test_acquire_timeout_defaults_to_60_without_config():
    """Without a RateLimitConfig, the acquire timeout stays at the historical 60s."""
    limiter = get_global_rate_limiter()
    limiter.configure(None)
    assert limiter.acquire_timeout("deepseek", "deepseek-chat") == 60.0


def test_acquire_timeout_reads_provider_config():
    """acquire_timeout reflects the configured provider/model limit."""
    limiter = get_global_rate_limiter()
    limiter.configure(
        RateLimitConfig(
            deepseek={"requests_per_minute": 100, "burst_size": 20, "acquire_timeout_seconds": 120},
        )
    )
    assert limiter.acquire_timeout("deepseek", "deepseek-chat") == 120.0
    # Unconfigured provider falls back to default_limits (60s by default).
    assert limiter.acquire_timeout("qwen", "qwen-max") == 60.0


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("siliconflow", (200, 30)),
        ("aliyun", (200, 30)),
        ("kimi-code", (60, 10)),
    ],
)
def test_catalog_providers_have_real_defaults(provider, expected):
    """Providers served by providers.yml must not fall through to the (60, 10)
    floor, which throttled paper's default concurrency."""

    limiter = GlobalRateLimiter()
    assert limiter._get_limit(provider, None) == expected


class TestAudit33LimiterHygiene:
    """Audit 3.3: bucket reconfigure in place; wait-warn state is instance-safe."""

    def test_reconfigure_preserves_accumulated_tokens(self):
        """A config change must not hand out a free full burst (A6)."""
        from ask_llm.utils.rate_limiter import _SyncTokenBucket

        bucket = _SyncTokenBucket(requests_per_minute=60, burst_size=2)
        assert bucket.acquire(timeout=1.0)
        assert bucket.acquire(timeout=1.0)
        # Bucket empty. Reconfigure to a bigger capacity: the leftover token
        # level carries over — no immediate full refill.
        bucket.reconfigure(requests_per_minute=6000, burst_size=50)
        assert bucket.acquire(timeout=0.0) is False

    def test_reconfigure_updates_limits_in_place(self):
        from ask_llm.utils.rate_limiter import _SyncTokenBucket

        bucket = _SyncTokenBucket(requests_per_minute=60, burst_size=2)
        bucket.reconfigure(requests_per_minute=120, burst_size=7)
        assert bucket.matches(120, 7)
        assert not bucket.matches(60, 2)

    def test_acquire_uses_reconfigured_bucket(self):
        """A live singleton bucket follows a configure() change without reset."""
        limiter = get_global_rate_limiter(
            RateLimitConfig(deepseek={"requests_per_minute": 6000, "burst_size": 3})
        )
        assert limiter.acquire("deepseek", "deepseek-chat", timeout=1.0)
        # Tighten the config; the same bucket object keeps serving.
        limiter.configure(RateLimitConfig(deepseek={"requests_per_minute": 6000, "burst_size": 5}))
        assert limiter.acquire("deepseek", "deepseek-chat", timeout=1.0)

    def test_last_wait_warn_is_instance_state(self):
        """Audit 3.3: the wait-warn de-dup map is per-instance, not a ClassVar."""
        limiter = get_global_rate_limiter()
        assert "_last_wait_warn" not in GlobalRateLimiter.__dict__
        assert isinstance(limiter._last_wait_warn, dict)
