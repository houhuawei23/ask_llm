"""全局同步速率限制器，用于限制跨文件/跨实例的 LLM API 并发请求。"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, ClassVar

from loguru import logger

if TYPE_CHECKING:
    from ask_llm.config.unified_config import RateLimitConfig


class _SyncTokenBucket:
    """线程安全的同步 token bucket。"""

    def __init__(self, requests_per_minute: int, burst_size: int) -> None:
        self._rate = requests_per_minute / 60.0
        self._capacity = max(1, burst_size)
        self._tokens = float(self._capacity)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def acquire(self, timeout: float | None = None) -> bool:
        """获取一个 token，必要时阻塞等待；timeout 为秒。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_update
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_update = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

                if deadline is not None:
                    remaining = deadline - now
                    if remaining <= 0:
                        return False
                    wait_time = min(remaining, (1.0 - self._tokens) / self._rate)
                else:
                    wait_time = (1.0 - self._tokens) / self._rate

                if wait_time <= 0:
                    wait_time = 0.01

                self._condition.wait(timeout=wait_time)


class GlobalRateLimiter:
    """按 provider/model 共享的同步速率限制器。

    支持从 ``RateLimitConfig`` 读取限流参数，从而让用户通过 ``default_config.yml``
    或环境变量覆盖 provider/model 级别的 RPM 与 burst。
    """

    # provider 名称 -> (requests_per_minute, burst_size)
    DEFAULT_LIMITS: ClassVar[dict[str, tuple[int, int]]] = {
        "openai": (500, 50),
        "deepseek": (100, 20),
        "anthropic": (1000, 100),
        "ollama": (10000, 1000),
        "qwen": (300, 30),
    }

    _instance: ClassVar[GlobalRateLimiter | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()
    _limiters: dict[str, _SyncTokenBucket]
    _lock: threading.Lock
    _config: RateLimitConfig | None

    def __new__(cls) -> GlobalRateLimiter:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._limiters = {}
                    cls._instance._lock = threading.Lock()
                    cls._instance._config = None
        return cls._instance

    def configure(self, config: RateLimitConfig | None) -> None:
        """Set the active rate-limit configuration."""
        with self._lock:
            self._config = config

    def _key(self, provider: str, model: str | None = None) -> str:
        provider = provider.lower()
        if model:
            return f"{provider}:{model.lower()}"
        return provider

    def _get_limit(self, provider: str, model: str | None = None) -> tuple[int, int]:
        if self._config is not None:
            limits = self._config.get_limits(provider, model)
            return limits.requests_per_minute, limits.burst_size
        return self.DEFAULT_LIMITS.get(provider.lower(), (60, 10))

    def burst_for(self, provider: str, model: str | None = None) -> int:
        """Return the configured burst size for provider/model."""
        return self._get_limit(provider, model)[1]

    def acquire(
        self,
        provider: str,
        model: str | None = None,
        timeout: float | None = 60.0,
    ) -> bool:
        """为指定 provider/model 获取一个请求许可。"""
        key = self._key(provider, model)
        with self._lock:
            limiter = self._limiters.get(key)
            if limiter is None:
                rpm, burst = self._get_limit(provider, model)
                limiter = _SyncTokenBucket(rpm, burst)
                self._limiters[key] = limiter
            else:
                # If config changed, recreate the bucket with new limits.
                rpm, burst = self._get_limit(provider, model)
                if limiter._capacity != max(1, burst) or limiter._rate != rpm / 60.0:
                    limiter = _SyncTokenBucket(rpm, burst)
                    self._limiters[key] = limiter

        start = time.monotonic()
        acquired = limiter.acquire(timeout=timeout)
        elapsed = time.monotonic() - start
        if acquired and elapsed > 0.05:
            rpm, burst = self._get_limit(provider, model)
            logger.warning(
                f"Rate limiter waited {elapsed:.2f}s for {key} "
                f"(RPM={rpm}, burst={burst}). Consider lowering concurrency or raising limits."
            )
        return acquired

    def set_limit(
        self,
        provider: str,
        requests_per_minute: int,
        burst_size: int,
        model: str | None = None,
    ) -> None:
        """运行时调整某个 provider/model 的限流参数。"""
        key = self._key(provider, model)
        with self._lock:
            self._limiters[key] = _SyncTokenBucket(requests_per_minute, burst_size)


def get_global_rate_limiter(config: RateLimitConfig | None = None) -> GlobalRateLimiter:
    """Return the singleton rate limiter, optionally configuring it."""
    limiter = GlobalRateLimiter()
    if config is not None:
        limiter.configure(config)
    return limiter
