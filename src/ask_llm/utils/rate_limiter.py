"""全局同步速率限制器，用于限制跨文件/跨实例的 LLM API 并发请求。"""

from __future__ import annotations

import threading
import time
from typing import ClassVar, Optional


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
    """按 provider/model 共享的同步速率限制器。"""

    # provider 名称 -> (requests_per_minute, burst_size)
    DEFAULT_LIMITS: ClassVar[dict[str, tuple[int, int]]] = {
        "openai": (500, 50),
        "deepseek": (100, 20),
        "anthropic": (1000, 100),
        "ollama": (10000, 1000),
        "qwen": (300, 30),
    }

    _instance: ClassVar[Optional["GlobalRateLimiter"]] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "GlobalRateLimiter":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._limiters: dict[str, _SyncTokenBucket] = {}
                    cls._instance._lock = threading.Lock()
        return cls._instance

    def _key(self, provider: str, model: str | None = None) -> str:
        provider = provider.lower()
        if model:
            return f"{provider}:{model.lower()}"
        return provider

    def _get_limit(self, provider: str) -> tuple[int, int]:
        return self.DEFAULT_LIMITS.get(provider.lower(), (60, 10))

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
                rpm, burst = self._get_limit(provider)
                limiter = _SyncTokenBucket(rpm, burst)
                self._limiters[key] = limiter
        return limiter.acquire(timeout=timeout)

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


def get_global_rate_limiter() -> GlobalRateLimiter:
    return GlobalRateLimiter()
