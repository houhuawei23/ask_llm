from __future__ import annotations

from dataclasses import dataclass, field

"""Retry policy abstraction for the bounded concurrent runner.

Centralizes the previously-hardcoded transient-error keyword list so retry
behavior can be customized per provider (e.g. Anthropic ``overloaded_error``)
without modifying the runner internals.
"""

# Default keywords indicating a transient / retryable error message.
DEFAULT_TRANSIENT_KEYWORDS: tuple[str, ...] = (
    "timeout",
    "connection",
    "network",
    "rate limit",
    "429",
    "503",
    "502",
    "500",
    # Common provider-specific overload signals
    "overloaded",
    "overloaded_error",
    "temporarily unavailable",
    "try again",
)


@dataclass
class RetryPolicy:
    """Decides whether an error is retryable and how many attempts to allow.

    Attributes:
        max_retries: Hard cap on retry attempts for any single task.
        transient_keywords: Lowercased substrings that mark an error transient.
    """

    max_retries: int = 3
    transient_keywords: tuple[str, ...] = DEFAULT_TRANSIENT_KEYWORDS

    def is_retryable(self, error_message: str) -> bool:
        """Return True if *error_message* looks transient/retryable."""
        if not error_message:
            return False
        lower = error_message.lower()
        return any(kw in lower for kw in self.transient_keywords)

    def should_retry(self, error_message: str, attempt: int) -> bool:
        """Return True if a failed task should be retried.

        Args:
            error_message: Error text from the failed result.
            attempt: Zero-based failure count so far.
        """
        if attempt >= self.max_retries:
            return False
        return self.is_retryable(error_message)

    def as_callable(self):
        """Return a ``Callable[[str], bool]`` usable as ``is_retryable_error``."""

        def _callable(error_message: str) -> bool:
            return self.is_retryable(error_message)

        return _callable


# Shared default policy; mirrors the historical hardcoded behavior plus a few
# provider-specific overload signals.
DEFAULT_RETRY_POLICY = RetryPolicy()


@dataclass
class ProviderRetryRegistry:
    """Per-provider retry policy registry.

    Falls back to :data:`DEFAULT_RETRY_POLICY` when a provider has no override.
    """

    default: RetryPolicy = field(default_factory=lambda: DEFAULT_RETRY_POLICY)
    overrides: dict[str, RetryPolicy] = field(default_factory=dict)

    def get(self, provider: str | None) -> RetryPolicy:
        """Return the retry policy for *provider*, falling back to default."""
        if provider and provider in self.overrides:
            return self.overrides[provider]
        return self.default

    def set(self, provider: str, policy: RetryPolicy) -> None:
        """Register a retry policy override for *provider*."""
        self.overrides[provider] = policy
