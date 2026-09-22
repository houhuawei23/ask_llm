"""Retry policy abstraction for the bounded concurrent runner.

Centralizes the previously-hardcoded transient-error keyword list so retry
behavior can be customized per provider (e.g. Anthropic ``overloaded_error``)
without modifying the runner internals.
"""

from __future__ import annotations

from dataclasses import dataclass

from ask_llm.core.error_keywords import TRANSIENT_KEYWORDS, keyword_matches

# Default keywords indicating a transient / retryable error message.
# Derived from the single keyword rule table (P4.8): every transient rule's
# keyword is retryable. This is a superset of the historical hardcoded list —
# rate-limit/timeout/network variants previously only used for categorization
# are now also retryable.
DEFAULT_TRANSIENT_KEYWORDS: tuple[str, ...] = TRANSIENT_KEYWORDS


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
        """Return True if *error_message* looks transient/retryable.

        Empty messages are treated as transient-by-default (audit 3.1): a
        blank ``str(e)`` (several SDK connection errors carry details only on
        attributes) previously classified terminal and the task died on its
        first attempt without a retry or fallback.
        """
        if not error_message:
            return True
        lower = error_message.lower()
        return any(keyword_matches(kw, lower) for kw in self.transient_keywords)


# Shared default policy; mirrors the historical hardcoded behavior plus a few
# provider-specific overload signals.
DEFAULT_RETRY_POLICY = RetryPolicy()
