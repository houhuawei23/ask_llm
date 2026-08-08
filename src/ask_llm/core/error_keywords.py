"""Single error-keyword rule table (P4.8).

One canonical mapping ``keyword -> (ErrorCategory, transient)`` consumed by:

- ``telemetry.classify_error`` — first matching rule (in table order) wins,
  producing the error category used in logs/reports.
- ``retry_policy.DEFAULT_TRANSIENT_KEYWORDS`` — derived as the keywords of all
  transient rules, driving retry decisions in the bounded runner.

Rule order matters: authentication is checked first, validation last, and the
transient-but-uncategorized server-error keywords (500/502/503/overloaded/...)
sit at the very end so they never hijack a more specific category.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCategory(str, Enum):
    """High-level failure category for an API call or task attempt."""

    SUCCESS = "success"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTENT_FILTER = "content_filter"
    MODEL_ERROR = "model_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class KeywordRule:
    """One error-signature keyword and its semantics."""

    keyword: str  # lowercase substring matched against the error message
    category: ErrorCategory
    transient: bool  # retrying the same call may succeed


def _rules() -> tuple[KeywordRule, ...]:
    # Short local aliases keep the table readable (one line per keyword).
    a = ErrorCategory.AUTHENTICATION
    r = ErrorCategory.RATE_LIMIT
    t = ErrorCategory.TIMEOUT
    c = ErrorCategory.CONTENT_FILTER
    m = ErrorCategory.MODEL_ERROR
    n = ErrorCategory.NETWORK_ERROR
    v = ErrorCategory.VALIDATION_ERROR
    u = ErrorCategory.UNKNOWN
    return (
        # Authentication — terminal (retrying with the same key never helps).
        KeywordRule("401", a, False),
        KeywordRule("403", a, False),
        KeywordRule("authentication", a, False),
        KeywordRule("unauthorized", a, False),
        KeywordRule("invalid api key", a, False),
        KeywordRule("api key invalid", a, False),
        KeywordRule("authentication_error", a, False),
        KeywordRule("access denied", a, False),
        KeywordRule("invalid token", a, False),
        # Rate limit — transient (backoff and retry).
        KeywordRule("429", r, True),
        KeywordRule("rate limit", r, True),
        KeywordRule("rate_limit", r, True),
        KeywordRule("too many requests", r, True),
        KeywordRule("throttled", r, True),
        KeywordRule("quota exceeded", r, True),
        KeywordRule("insufficient_quota", r, True),
        # Timeout — transient.
        KeywordRule("timeout", t, True),
        KeywordRule("timed out", t, True),
        KeywordRule("time out", t, True),
        KeywordRule("deadline exceeded", t, True),
        # Content filter — terminal.
        KeywordRule("content filter", c, False),
        KeywordRule("content_filter", c, False),
        KeywordRule("content policy", c, False),
        KeywordRule("moderation", c, False),
        KeywordRule("safety", c, False),
        KeywordRule("blocked", c, False),
        KeywordRule("inappropriate content", c, False),
        KeywordRule("content rejected", c, False),
        # Model error — terminal (bad request / context overflow).
        KeywordRule("model not found", m, False),
        KeywordRule("invalid model", m, False),
        KeywordRule("model error", m, False),
        KeywordRule("bad request", m, False),
        KeywordRule("invalid_request_error", m, False),
        KeywordRule("context length", m, False),
        KeywordRule("too long", m, False),
        KeywordRule("maximum context", m, False),
        # Network — mostly transient; TLS/proxy failures are usually config.
        KeywordRule("connection", n, True),
        KeywordRule("connect", n, True),
        KeywordRule("network", n, True),
        KeywordRule("dns", n, True),
        KeywordRule("unreachable", n, True),
        KeywordRule("refused", n, True),
        KeywordRule("ssl", n, False),
        KeywordRule("certificate", n, False),
        KeywordRule("proxy", n, False),
        # Validation — terminal.
        KeywordRule("validation", v, False),
        KeywordRule("invalid", v, False),
        KeywordRule("required", v, False),
        KeywordRule("missing", v, False),
        KeywordRule("not found in cache", v, False),
        # Transient server/overload signatures without a more specific
        # category (listed last so they never shadow the categories above).
        KeywordRule("overloaded_error", u, True),
        KeywordRule("overloaded", u, True),
        KeywordRule("temporarily unavailable", u, True),
        KeywordRule("try again", u, True),
        KeywordRule("500", u, True),
        KeywordRule("502", u, True),
        KeywordRule("503", u, True),
    )


ERROR_KEYWORD_RULES: tuple[KeywordRule, ...] = _rules()

# Derived: retryable keywords (drives retry_policy.DEFAULT_TRANSIENT_KEYWORDS).
TRANSIENT_KEYWORDS: tuple[str, ...] = tuple(r.keyword for r in ERROR_KEYWORD_RULES if r.transient)


def classify_error_message(error_message: str | None) -> ErrorCategory:
    """Classify a raw error message; first matching rule in table order wins."""
    if not error_message:
        return ErrorCategory.UNKNOWN
    text = error_message.lower()
    for rule in ERROR_KEYWORD_RULES:
        if rule.keyword in text:
            return rule.category
    return ErrorCategory.UNKNOWN
