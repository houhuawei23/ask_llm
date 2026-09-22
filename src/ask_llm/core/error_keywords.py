"""Single error-keyword rule table (P4.8).

One canonical mapping ``keyword -> (ErrorCategory, transient)`` consumed by:

- ``telemetry.classify_error`` — first matching rule (in table order) wins,
  producing the error category used in logs/reports.
- ``retry_policy.DEFAULT_TRANSIENT_KEYWORDS`` — derived as the keywords of all
  transient rules, driving retry decisions in the bounded runner.

Rule order matters: authentication is checked first, then rate limits and the
transient server-error signatures (500/502/503/overloaded/...), with the wide
validation keywords (``invalid``/``required``/``missing``) at the very end.

The server-vs-validation order is the M2 fix: first-match-wins substring
matching used to classify e.g. ``"502: invalid upstream response"`` as
VALIDATION_ERROR (terminal — never retried, never fell back) because
``"invalid"`` matched before ``"502"``. Status-prefixed transient errors must
win over the wide words.

Numeric status keywords (``"401"``, ``"500"``, ...) match on word boundaries
only (audit 3.1): a bare ``"500"`` substring used to hijack messages like
``"context length is 15000"`` into a phantom server-error retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(str, Enum):
    """High-level failure category for an API call or task attempt."""

    SUCCESS = "success"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    BILLING = "billing"
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
        # Quota/billing exhaustion — terminal (audit 3.1): retrying the same
        # key never succeeds; escalation to fallback providers still applies.
        KeywordRule("quota exceeded", ErrorCategory.BILLING, False),
        KeywordRule("your current quota", ErrorCategory.BILLING, False),
        KeywordRule("insufficient_quota", ErrorCategory.BILLING, False),
        KeywordRule("billing", ErrorCategory.BILLING, False),
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
        # Transient server/overload signatures without a more specific
        # category. Deliberately BEFORE the wide validation words (M2):
        # "502: invalid upstream response" must be treated as transient.
        KeywordRule("overloaded_error", u, True),
        KeywordRule("overloaded", u, True),
        KeywordRule("temporarily unavailable", u, True),
        KeywordRule("try again", u, True),
        KeywordRule("internal server error", u, True),
        KeywordRule("500", u, True),
        KeywordRule("502", u, True),
        KeywordRule("503", u, True),
        KeywordRule("504", u, True),
        # Validation — terminal, and intentionally LAST: these single words
        # match far too broadly to preempt the transient signatures above.
        KeywordRule("validation", v, False),
        KeywordRule("invalid", v, False),
        KeywordRule("required", v, False),
        KeywordRule("missing", v, False),
        KeywordRule("not found in cache", v, False),
    )


ERROR_KEYWORD_RULES: tuple[KeywordRule, ...] = _rules()

# Derived: retryable keywords (drives retry_policy.DEFAULT_TRANSIENT_KEYWORDS).
TRANSIENT_KEYWORDS: tuple[str, ...] = tuple(r.keyword for r in ERROR_KEYWORD_RULES if r.transient)

# Numeric keywords ("401", "500", ...) match on word boundaries only, so
# "context length is 15000" no longer trips the "500" server-error rule.
_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def keyword_matches(keyword: str, lower_text: str) -> bool:
    """Boundary-aware keyword containment against an already-lowercased text."""
    if not keyword.isdigit():
        return keyword in lower_text
    pattern = _BOUNDARY_CACHE.get(keyword)
    if pattern is None:
        pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)")
        _BOUNDARY_CACHE[keyword] = pattern
    return bool(pattern.search(lower_text))


def classify_error_message(error_message: str | None) -> ErrorCategory:
    """Classify a raw error message; first matching rule in table order wins."""
    if not error_message:
        return ErrorCategory.UNKNOWN
    text = error_message.lower()
    for rule in ERROR_KEYWORD_RULES:
        if keyword_matches(rule.keyword, text):
            return rule.category
    return ErrorCategory.UNKNOWN
