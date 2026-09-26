"""Single error-semantics authority (P4.8, unified in 2.25 refactor).

One canonical rule table ``keyword -> (ErrorCategory, transient, fallback)``.
Every retry/escalation decision in the codebase derives from this table:

- ``classify_error_message`` — first matching rule (in table order) wins,
  producing the error category used in logs/reports.
- ``is_retryable_error`` — derives from the ``transient`` column; drives the
  bounded runner's retry decisions.
- ``should_fallback_for_error`` — derives from the ``fallback`` column:
  whether a *different provider/model* could resolve the failure. This is a
  different question from retryability (retrying the same config) — e.g. a
  billing error is terminal for the same key but a fallback provider may
  still have quota.

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
    """One error-signature keyword and its semantics.

    Attributes:
        keyword: Lowercase substring matched against the error message.
        category: High-level failure category.
        transient: Retrying the *same* config may succeed (drives retries).
        fallback: A *different* provider/model may succeed (drives fallback
            escalation). Defaults to True — only request/key-intrinsic
            failures set it False.
    """

    keyword: str
    category: ErrorCategory
    transient: bool
    fallback: bool = True


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
        # Authentication — terminal, and no fallback can fix a bad key.
        KeywordRule("401", a, False, fallback=False),
        KeywordRule("403", a, False, fallback=False),
        KeywordRule("authentication", a, False, fallback=False),
        KeywordRule("unauthorized", a, False, fallback=False),
        KeywordRule("invalid api key", a, False, fallback=False),
        KeywordRule("api key invalid", a, False, fallback=False),
        KeywordRule("authentication_error", a, False, fallback=False),
        KeywordRule("access denied", a, False, fallback=False),
        KeywordRule("invalid token", a, False, fallback=False),
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
        # Content filter — terminal, input-intrinsic (no fallback helps).
        KeywordRule("content filter", c, False, fallback=False),
        KeywordRule("content_filter", c, False, fallback=False),
        KeywordRule("content policy", c, False, fallback=False),
        KeywordRule("moderation", c, False, fallback=False),
        KeywordRule("safety", c, False, fallback=False),
        KeywordRule("blocked", c, False, fallback=False),
        KeywordRule("inappropriate content", c, False, fallback=False),
        KeywordRule("content rejected", c, False, fallback=False),
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
        # M6/2.25: the terminal cert/proxy signatures must be checked BEFORE
        # the wide transient words — "connection failed: invalid SSL
        # certificate" matched "connection" first and was retried forever even
        # though no retry can fix a bad cert or proxy config.
        KeywordRule("ssl", n, False),
        KeywordRule("certificate", n, False),
        KeywordRule("proxy", n, False),
        KeywordRule("connection", n, True),
        KeywordRule("connect", n, True),
        KeywordRule("network", n, True),
        KeywordRule("dns", n, True),
        KeywordRule("unreachable", n, True),
        KeywordRule("refused", n, True),
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
        # Validation — terminal and request-intrinsic (no fallback helps),
        # intentionally LAST: these single words match far too broadly to
        # preempt the transient signatures above.
        KeywordRule("validation", v, False, fallback=False),
        KeywordRule("invalid", v, False, fallback=False),
        KeywordRule("required", v, False, fallback=False),
        KeywordRule("missing", v, False, fallback=False),
        KeywordRule("not found in cache", v, False, fallback=False),
    )


ERROR_KEYWORD_RULES: tuple[KeywordRule, ...] = _rules()

# Derived: retryable keywords (drives retry_policy.DEFAULT_TRANSIENT_KEYWORDS).
TRANSIENT_KEYWORDS: tuple[str, ...] = tuple(r.keyword for r in ERROR_KEYWORD_RULES if r.transient)

# Derived: terminal keywords (M6/2.25). classify_error_message resolves the
# table with first-match-wins precedence, but the retry policy only saw the
# transient half — "connection failed: invalid SSL certificate" matched the
# wide "connection" transient keyword and burned the whole retry budget on a
# cert/proxy config error no retry can fix. The policy now treats a terminal
# keyword match as authoritative, mirroring the table's precedence.
TERMINAL_KEYWORDS: tuple[str, ...] = tuple(
    r.keyword for r in ERROR_KEYWORD_RULES if not r.transient
)

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


def is_retryable_error(error_message: str) -> bool:
    """Return True if *error_message* looks transient/retryable.

    Empty messages are treated as transient-by-default (audit 3.1): a blank
    ``str(e)`` (several SDK connection errors carry details only on
    attributes) previously classified terminal and the task died on its first
    attempt without a retry or fallback. A terminal keyword match wins over
    any transient match (M6/2.25) so e.g. "connection failed: invalid SSL
    certificate" is not retried via its "connection" word.
    """
    if not error_message:
        return True
    lower = error_message.lower()
    if any(
        keyword_matches(rule.keyword, lower) for rule in ERROR_KEYWORD_RULES if not rule.transient
    ):
        return False
    return any(
        keyword_matches(rule.keyword, lower) for rule in ERROR_KEYWORD_RULES if rule.transient
    )


# Categories derived from the table: every rule of these categories is marked
# ``fallback=False``, i.e. no other provider/model can resolve the failure.
_NO_FALLBACK_CATEGORIES: frozenset[ErrorCategory] = frozenset(
    rule.category for rule in ERROR_KEYWORD_RULES if not rule.fallback
)


def should_fallback_for_error(category: ErrorCategory) -> bool:
    """Return whether a failed attempt should try the next fallback config.

    Derived from the rule table: categories where every matching keyword is
    ``fallback=False`` (authentication, content filter, validation) can never
    be fixed by a different provider/model.
    """
    return category not in _NO_FALLBACK_CATEGORIES
