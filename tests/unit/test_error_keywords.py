"""Unit tests for the single error-keyword rule table (P4.8)."""

from ask_llm.core.error_keywords import (
    ERROR_KEYWORD_RULES,
    ErrorCategory,
    classify_error_message,
    is_retryable_error,
)


class TestClassify:
    def test_precedence_auth_first(self):
        assert classify_error_message("401 invalid api key: timeout") == (
            ErrorCategory.AUTHENTICATION
        )

    def test_categories(self):
        assert classify_error_message("429 too many requests") == ErrorCategory.RATE_LIMIT
        assert classify_error_message("connection timed out") == ErrorCategory.TIMEOUT
        assert classify_error_message("blocked by content filter") == (ErrorCategory.CONTENT_FILTER)
        assert classify_error_message("context length exceeded, input too long") == (
            ErrorCategory.MODEL_ERROR
        )
        assert classify_error_message("dns resolution failed") == ErrorCategory.NETWORK_ERROR
        assert classify_error_message("validation error: field required") == (
            ErrorCategory.VALIDATION_ERROR
        )

    def test_unknown_fallback(self):
        assert classify_error_message("something weird happened") == ErrorCategory.UNKNOWN
        assert classify_error_message(None) == ErrorCategory.UNKNOWN
        assert classify_error_message("") == ErrorCategory.UNKNOWN


class TestTransientDerivation:
    def test_historical_keywords_still_transient(self):
        """Keywords from the pre-P4.8 hardcoded list remain retryable."""
        transient = {r.keyword for r in ERROR_KEYWORD_RULES if r.transient}
        for kw in (
            "timeout",
            "connection",
            "network",
            "rate limit",
            "429",
            "503",
            "502",
            "500",
            "overloaded",
            "overloaded_error",
            "temporarily unavailable",
            "try again",
        ):
            assert kw in transient

    def test_terminal_keywords_not_transient(self):
        for rule in ERROR_KEYWORD_RULES:
            if rule.category in (
                ErrorCategory.AUTHENTICATION,
                ErrorCategory.CONTENT_FILTER,
                ErrorCategory.VALIDATION_ERROR,
                ErrorCategory.MODEL_ERROR,
            ):
                assert not rule.transient, f"{rule.keyword} should be terminal"


def test_cert_and_proxy_errors_not_retried_via_connection_keyword():
    """M6/2.25: 'connection failed: invalid SSL certificate' must be terminal —
    the terminal cert/proxy rules are checked before the wide transient
    'connection'/'connect' rules, so a bad cert/proxy config can't burn the
    retry budget."""
    assert not is_retryable_error("connection failed: invalid SSL certificate")
    assert not is_retryable_error("connection error: proxy authentication required")
    # Genuine transient connection failures remain retryable.
    assert is_retryable_error("connection refused")
    assert is_retryable_error("connection timed out")
