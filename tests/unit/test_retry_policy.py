"""Unit tests for RetryPolicy abstraction."""

from ask_llm.core.retry_policy import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TRANSIENT_KEYWORDS,
    RetryPolicy,
)


class TestRetryPolicy:
    def test_default_detects_transient_keywords(self):
        assert DEFAULT_RETRY_POLICY.is_retryable("Request timeout after 30s")
        assert DEFAULT_RETRY_POLICY.is_retryable("HTTP 429 Too Many Requests")
        assert DEFAULT_RETRY_POLICY.is_retryable("connection reset by peer")
        assert DEFAULT_RETRY_POLICY.is_retryable("overloaded_error")

    def test_default_rejects_non_transient(self):
        assert not DEFAULT_RETRY_POLICY.is_retryable("Invalid API key")
        assert not DEFAULT_RETRY_POLICY.is_retryable("model not found")
        assert not DEFAULT_RETRY_POLICY.is_retryable("")
        assert not DEFAULT_RETRY_POLICY.is_retryable(None)  # type: ignore[arg-type]    def test_custom_keywords(self):
        policy = RetryPolicy(transient_keywords=("mycloud_down",))
        assert policy.is_retryable("ERROR mycloud_down please retry")
        assert not policy.is_retryable("timeout")

    def test_default_transient_keywords_present(self):
        # Ensure core transient signals are always in the default set
        assert "429" in DEFAULT_TRANSIENT_KEYWORDS
        assert "timeout" in DEFAULT_TRANSIENT_KEYWORDS
        assert "overloaded_error" in DEFAULT_TRANSIENT_KEYWORDS
