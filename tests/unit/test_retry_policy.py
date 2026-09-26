"""Unit tests for is_retryable_error (the former RetryPolicy)."""

import pytest

from ask_llm.core.error_keywords import is_retryable_error


class TestRetryableErrors:
    def test_default_detects_transient_keywords(self):
        assert is_retryable_error("Request timeout after 30s")
        assert is_retryable_error("HTTP 429 Too Many Requests")
        assert is_retryable_error("connection reset by peer")
        assert is_retryable_error("overloaded_error")

    def test_default_rejects_non_transient(self):
        assert not is_retryable_error("Invalid API key")
        assert not is_retryable_error("model not found")
        assert not is_retryable_error("insufficient_quota")

    @pytest.mark.parametrize("message", ["", None])
    def test_empty_message_is_transient_by_default(self, message):
        """Audit 3.1: blank str(e) must not be terminal — retry once more."""
        assert is_retryable_error(message)

    def test_numeric_codes_match_word_boundaries(self):
        """Audit 3.1: "500" must not hijack "15000 tokens" / "context 5000"."""
        assert is_retryable_error("HTTP 500 internal error")
        assert not is_retryable_error("context length is 15000 tokens")
        assert not is_retryable_error("cost 1500.50 exceeded budget")
