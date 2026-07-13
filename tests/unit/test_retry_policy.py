"""Unit tests for RetryPolicy abstraction."""

from ask_llm.core.retry_policy import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TRANSIENT_KEYWORDS,
    ProviderRetryRegistry,
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
        assert not DEFAULT_RETRY_POLICY.is_retryable(None)  # type: ignore[arg-type]

    def test_should_retry_respects_max(self):
        policy = RetryPolicy(max_retries=2)
        assert policy.should_retry("timeout", attempt=0)
        assert policy.should_retry("timeout", attempt=1)
        # attempt == max_retries -> stop
        assert not policy.should_retry("timeout", attempt=2)

    def test_should_retry_non_transient_never(self):
        policy = RetryPolicy(max_retries=5)
        assert not policy.should_retry("authentication failed", attempt=0)

    def test_custom_keywords(self):
        policy = RetryPolicy(transient_keywords=("mycloud_down",))
        assert policy.is_retryable("ERROR mycloud_down please retry")
        assert not policy.is_retryable("timeout")

    def test_default_transient_keywords_present(self):
        # Ensure core transient signals are always in the default set
        assert "429" in DEFAULT_TRANSIENT_KEYWORDS
        assert "timeout" in DEFAULT_TRANSIENT_KEYWORDS
        assert "overloaded_error" in DEFAULT_TRANSIENT_KEYWORDS

    def test_as_callable(self):
        policy = RetryPolicy(transient_keywords=("timeout",))
        fn = policy.as_callable()
        assert fn("read timeout") is True
        assert fn("auth error") is False


class TestProviderRetryRegistry:
    def test_get_falls_back_to_default(self):
        registry = ProviderRetryRegistry()
        assert registry.get("unknown") is registry.default
        assert registry.get(None) is registry.default

    def test_set_and_get_override(self):
        registry = ProviderRetryRegistry()
        custom = RetryPolicy(max_retries=1, transient_keywords=("boom",))
        registry.set("anthropic", custom)
        assert registry.get("anthropic") is custom
        # Other providers still use default
        assert registry.get("openai") is registry.default

    def test_override_isolation(self):
        registry = ProviderRetryRegistry()
        registry.set("provider-a", RetryPolicy(transient_keywords=("aaa",)))
        policy_b = registry.get("provider-b")
        assert "aaa" not in policy_b.transient_keywords
