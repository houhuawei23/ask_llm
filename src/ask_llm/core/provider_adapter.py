"""litellm-backed LLM provider adapter (replaces the llm-engine dependency, 2.27).

The adapter mirrors the llm-engine sync-path behavior ask_llm was built on —
flat return values (``str`` / :class:`ReasoningChunk` / iterators), stable error
message fragments (:mod:`ask_llm.core.error_keywords` classifies by keyword),
the DeepSeek ``thinking`` disable + JSON-mode gates — while delegating
transport, provider quirks, and ``reasoning_content`` normalization to litellm.

Grep invariant: ``import litellm`` appears ONLY inside :func:`_litellm` here —
importing litellm costs seconds and, by default, fetches a remote model-cost
map at import time, so the import must stay lazy and off the CLI startup path.
``ask_llm.utils.engine_facade`` is the only consumer of this module.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Generator
from typing import Any

from loguru import logger

from ask_llm.config.providers_catalog import load_first_providers_yml
from ask_llm.core.models import ProviderConfig
from ask_llm.core.protocols import ReasoningChunk, TokenUsage

__all__ = [
    "LiteLLMProviderAdapter",
    "litellm_model_string",
]

# Providers whose OpenAI-compatible endpoints expect the claude-code User-Agent.
_KIMI_PROVIDERS = frozenset({"kimi", "kimi-code"})
_KIMI_USER_AGENT = "claude-code/1.0.0"
# Ollama endpoints reject missing keys; the previous engine sent this placeholder.
_OLLAMA_PLACEHOLDER_KEY = "ollama"


def _litellm():
    """Lazy litellm import seam.

    Importing litellm costs seconds and fetches the remote model-cost map at
    import time by default; ask_llm ships its own pricing catalog, so the local
    bundled map is selected *before* the import and upgrade notices are
    suppressed after it. Tests patch this function to stub the module without
    importing litellm at all.
    """
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    litellm.suppress_debug_info = True
    return litellm


def litellm_model_string(api_provider: str, api_base: str, model: str) -> str:
    """Map ``(provider, model)`` to a litellm model string.

    Mirrors the llm-engine routing ask_llm has always exercised on its sync
    path (OpenAI-compatible chat completions against the configured
    ``api_base``):

    - ``deepseek`` → ``deepseek/{model}`` (litellm's DeepSeek route)
    - ``anthropic`` → ``anthropic/{model}`` (litellm's native Messages route)
    - everything else → ``openai/{model}`` + explicit ``api_base`` — including
      ``ollama``: Ollama serves an OpenAI-compatible endpoint at
      ``{base}/v1/chat/completions``, while litellm's native ``ollama/``
      prefix routes to the raw ``/api/generate`` endpoint, which would change
      the wire behavior. ``api_base`` is therefore passed through untouched.
    """
    del api_base  # routing is prefix-only; api_base is always sent explicitly
    provider = (api_provider or "").strip().lower()
    if provider == "deepseek":
        return f"deepseek/{model}"
    if provider == "anthropic":
        return f"anthropic/{model}"
    return f"openai/{model}"


def _deepseek_model_entry(model: str) -> dict[str, Any]:
    """Return the providers.yml catalog entry for a DeepSeek model (``{}`` if absent).

    Reference data only (model flags, no credentials), so the default search
    list — cwd included — matches what the previous llm-engine read. Any
    failure yields ``{}`` (gates off) rather than breaking adapter creation.
    """
    try:
        data, _source = load_first_providers_yml()
    except Exception as e:
        logger.debug(f"providers.yml catalog unavailable for deepseek entry: {e}")
        return {}
    if not data:
        return {}
    prov = (data.get("providers") or {}).get("deepseek")
    if not isinstance(prov, dict):
        return {}
    for entry in prov.get("models") or []:
        if isinstance(entry, dict) and entry.get("name") == model:
            return entry
    return {}


# Negation-aware JSON request sniffing, ported verbatim from the previous
# engine (llm_engine DeepSeekProvider._prompt_requests_json_output): prompts
# that explicitly do NOT want JSON must win over positive patterns.
_NEGATION_PATTERNS = [
    r"(?:不要|禁止|不允许|不应|不能|不可)[\s\S]{0,20}json",
    r"不(?:是|采用|使用)[\s\S]{0,20}json",
    r"(?:don't|do\s*not|not\s+.*|never|no\s+)[\s\S]{0,30}json",
    r"json[\s\S]{0,10}(?:不|禁止|不允许|no|not|never|don't)",
    r"(?:avoid|exclude|without)[\s\S]{0,20}json",
    r"markdown[\s\S]{0,30}not[\s\S]{0,10}json",
    r"not[\s\S]{0,10}json[\s\S]{0,10}but",
    r"直接以.*JSON",
    r"pure[\s\S]{0,10}json",
    r"full[\s\S]{0,10}json",
    r"整段[\s\S]{0,10}JSON",
]

_POSITIVE_PATTERNS = [
    r"(?:output|respond|return|give|provide|send)[\s\S]{0,20}json",
    r"json[\s\S]{0,15}(?:format|output|response|object)",
    r"(?:as|in)[\s\S]{0,10}json[\s\S]{0,10}format",
    r"json[\s\S]{0,10}object[\s\S]{0,10}only",
    r"reply[\s\S]{0,10}with[\s\S]{0,10}json",
    r"must[\s\S]{0,10}be[\s\S]{0,10}json",
    r"json[\s\S]{0,10}required",
    r"format[\s\S]{0,10}json[\s\S]{0,10}response",
]


def _prompt_requests_json_output(content: str) -> bool:
    """Check if a prompt explicitly requests JSON output (not just mentions it)."""
    lower = content.lower()
    for pattern in _NEGATION_PATTERNS:
        if re.search(pattern, lower):
            return False
    return any(re.search(pattern, lower) for pattern in _POSITIVE_PATTERNS)


def _message_reasoning(message: Any) -> str | None:
    """Extract non-streaming ``reasoning_content`` (stripped; ``None`` when absent)."""
    rc = getattr(message, "reasoning_content", None)
    if isinstance(rc, str) and rc.strip():
        return rc.strip()
    try:
        dumped = message.model_dump() if hasattr(message, "model_dump") else None
    except Exception:
        dumped = None
    if isinstance(dumped, dict):
        rc2 = dumped.get("reasoning_content")
        if isinstance(rc2, str) and rc2.strip():
            return rc2.strip()
    return None


def _delta_reasoning(delta: Any) -> str:
    """Extract a streaming ``reasoning_content`` delta (unstripped; ``""`` when absent)."""
    rc = getattr(delta, "reasoning_content", None)
    if isinstance(rc, str) and rc:
        return rc
    try:
        dumped = delta.model_dump() if hasattr(delta, "model_dump") else None
    except Exception:
        dumped = None
    if isinstance(dumped, dict):
        rc2 = dumped.get("reasoning_content")
        if isinstance(rc2, str) and rc2:
            return rc2
    return ""


def _usage_from(response: Any) -> TokenUsage | None:
    """Build :class:`TokenUsage` from a response's usage block (``None`` when absent)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if not isinstance(cache_hit, int):
        cache_hit = None
    return TokenUsage(input_tokens=prompt, output_tokens=completion, cache_hit_tokens=cache_hit)


def _fresh_http_client(lit: Any, timeout: float) -> Any:
    """Build a per-call litellm ``HTTPHandler`` (``None`` when unavailable).

    Reached through the litellm module handle so tests can stub it (a stub
    without the ``llms`` attribute yields ``None``) and so a litellm internal
    restructure degrades gracefully to litellm's own shared-client behavior.
    """
    try:
        import httpx

        handler_cls = lit.llms.custom_httpx.http_handler.HTTPHandler
        return handler_cls(timeout=httpx.Timeout(timeout, connect=5.0))
    except Exception as e:
        logger.debug(f"Per-call HTTP handler unavailable ({e}); using litellm default client.")
        return None


def _close_quietly(client: Any) -> None:
    """Best-effort ``close()`` on a per-call HTTP handler (hygiene, never raises)."""
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception as e:
            logger.debug(f"Per-call HTTP handler close() failed (ignored): {e}")


class LiteLLMProviderAdapter:
    """:class:`LLMProviderProtocol` adapter backed by ``litellm.completion``.

    Per-call parameters are assembled fresh from the (treated-as-immutable)
    ``ProviderConfig`` — the adapter never mutates shared state (the previous
    engine mutated its provider config per call, which was not thread-safe
    under the adapter cache).

    The DeepSeek gates (``thinking`` disable, JSON mode) are evaluated once at
    construction from the *default* model's providers.yml entry, matching the
    previous engine: per-call ``model=`` overrides do not re-evaluate them.

    Protocol members are plain attributes (``config`` is the untouched
    ``ProviderConfig`` — never mutated by this adapter; ``available_models``
    falls back to ``[default_model]`` when the config lists none).
    """

    def __init__(self, config: ProviderConfig, *, default_model: str | None = None) -> None:
        self.config = config
        self.name = config.api_provider
        models = list(config.models)
        if not default_model and not models:
            raise ValueError(
                f"Provider '{config.api_provider}' has no models configured "
                f"and no default_model was provided"
            )
        self.default_model = default_model if default_model else models[0]
        self.available_models = models if models else [self.default_model]

        provider = (config.api_provider or "").strip().lower()
        self._thinking_disabled = False
        self._json_output_enabled = False
        if provider == "deepseek":
            entry = _deepseek_model_entry(self.default_model)
            # Reasoning models marked ``reasoning: false`` must run with the
            # chain-of-thought disabled; they otherwise stream reasoning_content
            # for minutes, burn the max_tokens budget, and return empty output.
            self._thinking_disabled = not entry.get("reasoning", True)
            self._json_output_enabled = bool(
                (entry.get("functions") or {}).get("json_output", False)
            )

    def close(self) -> None:
        """No-op: litellm manages per-call HTTP clients itself.

        A real method keeps the adapter-cache teardown simple (it used to probe
        ``adapter._provider._client`` on the old engine shape).
        """

    def _api_key(self, provider: str) -> str:
        """Plain-text key for the litellm call (placeholder for keyless Ollama)."""
        key = self.config.get_api_key()
        if key.strip():
            return key
        if provider == "ollama":
            return _OLLAMA_PLACEHOLDER_KEY
        return key

    @staticmethod
    def _payload_requests_json(api_messages: list[dict[str, Any]]) -> bool:
        """True when any message content explicitly requests JSON output."""
        for message in api_messages:
            content = message.get("content", "")
            if isinstance(content, str) and _prompt_requests_json_output(content):
                return True
        return False

    def call(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        temperature: float | None = None,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | ReasoningChunk | Generator[str | ReasoningChunk, None, None]:
        """Call the provider API.

        Returns:
            Non-streaming: the response text (stripped), or a
            :class:`ReasoningChunk` (content + reasoning + provider-reported
            usage) when ``return_reasoning`` is set. Streaming: an iterator of
            content strings, or of :class:`ReasoningChunk` pairs when
            ``return_reasoning`` is set.

        Raises:
            ValueError: If neither *prompt* nor *messages* is provided.
            RuntimeError: On API failure — message fragments are stable so
                :mod:`ask_llm.core.error_keywords` can classify them.
        """
        return_reasoning = bool(kwargs.pop("return_reasoning", False))

        if messages:
            api_messages: list[dict[str, Any]] = messages
        elif prompt:
            api_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        model_name = model if model else self.default_model
        temp = temperature if temperature is not None else self.config.api_temperature
        provider = (self.config.api_provider or "").strip().lower()

        params: dict[str, Any] = {
            "model": litellm_model_string(
                self.config.api_provider, self.config.api_base, model_name
            ),
            "messages": api_messages,
            "temperature": temp,
            "stream": stream,
            "timeout": self.config.timeout,
            # Retries belong to ask_llm's BoundedRetryRunner — never double-retry.
            "num_retries": 0,
            "api_key": self._api_key(provider),
            "api_base": self.config.api_base,
        }
        if self.config.api_top_p is not None:
            params["top_p"] = self.config.api_top_p
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        for key in ("max_tokens", "top_p", "presence_penalty", "frequency_penalty"):
            if key in kwargs:
                params[key] = kwargs[key]

        # DeepSeek gates are frozen at construction (see class docstring).
        if self._thinking_disabled:
            # extra_body, not the top-level ``thinking`` param: litellm drops
            # ``{"type": "disabled"}`` on the deepseek route, extra_body
            # reaches the raw request body on every route we use.
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        if self._json_output_enabled and self._payload_requests_json(api_messages):
            params["response_format"] = {"type": "json_object"}
        if provider in _KIMI_PROVIDERS:
            params["default_headers"] = {"User-Agent": _KIMI_USER_AGENT}

        lit = _litellm()
        # Per-call HTTP handler: litellm's shared cached sync client races when
        # several sync streams start concurrently (httpcore reads a socket that
        # a sibling request's teardown closed → "[Errno 9] Bad file descriptor"
        # on the first chunk; litellm#14852 class). A fresh client per call has
        # no shared pool to corrupt — verified 7/7 concurrent streams vs 1/7 on
        # the shared client. Falls back to litellm's default (None) when its
        # internals move underneath the <2.0.0 pin.
        client = _fresh_http_client(lit, self.config.timeout)
        if client is not None:
            params["client"] = client
        try:
            if stream:
                try:
                    upstream = lit.completion(**params)
                except BaseException:
                    _close_quietly(client)
                    raise
                # The generator owns the client lifecycle from here on.
                return self._stream(upstream, return_reasoning=return_reasoning, http_client=client)
            try:
                response = lit.completion(**params)
            finally:
                _close_quietly(client)
            content, reasoning, usage = self._unpack(response)
            if return_reasoning:
                return ReasoningChunk(content=content, reasoning=reasoning or "", usage=usage)
            return content
        except lit.exceptions.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise RuntimeError("API authentication failed. Please check your API key.") from e
        except lit.exceptions.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise RuntimeError("API rate limit exceeded. Please try again later.") from e
        except lit.openai.APIError as e:
            # openai.APIError — the common base litellm's typed exceptions
            # (InternalServerError, Timeout, APIConnectionError, BadRequest…)
            # all subclass; matches the previous engine's generic API branch.
            mt = params.get("max_tokens")
            logger.error(f"API error: {e}  (model={model_name!r}, max_tokens={mt})")
            raise RuntimeError(f"API error: {e}  (model={model_name!r}, max_tokens={mt})") from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise RuntimeError(f"API call failed: {e}") from e

    @staticmethod
    def _unpack(response: Any) -> tuple[str, str | None, TokenUsage | None]:
        """Flatten a non-streaming litellm response to ``(content, reasoning, usage)``."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("provider returned an empty response")
        message = choices[0].message
        content = (getattr(message, "content", None) or "").strip()
        return content, _message_reasoning(message), _usage_from(response)

    @staticmethod
    def _stream(
        upstream: Any,
        *,
        return_reasoning: bool,
        http_client: Any = None,
    ) -> Generator[str | ReasoningChunk, None, None]:
        """Yield content strings (or :class:`ReasoningChunk` pairs) from a litellm stream.

        ``http_client`` (the per-call handler this stream was started on) is
        closed when the generator exits — completion, error, or consumer-abort.
        """
        try:
            for chunk in upstream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if not delta:
                    continue
                if return_reasoning:
                    content = getattr(delta, "content", None) or ""
                    reasoning = _delta_reasoning(delta)
                    if content or reasoning:
                        yield ReasoningChunk(content=content, reasoning=reasoning)
                else:
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
        except GeneratorExit:
            logger.debug("Stream generator closed")
            raise
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise RuntimeError(f"Stream failed: {e}") from e
        finally:
            _close_quietly(http_client)

    def test_connection(self) -> tuple[bool, str, float]:
        """Probe the provider API: ``(success, message, latency_seconds)``."""
        start = time.time()
        try:
            response = self.call(prompt="Hello", max_tokens=10, temperature=0.0)
            latency = time.time() - start
            preview = response if isinstance(response, str) else ""
            return True, f"Response: {preview[:50]}...", latency
        except Exception as e:
            latency = time.time() - start
            return False, str(e), latency
