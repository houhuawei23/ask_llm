"""Unit tests for ask_llm.core.provider_adapter (the litellm-backed adapter).

The litellm module handle is stubbed via ``_litellm`` so these tests never
import litellm (its import costs seconds and fetches a remote cost map).
Exception classes are stub hierarchies mirroring litellm's openai-backed MRO.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from ask_llm.core.models import ProviderConfig
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import ReasoningChunk, TokenUsage
from ask_llm.core.provider_adapter import (
    LiteLLMProviderAdapter,
    _prompt_requests_json_output,
    litellm_model_string,
)

# ---------------------------------------------------------------------------
# Stubs


def _exception_namespace() -> SimpleNamespace:
    """litellm.exceptions stand-in: same hierarchy shape, zero imports.

    ``openai.APIError`` is the common base litellm's typed exceptions subclass
    (verified against 1.82.6) — the adapter's generic API branch catches
    ``lit.openai.APIError`` through the module handle.
    """

    class APIError(Exception): ...

    class AuthenticationError(APIError): ...

    class RateLimitError(APIError): ...

    class BadRequestError(APIError): ...

    return SimpleNamespace(
        APIError=APIError,
        AuthenticationError=AuthenticationError,
        RateLimitError=RateLimitError,
        BadRequestError=BadRequestError,
        openai=SimpleNamespace(APIError=APIError),
    )


class _FakeMessage:
    def __init__(
        self,
        content: str | None = " hi ",
        reasoning_content: str | None = None,
        dump_reasoning: str | None = None,
    ):
        self.content = content
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content
        self._dump_reasoning = dump_reasoning

    def model_dump(self) -> dict:
        return {"reasoning_content": self._dump_reasoning}


class _FakeUsage:
    def __init__(self, prompt: int, completion: int, cache_hit: int | None = None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        if cache_hit is not None:
            self.prompt_cache_hit_tokens = cache_hit


def _response(
    content: str | None = " hi ",
    reasoning: str | None = None,
    dump_reasoning: str | None = None,
    usage: Any = "default",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_FakeMessage(content, reasoning, dump_reasoning))],
        usage=_FakeUsage(11, 7, 3) if usage == "default" else usage,
    )


class _FakeDelta(_FakeMessage):
    pass


def _stream_chunk(delta: Any) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def make_provider_config(provider: str = "openai", **overrides: Any) -> ProviderConfig:
    defaults: dict[str, Any] = {
        "api_provider": provider,
        "api_key": SecretStr("sk-test"),
        "api_base": "https://api.example.com/v1",
        "models": ["m1", "m2"],
        "api_temperature": 0.3,
        "api_top_p": None,
        "max_tokens": None,
        "timeout": 42.0,
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def make_adapter(provider: str = "openai", **overrides: Any) -> LiteLLMProviderAdapter:
    return LiteLLMProviderAdapter(make_provider_config(provider, **overrides))


class _CompletionStub:
    """Patch target context: swaps ``_litellm`` for a stub module handle."""

    def __init__(self, **attrs: Any):
        self.completion = MagicMock(**attrs)
        exc = _exception_namespace()
        # ``openai`` sits on the litellm module handle itself (litellm does
        # ``import openai``), mirroring how the adapter reaches the base class.
        self.lit = SimpleNamespace(completion=self.completion, exceptions=exc, openai=exc.openai)
        self.http_handlers: list[_FakeHTTPHandler] = []

    def with_http_handler(self) -> _CompletionStub:
        """Expose ``lit.llms.custom_httpx.http_handler.HTTPHandler`` (real-litellm shape)."""
        handlers = self.http_handlers

        def _handler_cls(**kwargs):
            handler = _FakeHTTPHandler(timeout=kwargs.get("timeout"))
            handlers.append(handler)
            return handler

        self.lit.llms = SimpleNamespace(
            custom_httpx=SimpleNamespace(http_handler=SimpleNamespace(HTTPHandler=_handler_cls))
        )
        return self

    def __enter__(self):
        p = patch("ask_llm.core.provider_adapter._litellm", return_value=self.lit)
        p.start()
        self._patcher = p
        return self

    def __exit__(self, *exc: Any):
        self._patcher.stop()
        return False


class _FakeHTTPHandler:
    """Stand-in for litellm's per-call ``HTTPHandler`` (records close)."""

    def __init__(self, timeout: Any = None):
        self.timeout = timeout
        self.closed = False

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Model-string mapping


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("deepseek", "deepseek/m1"),
        ("anthropic", "anthropic/m1"),
        ("kimi", "openai/m1"),
        ("kimi-code", "openai/m1"),
        ("openai", "openai/m1"),
        ("qwen", "openai/m1"),
        ("siliconflow", "openai/m1"),
        ("aliyun", "openai/m1"),
        ("ollama", "openai/m1"),  # openai/ prefix: litellm's ollama/ hits /api/generate
        ("totally-unknown", "openai/m1"),
    ],
)
def test_model_string_mapping(provider: str, expected: str):
    assert litellm_model_string(provider, "https://host/v1", "m1") == expected


def test_model_string_ignores_api_base():
    # Routing is prefix-only; api_base is sent explicitly on every call.
    assert litellm_model_string("deepseek", "https://custom.example.com/v1", "x") == "deepseek/x"


# ---------------------------------------------------------------------------
# Parameter assembly


def test_base_params_and_key_unwrapped():
    adapter = make_adapter()
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="hello", temperature=0.1)

    kwargs = stub.completion.call_args.kwargs
    assert kwargs["model"] == "openai/m1"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["temperature"] == 0.1  # per-call override wins
    assert kwargs["stream"] is False
    assert kwargs["timeout"] == 42.0
    assert kwargs["num_retries"] == 0  # retries belong to BoundedRetryRunner
    assert type(kwargs["api_key"]) is str  # SecretStr unwrapped exactly once
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["api_base"] == "https://api.example.com/v1"


def test_temperature_falls_back_to_config():
    with _CompletionStub(return_value=_response()) as stub:
        make_adapter(api_temperature=0.66).call(prompt="x")
    assert stub.completion.call_args.kwargs["temperature"] == 0.66


def test_model_override_changes_model_string():
    with _CompletionStub(return_value=_response()) as stub:
        make_adapter().call(prompt="x", model="m2")
    assert stub.completion.call_args.kwargs["model"] == "openai/m2"


def test_top_p_only_when_configured():
    with _CompletionStub(return_value=_response()) as stub:
        make_adapter().call(prompt="x")
    assert "top_p" not in stub.completion.call_args.kwargs

    with _CompletionStub(return_value=_response()) as stub:
        make_adapter(api_top_p=0.9).call(prompt="x")
    assert stub.completion.call_args.kwargs["top_p"] == 0.9


def test_max_tokens_kwargs_override_config():
    with _CompletionStub(return_value=_response()) as stub:
        make_adapter(max_tokens=1024).call(prompt="x", max_tokens=512)
    assert stub.completion.call_args.kwargs["max_tokens"] == 512

    with _CompletionStub(return_value=_response()) as stub:
        make_adapter(max_tokens=1024).call(prompt="x")
    assert stub.completion.call_args.kwargs["max_tokens"] == 1024

    with _CompletionStub(return_value=_response()) as stub:
        make_adapter().call(prompt="x")
    assert "max_tokens" not in stub.completion.call_args.kwargs


def test_penalty_kwargs_pass_through():
    with _CompletionStub(return_value=_response()) as stub:
        make_adapter().call(prompt="x", presence_penalty=0.5, frequency_penalty=0.25)
    kwargs = stub.completion.call_args.kwargs
    assert kwargs["presence_penalty"] == 0.5
    assert kwargs["frequency_penalty"] == 0.25


# ---------------------------------------------------------------------------
# Ollama / kimi specifics


def test_ollama_keyless_placeholder_and_base_passthrough():
    adapter = make_adapter("ollama", api_key=SecretStr(""), api_base="http://localhost:11434/v1")
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="x")

    kwargs = stub.completion.call_args.kwargs
    assert kwargs["api_key"] == "ollama"
    # Regression: no /v1 stripping — the OpenAI-compatible endpoint is the
    # wire behavior ask_llm has always used for Ollama.
    assert kwargs["api_base"] == "http://localhost:11434/v1"
    assert kwargs["model"] == "openai/m1"


def test_ollama_with_configured_key_uses_it():
    adapter = make_adapter("ollama", api_key=SecretStr("sk-local"))
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="x")
    assert stub.completion.call_args.kwargs["api_key"] == "sk-local"


def test_kimi_user_agent_header():
    for provider in ("kimi", "kimi-code"):
        with _CompletionStub(return_value=_response()) as stub:
            make_adapter(provider).call(prompt="x")
        headers = stub.completion.call_args.kwargs["default_headers"]
        assert headers["User-Agent"] == "claude-code/1.0.0"


def test_non_kimi_has_no_default_headers():
    with _CompletionStub(return_value=_response()) as stub:
        make_adapter("deepseek").call(prompt="x")
    assert "default_headers" not in stub.completion.call_args.kwargs


# ---------------------------------------------------------------------------
# Non-streaming returns & reasoning extraction


def test_non_stream_returns_stripped_str():
    with _CompletionStub(return_value=_response(content="  hello world  ")):
        assert make_adapter().call(prompt="x") == "hello world"


def test_return_reasoning_without_reasoning_content():
    with _CompletionStub(return_value=_response(content="answer", reasoning=None)):
        chunk = make_adapter().call(prompt="x", return_reasoning=True)
    assert isinstance(chunk, ReasoningChunk)
    assert chunk.content == "answer"
    assert chunk.reasoning == ""


def test_return_reasoning_with_reasoning_attribute():
    with _CompletionStub(return_value=_response(content="answer", reasoning="  thinking  ")):
        chunk = make_adapter().call(prompt="x", return_reasoning=True)
    assert isinstance(chunk, ReasoningChunk)
    assert chunk.reasoning == "thinking"


def test_reasoning_via_model_dump_fallback():
    with _CompletionStub(return_value=_response(content="answer", dump_reasoning="dumped")):
        chunk = make_adapter().call(prompt="x", return_reasoning=True)
    assert isinstance(chunk, ReasoningChunk)
    assert chunk.reasoning == "dumped"


# ---------------------------------------------------------------------------
# Real usage (TokenUsage)


def test_usage_extracted_when_present():
    with _CompletionStub(return_value=_response(usage=_FakeUsage(123, 45, 77))):
        chunk = make_adapter().call(prompt="x", return_reasoning=True)
    assert isinstance(chunk, ReasoningChunk)
    assert chunk.usage == TokenUsage(input_tokens=123, output_tokens=45, cache_hit_tokens=77)


def test_usage_none_when_absent():
    with _CompletionStub(return_value=_response(usage=None)):
        chunk = make_adapter().call(prompt="x", return_reasoning=True)
    assert isinstance(chunk, ReasoningChunk)
    assert chunk.usage is None


def test_usage_none_when_fields_not_int():
    with _CompletionStub(return_value=_response(usage=_FakeUsage("1", 2))):
        chunk = make_adapter().call(prompt="x", return_reasoning=True)
    assert isinstance(chunk, ReasoningChunk)
    assert chunk.usage is None


# ---------------------------------------------------------------------------
# Streaming


def test_stream_content_only():
    upstream = [
        _stream_chunk(_FakeDelta(content="a", reasoning_content="r1")),
        SimpleNamespace(choices=[]),
        _stream_chunk(None),
        _stream_chunk(_FakeDelta(content=None)),
        _stream_chunk(_FakeDelta(content="b")),
    ]
    with _CompletionStub(return_value=iter(upstream)):
        out = list(make_adapter().call(prompt="x", stream=True))
    assert out == ["a", "b"]  # reasoning pairs unwrapped; empty pieces skipped


def test_stream_reasoning_pairs():
    upstream = [
        _stream_chunk(_FakeDelta(content="a", reasoning_content="r1")),
        _stream_chunk(_FakeDelta(content="", reasoning_content="r2")),
        _stream_chunk(_FakeDelta(content="c", reasoning_content=None)),
        _stream_chunk(_FakeDelta(content="", reasoning_content=None)),  # both empty: skipped
        _stream_chunk(_FakeDelta(content="", dump_reasoning="r3")),  # dump fallback
    ]
    with _CompletionStub(return_value=iter(upstream)):
        out = list(make_adapter().call(prompt="x", stream=True, return_reasoning=True))
    assert out == [
        ReasoningChunk(content="a", reasoning="r1"),
        ReasoningChunk(content="", reasoning="r2"),
        ReasoningChunk(content="c", reasoning=""),
        ReasoningChunk(content="", reasoning="r3"),
    ]


def test_stream_delta_not_stripped():
    upstream = [_stream_chunk(_FakeDelta(content="  spaced  "))]
    with _CompletionStub(return_value=iter(upstream)):
        out = list(make_adapter().call(prompt="x", stream=True))
    assert out == ["  spaced  "]


def test_stream_error_mid_iteration():
    def _boom():
        yield _stream_chunk(_FakeDelta(content="ok"))
        raise ValueError("kaboom")

    with (
        _CompletionStub(return_value=_boom()),
        pytest.raises(RuntimeError, match=r"^Stream failed: kaboom$"),
    ):
        list(make_adapter().call(prompt="x", stream=True))


def test_stream_request_time_auth_error_is_mapped():
    # litellm issues the HTTP request eagerly; auth failures surface specific
    # fragments (keyword classification downstream works for either wording).
    with _CompletionStub() as stub:
        stub.completion.side_effect = stub.lit.exceptions.AuthenticationError("bad key")
        with pytest.raises(RuntimeError, match="API authentication failed"):
            make_adapter().call(prompt="x", stream=True)


# ---------------------------------------------------------------------------
# DeepSeek gates (frozen at construction from the default model)


def test_thinking_disabled_gate():
    # Gates freeze at construction — patch the catalog lookup BEFORE building.
    with patch(
        "ask_llm.core.provider_adapter._deepseek_model_entry",
        return_value={"name": "m1", "reasoning": False},
    ):
        adapter = make_adapter("deepseek")
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="x")
    # extra_body, NOT a top-level thinking param (litellm drops disabled on
    # the deepseek route); only our keys are asserted — litellm may co-mingle.
    extra = stub.completion.call_args.kwargs["extra_body"]
    assert extra["thinking"] == {"type": "disabled"}


def test_thinking_gate_off_when_reasoning_enabled():
    with patch(
        "ask_llm.core.provider_adapter._deepseek_model_entry",
        return_value={"name": "m1", "reasoning": True},
    ):
        adapter = make_adapter("deepseek")
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="x")
    assert "extra_body" not in stub.completion.call_args.kwargs


def test_deepseek_gates_never_for_other_providers():
    with patch(
        "ask_llm.core.provider_adapter._deepseek_model_entry",
        return_value={"name": "m1", "reasoning": False},
    ):
        adapter = make_adapter("siliconflow")
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="x")
    kwargs = stub.completion.call_args.kwargs
    assert "extra_body" not in kwargs
    assert "response_format" not in kwargs


def test_json_output_gate_positive_prompt():
    with patch(
        "ask_llm.core.provider_adapter._deepseek_model_entry",
        return_value={
            "name": "m1",
            "functions": {"json_output": True},
        },
    ):
        adapter = make_adapter("deepseek")
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="please output json data")
    assert stub.completion.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_json_output_gate_negated_prompt():
    with patch(
        "ask_llm.core.provider_adapter._deepseek_model_entry",
        return_value={
            "name": "m1",
            "functions": {"json_output": True},
        },
    ):
        adapter = make_adapter("deepseek")
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="不要 json，用普通文本")
    assert "response_format" not in stub.completion.call_args.kwargs


def test_json_output_gate_off():
    with patch(
        "ask_llm.core.provider_adapter._deepseek_model_entry",
        return_value={"name": "m1"},
    ):
        adapter = make_adapter("deepseek")
    with _CompletionStub(return_value=_response()) as stub:
        adapter.call(prompt="please output json data")
    assert "response_format" not in stub.completion.call_args.kwargs


# ---------------------------------------------------------------------------
# Error fragment mapping (stable strings for error_keywords classification)


def test_auth_error_fragment():
    with _CompletionStub() as stub:
        stub.completion.side_effect = stub.lit.exceptions.AuthenticationError("bad key")
        with pytest.raises(RuntimeError, match=r"^API authentication failed\.") as ei:
            make_adapter().call(prompt="x")
    assert isinstance(ei.value.__cause__, Exception)


def test_rate_limit_error_fragment():
    with _CompletionStub() as stub:
        stub.completion.side_effect = stub.lit.exceptions.RateLimitError("429")
        with pytest.raises(RuntimeError, match=r"^API rate limit exceeded\."):
            make_adapter().call(prompt="x")


def test_api_error_fragment_two_spaces_with_model_and_max_tokens():
    with _CompletionStub() as stub:
        stub.completion.side_effect = stub.lit.exceptions.BadRequestError("oops")
        with pytest.raises(RuntimeError) as ei:
            make_adapter(max_tokens=512).call(prompt="x", model="m2")
    # Two spaces after {e} — byte parity with the previous engine's log format.
    assert ei.value.args[0] == "API error: oops  (model='m2', max_tokens=512)"


def test_generic_error_fragment():
    with _CompletionStub(return_value=_response()) as stub:
        stub.completion.side_effect = ValueError("boom")
        with pytest.raises(RuntimeError, match=r"^API call failed: boom$"):
            make_adapter().call(prompt="x")


def test_empty_choices_maps_to_generic_failure():
    with (
        _CompletionStub(return_value=SimpleNamespace(choices=[])),
        pytest.raises(RuntimeError, match=r"^API call failed: "),
    ):
        make_adapter().call(prompt="x")


# ---------------------------------------------------------------------------
# Protocol surface


def test_protocol_attributes():
    pc = make_provider_config()
    adapter = LiteLLMProviderAdapter(pc, default_model="m2")
    assert adapter.config is pc
    assert adapter.name == "openai"
    assert adapter.default_model == "m2"
    assert adapter.available_models == ["m1", "m2"]
    adapter.close()  # no-op, must not raise


def test_available_models_fallback_to_default():
    adapter = LiteLLMProviderAdapter(make_provider_config(models=[]), default_model="solo")
    assert adapter.default_model == "solo"
    assert adapter.available_models == ["solo"]


def test_no_models_and_no_default_raises():
    with pytest.raises(ValueError, match="no models configured"):
        LiteLLMProviderAdapter(make_provider_config(models=[]))


def test_neither_prompt_nor_messages_raises():
    with (
        _CompletionStub(return_value=_response()),
        pytest.raises(ValueError, match="Either 'prompt' or 'messages'"),
    ):
        make_adapter().call()


def test_messages_passthrough_verbatim():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    with _CompletionStub(return_value=_response()) as stub:
        make_adapter().call(messages=msgs)
    assert stub.completion.call_args.kwargs["messages"] is msgs


# ---------------------------------------------------------------------------
# test_connection


def test_test_connection_success():
    with _CompletionStub(return_value=_response(content="pong")):
        ok, msg, latency = make_adapter().test_connection()
    assert ok is True
    assert msg.startswith("Response: pong")
    assert latency >= 0.0


def test_test_connection_failure():
    with _CompletionStub() as stub:
        stub.completion.side_effect = stub.lit.exceptions.AuthenticationError("nope")
        ok, msg, latency = make_adapter().test_connection()
    assert ok is False
    assert "API authentication failed" in msg
    assert latency >= 0.0


# ---------------------------------------------------------------------------
# JSON request sniffing (ported truth table)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("please output json data", True),
        ("return the result in json format", True),
        ("reply with json only", True),
        ("结果必须是 json object", True),
        ("不要 json，用纯文本", False),
        ("禁止输出 JSON", False),
        ("not json but markdown", False),
        ("give me markdown, avoid json", False),
        ("json is a nice word", False),
        ("plain text only", False),
    ],
)
def test_prompt_requests_json_output(text: str, expected: bool):
    assert _prompt_requests_json_output(text) is expected


# ---------------------------------------------------------------------------
# Laziness: importing this module must not import litellm


def test_litellm_not_imported_by_module():
    # Subprocess-isolated: other tests in the session may legitimately have
    # litellm in sys.modules already; only a fresh interpreter proves the
    # adapter module itself keeps the import lazy.
    code = (
        "import sys, ask_llm.core.provider_adapter; "
        "assert 'litellm' not in sys.modules, 'litellm imported at module import time'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Real-usage preference in RequestProcessor.process_with_metadata


class _UsageProvider:
    """Minimal protocol double whose non-streaming call reports real usage."""

    name: ClassVar[str] = "fake"
    default_model: ClassVar[str] = "m1"
    available_models: ClassVar[list[str]] = ["m1"]

    def __init__(self, chunk: ReasoningChunk | str):
        self._chunk = chunk

    @property
    def config(self):
        return make_provider_config()

    def call(self, **_kwargs: Any):
        return self._chunk


def test_process_with_metadata_prefers_real_usage():
    provider = _UsageProvider(
        ReasoningChunk("answer", "", TokenUsage(input_tokens=100, output_tokens=33))
    )
    result = RequestProcessor(provider).process_with_metadata("hello")
    assert result.metadata is not None
    assert result.metadata.input_tokens == 100
    assert result.metadata.output_tokens == 33


def test_process_with_metadata_falls_back_to_estimates():
    provider = _UsageProvider(ReasoningChunk("answer", "", None))
    result = RequestProcessor(provider).process_with_metadata("hello")
    assert result.metadata is not None
    assert result.metadata.input_tokens > 0  # estimated, not zero
    assert result.metadata.output_tokens > 0


# ---------------------------------------------------------------------------
# Per-call HTTP handler (litellm shared-client cold-start race, litellm#14852)


def test_no_client_key_without_llms_on_handle():
    # Stub without ``lit.llms`` → graceful fallback: no client param, litellm
    # uses its own default (shared) client.
    with _CompletionStub(return_value=_response()) as stub:
        make_adapter().call(prompt="x")
    assert "client" not in stub.completion.call_args.kwargs


def test_per_call_client_passed_and_closed_nonstream():
    stub = _CompletionStub(return_value=_response()).with_http_handler()
    with stub:
        make_adapter().call(prompt="x")
    assert len(stub.http_handlers) == 1
    assert stub.http_handlers[0].closed is True
    assert stub.completion.call_args.kwargs["client"] is stub.http_handlers[0]


def test_stream_closes_client_on_completion():
    upstream = [_stream_chunk(_FakeDelta(content="a")), _stream_chunk(_FakeDelta(content="b"))]
    stub = _CompletionStub(return_value=iter(upstream)).with_http_handler()
    with stub:
        gen = make_adapter().call(prompt="x", stream=True)
        list(gen)
    assert stub.http_handlers[0].closed is True


def test_stream_closes_client_on_error():
    def _boom():
        yield _stream_chunk(_FakeDelta(content="ok"))
        raise ValueError("kaboom")

    stub = _CompletionStub(return_value=_boom()).with_http_handler()
    with stub, pytest.raises(RuntimeError, match="Stream failed"):
        list(make_adapter().call(prompt="x", stream=True))
    assert stub.http_handlers[0].closed is True


def test_stream_abandon_closes_client():
    upstream = iter(
        [_stream_chunk(_FakeDelta(content="a")), _stream_chunk(_FakeDelta(content="b"))]
    )
    stub = _CompletionStub(return_value=upstream).with_http_handler()
    with stub:
        gen = make_adapter().call(prompt="x", stream=True)
        next(gen)
        gen.close()  # consumer aborts mid-stream
    assert stub.http_handlers[0].closed is True


def test_completion_error_closes_client_nonstream():
    stub = _CompletionStub().with_http_handler()
    stub.completion.side_effect = ValueError("boom")
    with stub, pytest.raises(RuntimeError, match="API call failed: boom"):
        make_adapter().call(prompt="x")
    assert stub.http_handlers[0].closed is True
