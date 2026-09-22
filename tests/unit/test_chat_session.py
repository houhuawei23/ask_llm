"""Unit tests for ChatSession bootstrap (from_initial_context)."""

from __future__ import annotations

from ask_llm.core.chat import ChatSession
from ask_llm.core.models import MessageRole
from ask_llm.core.protocols import ReasoningChunk


class FakeProvider:
    """Minimal LLMProviderProtocol double streaming a canned reply."""

    name = "fake"
    default_model = "fake-default"

    def __init__(self, chunks=None, error=None):
        self._chunks = chunks if chunks is not None else ["Hi there"]
        self._error = error
        self.calls = []

    def call(self, *, messages, temperature, model, stream):
        self.calls.append(
            {"messages": messages, "temperature": temperature, "model": model, "stream": stream}
        )
        if self._error:
            raise self._error
        return iter(self._chunks)


def test_from_initial_context_without_context_starts_empty():
    provider = FakeProvider()

    session = ChatSession.from_initial_context(provider, model="m1")

    assert session.history.messages == []
    assert session.model == "m1"
    assert provider.calls == []


def test_from_initial_context_renders_template_with_replace_semantics():
    # Regression: the old CLI assembly used str.format, which crashes on
    # literal braces (LaTeX/JSON) in the template or the context.
    provider = FakeProvider(chunks=["ok"])
    template = 'Summarize: {"key": 1}\n\n{content}'

    session = ChatSession.from_initial_context(
        provider,
        model="m1",
        system_prompt="sys",
        initial_context="data {x} and $y^2$",
        prompt_template=template,
    )

    roles = [m.role for m in session.history.messages]
    assert roles == [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT]
    user_msg = session.history.messages[1].content
    assert user_msg == 'Summarize: {"key": 1}\n\ndata {x} and $y^2$'
    assert session.history.messages[2].content == "ok"


def test_from_initial_context_appends_context_when_no_placeholder():
    provider = FakeProvider()

    ChatSession.from_initial_context(
        provider,
        model="m1",
        initial_context="body text",
        prompt_template="Plain template",
    )

    user_msg = session_user_message(provider)
    assert user_msg == "Plain template\n\nbody text"


def session_user_message(provider):
    call = provider.calls[0]
    return call["messages"][-1]["content"]


def test_from_initial_context_streams_reasoning_chunks_as_content():
    provider = FakeProvider(chunks=[ReasoningChunk(content="answer", reasoning="thinking"), "!"])

    session = ChatSession.from_initial_context(provider, model="m1", initial_context="ctx")

    assert session.history.messages[-1].role == MessageRole.ASSISTANT
    assert session.history.messages[-1].content == "answer!"


def test_from_initial_context_uses_resolved_model_for_call():
    provider = FakeProvider()

    ChatSession.from_initial_context(provider, model="resolved-model", initial_context="ctx")

    assert provider.calls[0]["model"] == "resolved-model"
    assert provider.calls[0]["stream"] is True


def test_from_initial_context_rolls_back_user_message_on_failure():
    provider = FakeProvider(error=RuntimeError("boom"))

    session = ChatSession.from_initial_context(provider, model="m1", initial_context="ctx")

    assert session.history.messages == []


class TestAudit46ShellMetachar:
    """Audit 4.6: !commands with shell metacharacters are rejected, not mangled."""

    def test_metachar_detection(self):
        from ask_llm.core.chat import ChatSession

        assert ChatSession._find_unquoted_metachar("echo hi | grep hi") == "|"
        assert ChatSession._find_unquoted_metachar("ls > out.txt") == ">"
        assert ChatSession._find_unquoted_metachar("echo $HOME") == "$"
        assert ChatSession._find_unquoted_metachar("echo `whoami`") == "`"
        # Quoted metacharacters are legitimate literal arguments.
        assert ChatSession._find_unquoted_metachar('grep "a|b" file.txt') is None
        assert ChatSession._find_unquoted_metachar("echo 'it; ok'") is None
        assert ChatSession._find_unquoted_metachar("plain command --flag") is None

    def test_pipe_command_rejected_not_argv_split(self):
        """'echo hi | grep hi' must not run with | as an argv literal."""
        from unittest.mock import MagicMock, patch

        provider = FakeProvider()
        session = ChatSession.from_initial_context(provider, model="m1")

        run = MagicMock()
        with patch("ask_llm.core.chat.subprocess.run", run):
            handled = session._handle_shell_command("echo hi | grep hi")

        assert handled is True
        run.assert_not_called()  # nothing executed

    def test_quoted_metachar_still_executes(self):
        from unittest.mock import MagicMock, patch

        provider = FakeProvider()
        session = ChatSession.from_initial_context(provider, model="m1")

        completed = MagicMock(returncode=0, stdout="a|b", stderr="")
        run = MagicMock(return_value=completed)
        with patch("ask_llm.core.chat.subprocess.run", run):
            handled = session._handle_shell_command('grep "a|b" file.txt')

        assert handled is True
        run.assert_called_once()
        # argv form: the quoted | is one literal argument, never a shell pipe.
        argv = run.call_args.args[0]
        assert argv == ["grep", "a|b", "file.txt"]
