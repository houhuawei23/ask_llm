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
    """Plan 5.4: a failed initial reply KEEPS the seeded context in history
    (previously the generic rollback silently dropped it). M14/2.25: the
    re-added message is the templated content the seed used."""
    provider = FakeProvider(error=RuntimeError("boom"))

    session = ChatSession.from_initial_context(provider, model="m1", initial_context="ctx")

    assert [m.content for m in session.history.messages] == [
        "Please process the following text:\n\nctx"
    ]


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

    def test_rejected_command_not_recorded_for_repeat(self):
        """M16/2.25: `!!` must not repeat a command that was just rejected
        (e.g. for metacharacters) — _last_shell_cmd records only validated
        commands."""
        from unittest.mock import MagicMock, patch

        provider = FakeProvider()
        session = ChatSession.from_initial_context(provider, model="m1")

        with patch("ask_llm.core.chat.subprocess.run") as run:
            session._handle_shell_command("echo bad | pipe")  # rejected
            assert session._last_shell_cmd is None
            session._handle_shell_command("echo good")  # accepted
            assert session._last_shell_cmd == "echo good"
            session._handle_shell_command("!")  # repeat trigger: runs "echo good"
            executed = [c.args[0] for c in run.call_args_list]
        assert executed[-1] == ["echo", "good"]
        assert ["echo", "bad", "|", "pipe"] not in executed

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


class TestAudit54SessionEnhancements:
    """Plan 5.4: /save /resume round-trip, history trimming, /search escaping,
    seeded-context retention."""

    def _session(self):
        return ChatSession.from_initial_context(FakeProvider(), model="m1")

    def test_save_resume_roundtrip(self, tmp_path):
        import json

        session = self._session()
        session._send_message = lambda content: (
            session.history.add_message(MessageRole.USER, content)
            or session.history.add_message(MessageRole.ASSISTANT, "echo " + content)
        )
        session._send_message("first question")
        saved = tmp_path / "session.json"
        session._cmd_save(str(saved))

        data = json.loads(saved.read_text(encoding="utf-8"))
        assert data["provider"] == "fake"
        assert data["model"] == "m1"
        assert data["messages"][-1]["role"] == "assistant"

        # A fresh session resumes the file.
        fresh = ChatSession.from_initial_context(FakeProvider(), model="m1")
        fresh._cmd_resume(str(saved))
        roles = [m.role for m in fresh.history.messages]
        assert roles.count(MessageRole.USER) == 1
        assert roles.count(MessageRole.ASSISTANT) == 1
        assert fresh.history.messages[-1].content == "echo first question"

    def test_history_trims_oldest_round_over_budget(self, monkeypatch):
        from ask_llm.core.chat import ChatSession

        session = self._session()
        monkeypatch.setattr(ChatSession, "MAX_HISTORY_TOKENS", 30)
        for i in range(6):
            session.history.add_message(MessageRole.USER, f"question {i} " + "word " * 20)
            session.history.add_message(MessageRole.ASSISTANT, f"answer {i} " + "word " * 20)

        session._trim_history_to_budget()

        total = sum(len(m.content) for m in session.history.messages)
        assert total < 6 * 40  # some rounds evicted
        roles = [m.role for m in session.history.messages]
        assert roles.count(MessageRole.USER) == roles.count(MessageRole.ASSISTANT)
        # Rounds evict oldest-first.
        assert "question 0" not in session.history.messages[0].content

    def test_system_prompt_survives_trimming(self):
        session = self._session()
        session.history.add_message(MessageRole.SYSTEM, "system prompt stays")
        for _ in range(10):
            session.history.add_message(MessageRole.USER, "u" * 200)
            session.history.add_message(MessageRole.ASSISTANT, "a" * 200)
        session._trim_history_to_budget()
        assert any(m.role == MessageRole.SYSTEM for m in session.history.messages)

    def test_search_escapes_regex_metachars(self, capsys):
        session = self._session()
        session.history.add_message(MessageRole.USER, "price is (as of) 2026")

        # Previously raised re.error through the generic handler.
        session._cmd_search("(as of")
        out = capsys.readouterr().out
        assert "1 match" in out

    def test_seeded_context_kept_when_initial_reply_fails(self):
        provider = FakeProvider(error=RuntimeError("boom"))

        session = ChatSession.from_initial_context(
            provider, model="m1", initial_context="the seeded context"
        )

        roles = [m.role for m in session.history.messages]
        assert roles == [MessageRole.USER]
        # M14/2.25: re-added with the same templating the seed applied.
        assert session.history.messages[0].content.endswith("the seeded context")
