"""End-to-end tests for the real execution chain: run_global_batch_tasks +
run_with_checkpoint with a fake (in-process) engine adapter.

These are the only tests that execute the actual scheduling path — the other
checkpoint/batch tests mock ``run_global_batch_tasks`` entirely, so retry
gating, fallback escalation and interrupt handling previously had zero
coverage. TokenCounter runs for real here (no stubs): chunk-estimate
regressions surface as wrong scheduling, not silent passes.
"""

from __future__ import annotations

import os
import signal
import threading

import pytest

from typing import ClassVar

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch_models import BatchResult, BatchTask, ModelConfig, TaskStatus
from ask_llm.core.command_runner import compute_checkpoint_digest, run_with_checkpoint
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.models import AppConfig, ProviderConfig
from ask_llm.utils.provider_cache import ProviderAdapterCache


class _FakeAdapter:
    """In-process provider adapter with programmable per-call behavior."""

    provider = "test"
    name = "test"
    available_models: ClassVar[list[str]] = []

    def __init__(self, config, default_model=None, behavior=None):
        self.config = config
        self.default_model = default_model
        self.behavior = behavior if behavior is not None else (lambda content: "ok")
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def test_connection(self):
        return True, "ok", 0.01

    def call(self, prompt=None, messages=None, temperature=None, model=None, stream=False, **kw):
        content = ""
        if messages:
            content = messages[-1].get("content", "")
        elif prompt:
            content = prompt
        with self.lock:
            self.calls.append(content)
            reply = self.behavior(content)
        if isinstance(reply, Exception):
            raise reply
        if stream:
            yield reply
        else:
            return reply


def _make_config_manager() -> ConfigManager:
    app_config = AppConfig(
        default_provider="test",
        default_model="test-model",
        providers={
            "test": ProviderConfig(
                api_provider="test",
                api_key="sk-real-key-123",
                api_base="https://test.example.com/v1",
                models=["test-model"],
            )
        },
    )
    return ConfigManager(app_config)


def _task(i: int) -> BatchTask:
    return BatchTask(
        task_id=i,
        prompt="Do: {content}",
        content=f"payload-{i}",
        model_settings=ModelConfig(provider="test", model="test-model"),
    )


@pytest.fixture(autouse=True)
def _clear_adapter_cache():
    ProviderAdapterCache.clear()
    yield
    ProviderAdapterCache.clear()


def test_transient_failures_are_retried_to_success(monkeypatch):
    """Retry gate: a 429-style first attempt is retried; results end SUCCESS."""
    attempts: dict[str, int] = {}

    def behavior(content: str):
        # The executor sends the rendered prompt ("Do: payload-N") — key by id.
        payload_id = content.rsplit("payload-", 1)[-1]
        attempts[payload_id] = attempts.get(payload_id, 0) + 1
        if attempts[payload_id] == 1:
            raise RuntimeError("Error code: 429 - rate limit exceeded")
        return f"done {content}"

    def fake_factory(config, default_model=None):
        return _FakeAdapter(config, default_model, behavior)

    monkeypatch.setattr("ask_llm.utils.provider_cache.create_engine_adapter", fake_factory)
    results, _processor = run_global_batch_tasks(
        [_task(i) for i in range(4)],
        _make_config_manager(),
        max_workers=2,
        max_retries=2,
        retry_delay=0.01,
        retry_delay_max=0.02,
        show_progress=False,
    )

    assert [r.status for r in sorted(results, key=lambda r: r.task_id)] == [TaskStatus.SUCCESS] * 4
    # Every task was attempted twice: one transient failure + one success.
    assert all(attempts[str(i)] == 2 for i in range(4))


def test_sigint_interrupt_preserves_checkpoint_and_resume_completes(monkeypatch, tmp_path):
    """Full lifecycle: Ctrl-C mid-run drains in-flight work, persists a
    checkpoint, and a resumed run finishes the remainder on the same digest."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("SIGINT graceful-drain requires the main thread")

    config_manager = _make_config_manager()
    checkpoint_path = str(tmp_path / "e2e.checkpoint.json")
    digest = compute_checkpoint_digest(None, [_task(i) for i in range(20)])

    triggered = threading.Event()
    call_count = {"n": 0}

    def slow_interrupting_behavior(content: str) -> str:
        with threading.Lock():
            call_count["n"] += 1
            n = call_count["n"]
        if n >= 3 and not triggered.is_set():
            triggered.set()
            os.kill(os.getpid(), signal.SIGINT)
        import time

        time.sleep(0.02)
        return f"done {content}"

    def slow_factory(config, default_model=None):
        return _FakeAdapter(config, default_model, slow_interrupting_behavior)

    monkeypatch.setattr("ask_llm.utils.provider_cache.create_engine_adapter", slow_factory)
    outcome = run_with_checkpoint(
        command="batch",
        config_digest=digest,
        checkpoint_path=checkpoint_path,
        tasks=[_task(i) for i in range(20)],
        config_manager=config_manager,
        resume=False,
        max_retries=0,
        max_workers=2,
        show_progress=False,
    )

    assert outcome.interrupted is True
    assert os.path.exists(checkpoint_path), "interrupted run must persist the checkpoint"
    assert 1 <= len(outcome.results) < 20
    interrupted_results = {r.task_id for r in outcome.results if r.status == TaskStatus.SUCCESS}

    # Resume: fast adapter, same digest -> only the remainder runs.
    def fast_factory(config, default_model=None):
        return _FakeAdapter(config, default_model, lambda content: f"done {content}")

    monkeypatch.setattr("ask_llm.utils.provider_cache.create_engine_adapter", fast_factory)
    resumed = run_with_checkpoint(
        command="batch",
        config_digest=digest,
        checkpoint_path=checkpoint_path,
        tasks=[_task(i) for i in range(20)],
        config_manager=config_manager,
        resume=True,
        max_retries=0,
        max_workers=4,
        show_progress=False,
    )

    assert not resumed.interrupted
    assert resumed.checkpoint_deleted, "clean full success must unlink the checkpoint"
    successes = {r.task_id for r in resumed.results if r.status == TaskStatus.SUCCESS}
    assert successes == set(range(20)), "resume must complete every task exactly once"
    assert interrupted_results <= successes


def test_on_result_streams_each_success_incrementally(monkeypatch):
    """D6 contract at the runner boundary: every success reaches on_result."""
    streamed: list[int] = []

    def factory(config, default_model=None):
        return _FakeAdapter(config, default_model, lambda content: f"done {content}")

    monkeypatch.setattr("ask_llm.utils.provider_cache.create_engine_adapter", factory)
    results, _ = run_global_batch_tasks(
        [_task(i) for i in range(5)],
        _make_config_manager(),
        max_workers=2,
        max_retries=0,
        show_progress=False,
        on_result=lambda r: streamed.append(r.task_id),
    )
    assert sorted(streamed) == sorted(r.task_id for r in results)


def test_terminal_auth_failure_is_not_retried(monkeypatch):
    """Auth errors are terminal: one attempt, no retry, no fallback burn."""

    def auth_behavior(content: str):
        raise RuntimeError("401 Unauthorized: invalid api key")

    calls = {"n": 0}
    orig_init = _FakeAdapter.__init__

    def counting_init(self, config, default_model=None, behavior=None):
        orig_init(self, config, default_model, behavior or auth_behavior)

        orig_call = self.call

        def counting_call(*args, **kwargs):
            with self.lock:
                calls["n"] += 1
            return orig_call(*args, **kwargs)

        self.call = counting_call

    monkeypatch.setattr(_FakeAdapter, "__init__", counting_init)
    monkeypatch.setattr(
        "ask_llm.utils.provider_cache.create_engine_adapter",
        lambda config, default_model=None: _FakeAdapter(config, default_model),
    )
    results, _ = run_global_batch_tasks(
        [_task(0), _task(1)],
        _make_config_manager(),
        max_workers=2,
        max_retries=3,
        show_progress=False,
    )
    assert all(r.status == TaskStatus.FAILED for r in results)
    # Two tasks, one attempt each — auth must not consume the retry budget.
    assert calls["n"] == 2


def _unused(*args):  # pragma: no cover - keeps BatchResult import referenced
    return BatchResult
