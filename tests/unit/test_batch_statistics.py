"""Unit tests for :meth:`BatchStatistics.from_results` — the unified aggregator.

Locks the single-source-of-truth behavior introduced in P1.5
(ARCHITECTURE_REVIEW.md §4.1.2): per-(provider, model) grouping, success/fail
counts, latency and token aggregation.
"""

from __future__ import annotations

from ask_llm.core.batch_models import BatchResult, BatchStatistics, ModelConfig, TaskStatus
from ask_llm.core.models import RequestMetadata


def _result(provider: str, model: str, *, status: TaskStatus, metadata=None) -> BatchResult:
    return BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        model_settings=ModelConfig(provider=provider, model=model),
        status=status,
        metadata=metadata,
    )


def _meta(latency: float, inp: int, out: int) -> RequestMetadata:
    return RequestMetadata(
        provider="p",
        model="m",
        temperature=0.7,
        latency=latency,
        input_tokens=inp,
        output_tokens=out,
    )


def test_from_results_groups_by_provider_and_model():
    results = [
        _result("openai", "gpt-4", status=TaskStatus.SUCCESS),
        _result("openai", "gpt-4", status=TaskStatus.FAILED),
        _result("deepseek", "chat", status=TaskStatus.SUCCESS),
    ]

    stats = BatchStatistics.from_results(results)

    assert set(stats.keys()) == {"openai/gpt-4", "deepseek/chat"}
    assert stats["openai/gpt-4"].total_tasks == 2
    assert stats["deepseek/chat"].total_tasks == 1


def test_from_results_counts_success_and_failure():
    results = [
        _result("p", "m", status=TaskStatus.SUCCESS),
        _result("p", "m", status=TaskStatus.SUCCESS),
        _result("p", "m", status=TaskStatus.FAILED),
    ]

    stats = BatchStatistics.from_results(results)["p/m"]

    assert stats.successful_tasks == 2
    assert stats.failed_tasks == 1


def test_from_results_aggregates_latency_and_tokens_from_successful_only():
    results = [
        _result("p", "m", status=TaskStatus.SUCCESS, metadata=_meta(1.0, 10, 20)),
        _result("p", "m", status=TaskStatus.SUCCESS, metadata=_meta(3.0, 5, 7)),
        # Failed result with metadata must NOT contribute to token/latency sums.
        _result("p", "m", status=TaskStatus.FAILED, metadata=_meta(100.0, 999, 999)),
    ]

    stats = BatchStatistics.from_results(results)["p/m"]

    assert stats.total_latency == 4.0
    assert stats.average_latency == 2.0
    assert stats.total_input_tokens == 15
    assert stats.total_output_tokens == 27


def test_from_results_handles_missing_metadata():
    results = [
        _result("p", "m", status=TaskStatus.SUCCESS),  # no metadata
        _result("p", "m", status=TaskStatus.SUCCESS, metadata=_meta(2.0, 3, 4)),
    ]

    stats = BatchStatistics.from_results(results)["p/m"]

    # Only the result with metadata contributes to latency/tokens.
    assert stats.total_latency == 2.0
    assert stats.average_latency == 2.0
    assert stats.total_input_tokens == 3
    assert stats.total_output_tokens == 4


def test_from_results_empty_input_returns_empty_dict():
    assert BatchStatistics.from_results([]) == {}


def test_from_results_all_failed_has_zero_latency():
    results = [_result("p", "m", status=TaskStatus.FAILED) for _ in range(3)]

    stats = BatchStatistics.from_results(results)["p/m"]

    assert stats.successful_tasks == 0
    assert stats.failed_tasks == 3
    assert stats.total_latency == 0.0
    assert stats.average_latency == 0.0
