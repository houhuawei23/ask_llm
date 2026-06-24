"""Tests for execution report generation and serialization."""

from datetime import datetime

import pytest

from ask_llm.core.batch_models import BatchResult, ModelConfig, TaskStatus
from ask_llm.core.execution_report import (
    ExecutionReport,
    build_report_from_batch_results,
)
from ask_llm.core.models import RequestMetadata
from ask_llm.core.telemetry import ErrorCategory


def _success_result(task_id: int, provider: str, model: str) -> BatchResult:
    return BatchResult(
        task_id=task_id,
        prompt="prompt",
        content="content",
        model_settings=ModelConfig(provider=provider, model=model),
        status=TaskStatus.SUCCESS,
        metadata=RequestMetadata(
            provider=provider,
            model=model,
            temperature=0.7,
            input_tokens=10,
            output_tokens=20,
            latency=1.5,
            timestamp=datetime.now(),
        ),
    )


def _failed_result(
    task_id: int,
    provider: str,
    model: str,
    error: str,
    category: ErrorCategory,
) -> BatchResult:
    return BatchResult(
        task_id=task_id,
        prompt="prompt",
        content="content",
        model_settings=ModelConfig(provider=provider, model=model),
        status=TaskStatus.FAILED,
        error=error,
        error_category=category,
    )


def test_build_report_from_batch_results_success():
    results = [_success_result(0, "deepseek", "deepseek-chat")]
    report = build_report_from_batch_results("batch", results)
    assert report.command == "batch"
    assert report.total_tasks == 1
    assert report.successful_tasks == 1
    assert report.failed_tasks == 0
    assert report.token_summary.total_input_tokens == 10
    assert report.token_summary.total_output_tokens == 20


def test_build_report_from_batch_results_failure_category():
    results = [
        _failed_result(
            0,
            "deepseek",
            "deepseek-chat",
            "401 Unauthorized",
            ErrorCategory.AUTHENTICATION,
        ),
    ]
    report = build_report_from_batch_results("trans", results)
    assert report.failed_tasks == 1
    assert report.failure_summary.total_failed_tasks == 1
    assert report.failure_summary.by_category["authentication"] == 1
    assert report.tasks[0].final_error_category == ErrorCategory.AUTHENTICATION


def test_build_report_uses_successful_result_when_multiple_models():
    # Legacy BatchProcessor may return the same task_id once per model.
    results = [
        _failed_result(0, "deepseek", "x", "timeout", ErrorCategory.TIMEOUT),
        _success_result(0, "qwen", "qwen-max"),
    ]
    report = build_report_from_batch_results("batch", results)
    assert report.total_tasks == 1
    assert report.successful_tasks == 1
    assert len(report.tasks[0].attempts) == 2


def test_report_serialization_roundtrip(tmp_path):
    results = [
        _success_result(0, "deepseek", "deepseek-chat"),
        _failed_result(1, "deepseek", "x", "rate limit", ErrorCategory.RATE_LIMIT),
    ]
    report = build_report_from_batch_results("paper", results)
    path = str(tmp_path / "report.json")
    report.to_json_file(path)
    loaded = ExecutionReport.from_json_file(path)
    assert loaded.command == report.command
    assert loaded.total_tasks == report.total_tasks
    assert loaded.failure_summary.by_category == report.failure_summary.by_category


def test_attempt_history_is_included_in_report():
    primary = ModelConfig(provider="deepseek", model="deepseek-chat")
    fallback = ModelConfig(provider="qwen", model="qwen-max")
    primary_result = BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        model_settings=primary,
        status=TaskStatus.FAILED,
        error="timeout",
        error_category=ErrorCategory.TIMEOUT,
    )
    fallback_result = BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        model_settings=fallback,
        status=TaskStatus.SUCCESS,
        metadata=RequestMetadata(
            provider="qwen",
            model="qwen-max",
            temperature=0.7,
            input_tokens=5,
            output_tokens=10,
            latency=1.0,
        ),
    )
    final = fallback_result.model_copy()
    final.attempt_history = [primary_result, fallback_result]

    report = build_report_from_batch_results("batch", [final])
    assert report.successful_tasks == 1
    assert len(report.tasks[0].attempts) == 2
    providers = {a.provider for a in report.tasks[0].attempts}
    assert providers == {"deepseek", "qwen"}
