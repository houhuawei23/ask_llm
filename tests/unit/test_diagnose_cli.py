"""Tests for the `ask-llm diagnose` command."""

from datetime import datetime

import pytest
from typer.testing import CliRunner

from ask_llm.cli.app import app
from ask_llm.core.batch_models import BatchResult, ModelConfig, TaskStatus
from ask_llm.core.execution_report import build_report_from_batch_results
from ask_llm.core.models import RequestMetadata
from ask_llm.core.telemetry import ErrorCategory


runner = CliRunner()


def _make_report_file(tmp_path, results, command="batch"):
    report = build_report_from_batch_results(command, results)
    path = tmp_path / "report.json"
    report.to_json_file(str(path))
    return str(path)


def test_diagnose_successful_report(tmp_path):
    result = BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        model_settings=ModelConfig(provider="deepseek", model="deepseek-chat"),
        status=TaskStatus.SUCCESS,
        metadata=RequestMetadata(
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            input_tokens=10,
            output_tokens=20,
            latency=1.0,
        ),
    )
    path = _make_report_file(tmp_path, [result])
    response = runner.invoke(app, ["diagnose", path])
    assert response.exit_code == 0
    assert "Total tasks: 1" in response.output
    assert "Successful: 1" in response.output


def test_diagnose_failure_breakdown(tmp_path):
    result = BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        model_settings=ModelConfig(provider="deepseek", model="x"),
        status=TaskStatus.FAILED,
        error="401 Unauthorized",
        error_category=ErrorCategory.AUTHENTICATION,
    )
    path = _make_report_file(tmp_path, [result])
    response = runner.invoke(app, ["diagnose", path])
    assert response.exit_code == 0
    assert "Failed: 1" in response.output
    assert "authentication" in response.output
    assert "terminal error categories" in response.output


def test_diagnose_missing_report(tmp_path):
    path = str(tmp_path / "missing.json")
    response = runner.invoke(app, ["diagnose", path])
    assert response.exit_code == 1
    assert "Report not found" in response.output
