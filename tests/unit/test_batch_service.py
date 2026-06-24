"""Unit tests for BatchService export and statistics helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig
from ask_llm.core.batch_models import BatchStatistics
from ask_llm.services.batch_service import BatchExportResult, BatchRunResult, BatchService


def _make_run_result(**kwargs):
    defaults = {
        "all_results": [],
        "model_statistics": {},
        "validated_models": [],
        "skipped_models": [],
        "original_tasks": [],
        "batch_mode": "prompt-content-pairs",
        "batch_config": {},
        "config_file": "batch.yml",
    }
    defaults.update(kwargs)
    return BatchRunResult(**defaults)


def _make_batch_config():
    cfg = MagicMock()
    cfg.default_output_format = "json"
    cfg.batch_output_dir = "batch_output"
    cfg.batch_results_dir = "batch_results"
    cfg.output_suffix = "_results"
    return cfg


def test_print_statistics_shows_per_model_summary():
    stats = BatchStatistics(total_tasks=2)
    stats.successful_tasks = 2
    stats.failed_tasks = 0
    stats.total_latency = 1.0
    stats.average_latency = 0.5
    stats.total_input_tokens = 10
    stats.total_output_tokens = 20

    run_result = _make_run_result(
        model_statistics={"openai/gpt-4": stats},
    )
    service = BatchService(run_result, _make_batch_config(), pricing_map={})

    with patch("ask_llm.services.batch_service.console") as mock_console:
        service.print_statistics()

    printed = " ".join(str(call) for call in mock_console.method_calls)
    assert "Statistics for openai/gpt-4" in printed
    assert "Total Tasks: 2" in printed
    assert "Successful: 2" in printed


def test_print_skipped_providers_shows_warning():
    run_result = _make_run_result(skipped_models=["deepseek/missing"])
    service = BatchService(run_result, _make_batch_config())

    with patch("ask_llm.services.batch_service.console") as mock_console:
        service.print_skipped_providers()

    printed = " ".join(str(call) for call in mock_console.method_calls)
    assert "Skipped 1 provider(s)" in printed
    assert "deepseek/missing" in printed


def test_print_skipped_providers_is_silent_when_none():
    run_result = _make_run_result()
    service = BatchService(run_result, _make_batch_config())

    with patch("ask_llm.services.batch_service.console") as mock_console:
        service.print_skipped_providers()

    mock_console.print_warning.assert_not_called()


def test_export_single_file(tmp_path):
    model_config = ModelConfig(provider="openai", model="gpt-4")
    result = BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        output_filename=None,
        model_settings=model_config,
    )
    result.status = MagicMock()
    stats = BatchStatistics(total_tasks=1)
    stats.successful_tasks = 1
    stats.failed_tasks = 0

    run_result = _make_run_result(
        all_results=[result],
        model_statistics={"openai/gpt-4": stats},
        validated_models=[model_config],
        config_file=str(tmp_path / "batch.yml"),
    )
    service = BatchService(run_result, _make_batch_config())

    with patch("ask_llm.services.batch_service.BatchResultExporter") as mock_exporter_cls:
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = str(tmp_path / "batch_results.json")
        mock_exporter_cls.return_value = mock_exporter
        export_result = service.export_results(None, "json")

    assert export_result.export_mode == "single"
    assert len(export_result.exported_paths) == 1
    mock_exporter.export.assert_called_once()


def test_export_separate_files_per_model():
    model_config_1 = ModelConfig(provider="openai", model="gpt-4")
    model_config_2 = ModelConfig(provider="openai", model="gpt-3.5-turbo")
    result = BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        output_filename=None,
        model_settings=model_config_1,
    )
    stats = BatchStatistics(total_tasks=1)
    stats.successful_tasks = 1
    stats.failed_tasks = 0

    run_result = _make_run_result(
        all_results=[result],
        model_statistics={"openai/gpt-4": stats},
        validated_models=[model_config_1, model_config_2],
    )
    service = BatchService(run_result, _make_batch_config())

    with patch(
        "ask_llm.services.batch_service.BatchResultExporter.export_multiple_models"
    ) as mock_export:
        mock_export.return_value = ["batch_results/openai_gpt-4.json"]
        export_result = service.export_results(None, "json", separate_files=True)

    assert export_result.export_mode == "separate"
    assert export_result.exported_paths == ["batch_results/openai_gpt-4.json"]


def test_export_split_files(tmp_path):
    model_config = ModelConfig(provider="openai", model="gpt-4")
    task = BatchTask(task_id=0, prompt="p", content="c")
    result = BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        output_filename=None,
        model_settings=model_config,
    )

    run_result = _make_run_result(
        all_results=[result],
        original_tasks=[task],
        validated_models=[model_config],
        batch_mode="prompt-content-pairs",
        config_file=str(tmp_path / "batch.yml"),
    )
    service = BatchService(run_result, _make_batch_config())

    with patch(
        "ask_llm.services.batch_service.BatchResultExporter.export_split_files"
    ) as mock_export:
        mock_export.return_value = ["batch_output/task_0.txt"]
        export_result = service.export_results(None, "json", split=True)

    assert export_result.export_mode == "split"
    assert export_result.exported_paths == ["batch_output/task_0.txt"]


def test_export_results_raises_when_no_results():
    run_result = _make_run_result()
    service = BatchService(run_result, _make_batch_config())

    with pytest.raises(ValueError, match="No providers were successfully processed"):
        service.export_results(None, "json")


def test_export_split_rejects_file_output(tmp_path):
    model_config = ModelConfig(provider="openai", model="gpt-4")
    result = BatchResult(
        task_id=0,
        prompt="p",
        content="c",
        output_filename=None,
        model_settings=model_config,
    )
    task = BatchTask(task_id=0, prompt="p", content="c")
    output_file = tmp_path / "out.json"
    output_file.write_text("{}", encoding="utf-8")

    run_result = _make_run_result(
        all_results=[result],
        original_tasks=[task],
        validated_models=[model_config],
        config_file=str(tmp_path / "batch.yml"),
    )
    service = BatchService(run_result, _make_batch_config())

    with pytest.raises(ValueError, match="output must be a directory"):
        service.export_results(str(output_file), "json", split=True)
