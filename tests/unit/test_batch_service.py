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


def _make_app_config_with_fallback():
    from ask_llm.core.models import AppConfig, FallbackConfig, ProviderConfig

    return AppConfig(
        default_provider="openai",
        providers={
            "openai": ProviderConfig(
                api_provider="openai",
                api_key="sk-test",
                api_base="https://api.openai.com/v1",
                models=["gpt-4"],
                fallback_to=[FallbackConfig(provider="fallback", model="fallback-model")],
            ),
            "fallback": ProviderConfig(
                api_provider="fallback",
                api_key="sk-fallback",
                api_base="https://fallback.example.com/v1",
                models=["fallback-model"],
            ),
        },
    )


def _make_batch_config_file(tmp_path):
    config_path = tmp_path / "batch.yml"
    config_path.write_text(
        "provider-models:\n"
        "  - provider: openai\n"
        "    models:\n"
        "      - gpt-4\n"
        'prompt: "Translate: {content}"\n'
        "contents:\n"
        '  - "hello"\n',
        encoding="utf-8",
    )
    return config_path


def test_run_batch_from_config_applies_fallback_chain(tmp_path):
    from ask_llm.services.batch_service import run_batch_from_config

    config_path = _make_batch_config_file(tmp_path)
    app_config = _make_app_config_with_fallback()
    config_manager = MagicMock()
    config_manager.config = app_config
    config_manager.current_provider_name = "openai"
    config_manager.get_default_model.return_value = "gpt-4"
    config_manager.get_provider_config.return_value = app_config.providers["openai"]

    processor = MagicMock()
    processor.calculate_statistics.return_value = {}

    with (
        patch("ask_llm.services.batch_service.run_global_batch_tasks") as mock_run,
        patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_adapter,
    ):
        mock_provider = MagicMock()
        mock_provider.test_connection.return_value = (True, "ok", 0.1)
        mock_adapter.return_value = mock_provider
        mock_run.return_value = ([], MagicMock())
        run_batch_from_config(
            str(config_path),
            app_config,
            config_manager,
            MagicMock(mode="prompt-contents", threads=1, retries=0),
            output_format="json",
            threads=1,
            retries=0,
            retry_delay=0.0,
            retry_delay_max=0.0,
            use_fallback=True,
        )

    assert mock_run.called
    tasks = mock_run.call_args.args[0]
    assert len(tasks) == 1
    assert len(tasks[0].fallback_model_configs) == 1
    assert tasks[0].fallback_model_configs[0].provider == "fallback"
    assert tasks[0].fallback_model_configs[0].model == "fallback-model"


def test_run_batch_from_config_skips_fallback_when_disabled(tmp_path):
    from ask_llm.services.batch_service import run_batch_from_config

    config_path = _make_batch_config_file(tmp_path)
    app_config = _make_app_config_with_fallback()
    config_manager = MagicMock()
    config_manager.config = app_config
    config_manager.current_provider_name = "openai"
    config_manager.get_default_model.return_value = "gpt-4"
    config_manager.get_provider_config.return_value = app_config.providers["openai"]

    processor = MagicMock()
    processor.calculate_statistics.return_value = {}

    with (
        patch("ask_llm.services.batch_service.run_global_batch_tasks") as mock_run,
        patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_adapter,
    ):
        mock_provider = MagicMock()
        mock_provider.test_connection.return_value = (True, "ok", 0.1)
        mock_adapter.return_value = mock_provider
        mock_run.return_value = ([], MagicMock())
        run_batch_from_config(
            str(config_path),
            app_config,
            config_manager,
            MagicMock(mode="prompt-contents", threads=1, retries=0),
            output_format="json",
            threads=1,
            retries=0,
            retry_delay=0.0,
            retry_delay_max=0.0,
            use_fallback=False,
        )

    tasks = mock_run.call_args.args[0]
    assert len(tasks) == 1
    assert tasks[0].fallback_model_configs == []
