"""Integration-style tests for batch checkpoint/resume."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig
from ask_llm.core.batch_checkpoint import BatchCheckpoint
from ask_llm.core.batch_models import TaskStatus
from ask_llm.services.batch_service import run_batch_from_config


def _make_batch_config(tmp_path: Path) -> Path:
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


def _make_app_config():
    from ask_llm.core.models import AppConfig, ProviderConfig

    return AppConfig(
        default_provider="openai",
        providers={
            "openai": ProviderConfig(
                api_provider="openai",
                api_key="sk-test",
                api_base="https://api.openai.com/v1",
                models=["gpt-4"],
            )
        },
    )


def test_batch_run_creates_checkpoint_for_failed_task(tmp_path):
    config_path = _make_batch_config(tmp_path)
    app_config = _make_app_config()
    config_manager = MagicMock()
    config_manager.config = app_config
    config_manager.current_provider_name = "openai"
    config_manager.get_default_model.return_value = "gpt-4"
    config_manager.get_provider_config.return_value = app_config.providers["openai"]

    result = BatchResult(
        task_id=0,
        prompt="Translate: {content}",
        content="hello",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
        error="API error",
        status=TaskStatus.FAILED,
    )
    processor = MagicMock()
    processor.calculate_statistics.return_value = {}

    with patch("ask_llm.core.command_runner.run_global_batch_tasks") as mock_run:
        mock_run.return_value = ([result], processor)
        with patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_adapter:
            mock_provider = MagicMock()
            mock_provider.test_connection.return_value = (True, "ok", 0.1)
            mock_adapter.return_value = mock_provider
            run_batch_from_config(
                str(config_path),
                app_config,
                config_manager,
                MagicMock(mode="prompt-content-pairs", threads=1, retries=0),
                output_format="json",
                threads=1,
                retries=0,
                retry_delay=0.0,
                retry_delay_max=0.0,
            )

    checkpoint_path = tmp_path / "batch.yml.checkpoint.json"
    assert checkpoint_path.exists()
    loaded = BatchCheckpoint.load(checkpoint_path)
    assert loaded.completed_task_ids == []
    assert len(loaded.failed_tasks) == 1


def test_batch_resume_skips_completed_tasks(tmp_path):
    config_path = _make_batch_config(tmp_path)
    checkpoint_path = tmp_path / "resume.json"
    prior_result = BatchResult(
        task_id=0,
        prompt="Translate: {content}",
        content="hello",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
        response="bonjour",
        status=TaskStatus.SUCCESS,
    )
    checkpoint = BatchCheckpoint.create(command="batch", config_digest=str(config_path))
    checkpoint.merge([prior_result])
    checkpoint.save(checkpoint_path)

    app_config = _make_app_config()
    config_manager = MagicMock()
    config_manager.config = app_config
    config_manager.current_provider_name = "openai"
    config_manager.get_default_model.return_value = "gpt-4"
    config_manager.get_provider_config.return_value = app_config.providers["openai"]

    processor = MagicMock()
    processor.calculate_statistics.return_value = {}

    with (
        patch("ask_llm.core.command_runner.run_global_batch_tasks") as mock_run,
        patch("ask_llm.utils.provider_cache.create_engine_adapter") as mock_adapter,
    ):
        mock_provider = MagicMock()
        mock_provider.test_connection.return_value = (True, "ok", 0.1)
        mock_adapter.return_value = mock_provider
        run_batch_from_config(
            str(config_path),
            app_config,
            config_manager,
            MagicMock(mode="prompt-content-pairs", threads=1, retries=0),
            output_format="json",
            threads=1,
            retries=0,
            retry_delay=0.0,
            retry_delay_max=0.0,
            resume_checkpoint_path=str(checkpoint_path),
        )

    mock_run.assert_not_called()
