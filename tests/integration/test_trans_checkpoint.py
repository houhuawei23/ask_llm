"""Integration-style tests for translation checkpoint/resume."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig
from ask_llm.core.batch_checkpoint import BatchCheckpoint
from ask_llm.core.batch_models import TaskStatus

# Import CLI first to resolve the trans/service circular import at module load time.
import ask_llm.cli.app
from ask_llm.services.translation_service import (
    TranslationOptions,
    TranslationService,
    _TextTranslationJob,
)


def _make_options(resume: bool = False) -> TranslationOptions:
    return TranslationOptions(
        target_language="zh",
        source_language="auto",
        style="formal",
        threads=1,
        max_parallel_files=1,
        retries=0,
        balance_translation_chunks=False,
        max_chunk_tokens=6000,
        min_chunk_merge_tokens=3000,
        max_output_tokens=8192,
        preserve_format=True,
        include_original=False,
        temperature=None,
        translatable_extensions=[".md", ".txt"],
        recursive_dir=False,
        resume=resume,
    )


def _make_job(tmp_path: Path) -> _TextTranslationJob:
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello world", encoding="utf-8")
    output_path = tmp_path / "doc_trans.txt"
    task = BatchTask(
        task_id=0,
        prompt="Translate: {content}",
        content="hello world",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
    )
    return _TextTranslationJob(
        file_path=str(file_path),
        file_type="text",
        chunks=[MagicMock()],
        tasks=[task],
        output_path=str(output_path),
    )


def test_translation_creates_checkpoint_on_failure(tmp_path):
    config_manager = MagicMock()
    service = TranslationService(
        config_manager=config_manager,
        unified_config=MagicMock(),
        provider="openai",
        model="gpt-4",
    )
    job = _make_job(tmp_path)
    failed_result = BatchResult(
        task_id=0,
        prompt="Translate: {content}",
        content="hello world",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
        error="API error",
        status=TaskStatus.FAILED,
    )
    processor = MagicMock()
    processor.last_metrics = MagicMock(retried=0)

    with patch("ask_llm.core.command_runner.run_global_batch_tasks") as mock_run:
        mock_run.return_value = ([failed_result], processor)
        with patch.object(service._text_translator, "export_text_file") as mock_export:
            mock_export.return_value = MagicMock(success=False)
            service._translate_and_export_text_file(
                job,
                _make_options(),
                force=False,
                stream=False,
                stream_api=True,
            )

    checkpoint_path = Path(f"{job.output_path}.trans_checkpoint.json")
    assert checkpoint_path.exists()
    loaded = BatchCheckpoint.load(checkpoint_path)
    assert len(loaded.failed_tasks) == 1
    assert loaded.failed_tasks[0].task_id == 0


def test_translation_resume_skips_completed_chunks(tmp_path):
    config_manager = MagicMock()
    service = TranslationService(
        config_manager=config_manager,
        unified_config=MagicMock(),
        provider="openai",
        model="gpt-4",
    )
    job = _make_job(tmp_path)
    prior_result = BatchResult(
        task_id=0,
        prompt="Translate: {content}",
        content="hello world",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
        response="你好世界",
        status=TaskStatus.SUCCESS,
    )
    checkpoint = BatchCheckpoint.create(command="trans", config_digest=job.file_path)
    checkpoint.merge([prior_result])
    checkpoint.save(f"{job.output_path}.trans_checkpoint.json")

    with (
        patch("ask_llm.core.command_runner.run_global_batch_tasks") as mock_run,
        patch.object(service._text_translator, "export_text_file") as mock_export,
    ):
        mock_export.return_value = MagicMock(success=True)
        service._translate_and_export_text_file(
            job,
            _make_options(resume=True),
            force=False,
            stream=False,
            stream_api=True,
        )

    mock_run.assert_not_called()
    mock_export.assert_called_once()
    passed_results = mock_export.call_args[0][1]
    assert len(passed_results) == 1
    assert passed_results[0].task_id == 0
