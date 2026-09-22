"""Unit tests for TranslationService fallback wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.batch_models import BatchTask, ModelConfig
from ask_llm.core.models import AppConfig, FallbackConfig, ProviderConfig
from ask_llm.core.text_splitter import TextChunk
from ask_llm.services.translation_service import TranslationOptions, TranslationService


def _make_app_config_with_fallback() -> AppConfig:
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


def _make_options(use_fallback: bool = True) -> TranslationOptions:
    return TranslationOptions(
        target_language="zh",
        source_language="en",
        style="technical",
        threads=1,
        max_parallel_files=1,
        retries=0,
        balance_translation_chunks=False,
        max_chunk_tokens=2400,
        max_output_tokens=2000,
        preserve_format=True,
        include_original=False,
        temperature=0.7,
        translatable_extensions=[".txt", ".md"],
        recursive_dir=False,
        use_fallback=use_fallback,
    )


def _make_service(app_config: AppConfig | None = None) -> TranslationService:
    config_manager = MagicMock()
    unified_config = MagicMock()
    unified_config.file.translated_suffix = ".translated"
    return TranslationService(
        config_manager=config_manager,
        unified_config=unified_config,
        provider="openai",
        model="gpt-4",
        app_config=app_config,
    )


def test_prepare_text_file_applies_fallback_chain(tmp_path: Path):
    service = _make_service(_make_app_config_with_fallback())
    input_file = tmp_path / "test.txt"
    input_file.write_text("hello world", encoding="utf-8")
    options = _make_options(use_fallback=True)

    chunk = TextChunk(content="hello world", chunk_id=0, start_pos=0, end_pos=11, metadata={})
    task = BatchTask(
        task_id=0,
        prompt="Translate: {content}",
        content="hello world",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
    )

    with (
        patch(
            "ask_llm.services.translation_service.TextSplitter.detect_file_type",
            return_value="text",
        ),
        patch("ask_llm.services.text_file_translator.FileHandler.read", return_value="hello world"),
        patch(
            "ask_llm.services.text_file_translator.plain_text_chunks_by_tokens",
            return_value=[chunk],
        ),
        patch(
            "ask_llm.services.text_file_translator.rebalance_translation_chunks",
            return_value=[chunk],
        ),
        patch("ask_llm.services.text_file_translator.Translator") as mock_translator_cls,
    ):
        mock_translator = MagicMock()
        mock_translator.create_translation_tasks.return_value = [task]
        mock_translator_cls.return_value = mock_translator

        job = service._text_translator.prepare(
            str(input_file),
            options,
            output=None,
            output_is_dir=False,
            effective_suffix=".translated",
            glossary_pairs=[],
            stream=False,
        )

    assert job is not None
    assert len(job.tasks) == 1
    assert len(job.tasks[0].fallback_model_configs) == 1
    assert job.tasks[0].fallback_model_configs[0].provider == "fallback"
    assert job.tasks[0].fallback_model_configs[0].model == "fallback-model"


def test_prepare_text_file_skips_fallback_when_disabled(tmp_path: Path):
    service = _make_service(_make_app_config_with_fallback())
    input_file = tmp_path / "test.txt"
    input_file.write_text("hello world", encoding="utf-8")
    options = _make_options(use_fallback=False)

    chunk = TextChunk(content="hello world", chunk_id=0, start_pos=0, end_pos=11, metadata={})
    task = BatchTask(
        task_id=0,
        prompt="Translate: {content}",
        content="hello world",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
    )

    with (
        patch(
            "ask_llm.services.translation_service.TextSplitter.detect_file_type",
            return_value="text",
        ),
        patch("ask_llm.services.text_file_translator.FileHandler.read", return_value="hello world"),
        patch(
            "ask_llm.services.text_file_translator.plain_text_chunks_by_tokens",
            return_value=[chunk],
        ),
        patch(
            "ask_llm.services.text_file_translator.rebalance_translation_chunks",
            return_value=[chunk],
        ),
        patch("ask_llm.services.text_file_translator.Translator") as mock_translator_cls,
    ):
        mock_translator = MagicMock()
        mock_translator.create_translation_tasks.return_value = [task]
        mock_translator_cls.return_value = mock_translator

        job = service._text_translator.prepare(
            str(input_file),
            options,
            output=None,
            output_is_dir=False,
            effective_suffix=".translated",
            glossary_pairs=[],
            stream=False,
        )

    assert job is not None
    assert len(job.tasks) == 1
    assert job.tasks[0].fallback_model_configs == []


def test_output_validated_before_first_api_call(tmp_path: Path):
    """Audit 2.1: colliding targets must fail before ANY translate call runs."""
    service = _make_service(_make_app_config_with_fallback())
    for name in ("one.md", "two.md"):
        (tmp_path / name).write_text("content " + name, encoding="utf-8")
    options = _make_options()

    def _boom(*_a, **_k):
        raise AssertionError("translate_and_export must not run after a target collision")

    with (
        patch.object(service._text_translator, "translate_and_export", _boom),
        pytest.raises(Exception, match="same output path"),
    ):
        service.translate_files(
            [str(tmp_path / "one.md"), str(tmp_path / "two.md")],
            options,
            output=str(tmp_path / "combined.md"),
        )


def test_existing_output_refused_before_first_api_call(tmp_path: Path):
    """Audit 2.1: an existing target without --force fails before any spend."""
    service = _make_service(_make_app_config_with_fallback())
    (tmp_path / "one.md").write_text("content", encoding="utf-8")
    existing = tmp_path / "exists.md"
    existing.write_text("prior", encoding="utf-8")
    options = _make_options()

    def _boom(*_a, **_k):
        raise AssertionError("translate_and_export must not run after target refusal")

    with (
        patch.object(service._text_translator, "translate_and_export", _boom),
        pytest.raises(Exception, match="--force"),
    ):
        service.translate_files(
            [str(tmp_path / "one.md")],
            options,
            output=str(existing),
        )


class TestPartialChunkFailure:
    """Audit 2.2: a file with any failed chunk is partial, not a clean success."""

    def _make_translator(self):
        from ask_llm.services.text_file_translator import TextFileTranslator

        return TextFileTranslator(
            MagicMock(),
            provider="openai",
            model="gpt-4",
            app_config=_make_app_config_with_fallback(),
        )

    def _job(self, tmp_path: Path):
        from ask_llm.services.text_file_translator import TextTranslationJob

        chunks = [
            TextChunk(content="First chunk", chunk_id=0, start_pos=0, end_pos=11, metadata={}),
            TextChunk(content="Second chunk", chunk_id=1, start_pos=12, end_pos=24, metadata={}),
        ]
        return TextTranslationJob(
            file_path=str(tmp_path / "doc.md"),
            file_type="markdown",
            output_path=str(tmp_path / "doc_trans.md"),
            chunks=chunks,
            tasks=[],
        )

    def _results(self):
        from ask_llm.core.batch_models import BatchResult, ModelConfig, TaskStatus
        from ask_llm.core.models import RequestMetadata

        mc = ModelConfig(provider="openai", model="gpt-4")
        meta = RequestMetadata(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            input_tokens=10,
            output_tokens=15,
            latency=0.5,
        )
        return [
            BatchResult(
                task_id=0,
                prompt="p",
                content="First chunk",
                model_settings=mc,
                response="译文一",
                status=TaskStatus.SUCCESS,
                metadata=meta,
            ),
            BatchResult(
                task_id=1,
                prompt="p",
                content="Second chunk",
                model_settings=mc,
                response="",
                status=TaskStatus.FAILED,
                error="rate limit",
                metadata=meta,
            ),
        ]

    def test_partial_chunk_failure_counts_file_failed(self, tmp_path: Path):
        translator = self._make_translator()
        result = translator.export_text_file(
            self._job(tmp_path),
            self._results(),
            preserve_format=True,
            include_original=False,
            force=True,
        )
        assert result.partial is True
        assert result.success is False
        assert "1 chunk" in (result.error or "")

    def test_all_success_is_clean(self, tmp_path: Path):
        from ask_llm.core.batch_models import BatchResult, ModelConfig, TaskStatus
        from ask_llm.core.models import RequestMetadata

        translator = self._make_translator()
        mc = ModelConfig(provider="openai", model="gpt-4")
        meta = RequestMetadata(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            input_tokens=10,
            output_tokens=15,
            latency=0.5,
        )
        results = [
            BatchResult(
                task_id=0,
                prompt="p",
                content="First chunk",
                model_settings=mc,
                response="译文一",
                status=TaskStatus.SUCCESS,
                metadata=meta,
            ),
            BatchResult(
                task_id=1,
                prompt="p",
                content="Second chunk",
                model_settings=mc,
                response="译文二",
                status=TaskStatus.SUCCESS,
                metadata=meta,
            ),
        ]
        result = translator.export_text_file(
            self._job(tmp_path),
            results,
            preserve_format=True,
            include_original=False,
            force=True,
        )
        assert result.success is True
        assert result.partial is False

    def test_session_counts_partial_as_failed(self):
        from ask_llm.services.translation_options import (
            TranslationJobResult,
            TranslationSessionResult,
        )

        service = _make_service()
        session = TranslationSessionResult()
        service._accumulate(
            session,
            TranslationJobResult(
                file_path="a.md",
                output_path="a_trans.md",
                input_tokens=10,
                output_tokens=5,
                success=False,
                partial=True,
            ),
        )
        assert session.successful_files == 0
        assert session.partial_files == 1
        assert session.failed_files == 1  # exit-code driver
        assert session.total_input_tokens == 10  # partial spend still counted
