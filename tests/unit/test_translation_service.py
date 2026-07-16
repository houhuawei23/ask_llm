"""Unit tests for TranslationService fallback wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.batch import BatchTask, ModelConfig
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
        min_chunk_merge_tokens=400,
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

        job = service._prepare_text_file(
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

        job = service._prepare_text_file(
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
