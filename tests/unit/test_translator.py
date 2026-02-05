"""Tests for translator module."""

import pytest

from ask_llm.core.batch import ModelConfig
from ask_llm.core.text_splitter import TextChunk
from ask_llm.core.translator import Translator, TranslationStyle


class TestTranslator:
    """Test Translator class."""

    def test_generate_prompt_formal(self):
        """Test generating formal translation prompt."""
        translator = Translator(
            target_language="zh",
            source_language="en",
            style=TranslationStyle.FORMAL,
        )
        prompt = translator.generate_prompt("Hello world")

        assert "翻译" in prompt or "translate" in prompt.lower()
        assert "Hello world" in prompt
        assert "正式" in prompt or "formal" in prompt.lower()

    def test_generate_prompt_casual(self):
        """Test generating casual translation prompt."""
        translator = Translator(
            target_language="zh",
            source_language="en",
            style=TranslationStyle.CASUAL,
        )
        prompt = translator.generate_prompt("Hello world")

        assert "Hello world" in prompt
        assert "自然" in prompt or "casual" in prompt.lower() or "口语" in prompt

    def test_generate_prompt_technical(self):
        """Test generating technical translation prompt."""
        translator = Translator(
            target_language="zh",
            source_language="en",
            style=TranslationStyle.TECHNICAL,
        )
        prompt = translator.generate_prompt("Hello world")

        assert "Hello world" in prompt
        assert "技术" in prompt or "technical" in prompt.lower() or "专业术语" in prompt

    def test_generate_prompt_custom_template(self):
        """Test generating prompt with custom template."""
        custom_template = "Translate this: {content}"
        translator = Translator(
            target_language="zh",
            source_language="en",
            custom_prompt_template=custom_template,
        )
        prompt = translator.generate_prompt("Hello world")

        assert prompt == "Translate this: Hello world"

    def test_generate_prompt_auto_source_language(self):
        """Test generating prompt with auto source language."""
        translator = Translator(
            target_language="zh",
            source_language="auto",
        )
        prompt = translator.generate_prompt("Hello world")

        assert "Hello world" in prompt

    def test_create_translation_tasks(self):
        """Test creating translation tasks from chunks."""
        translator = Translator(target_language="zh", source_language="en")
        chunks = [
            TextChunk(content="First chunk", chunk_id=0),
            TextChunk(content="Second chunk", chunk_id=1),
        ]
        model_config = ModelConfig(provider="test", model="test-model")

        tasks = translator.create_translation_tasks(chunks, model_config)

        assert len(tasks) == 2
        assert tasks[0].task_id == 0
        assert tasks[1].task_id == 1
        assert tasks[0].content == "First chunk"
        assert tasks[1].content == "Second chunk"
        assert tasks[0].task_model_config == model_config

    def test_format_language_name(self):
        """Test language name formatting."""
        assert Translator._format_language_name("zh") == "中文"
        assert Translator._format_language_name("en") == "英文"
        assert Translator._format_language_name("auto") == "原文"
        assert Translator._format_language_name("unknown") == "unknown"
