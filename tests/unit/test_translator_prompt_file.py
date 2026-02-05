"""Tests for translator prompt file loading."""

import tempfile
from pathlib import Path

import pytest

from ask_llm.core.translator import Translator


class TestTranslatorPromptFile:
    """Test Translator prompt file loading."""

    def test_load_prompt_from_file(self):
        """Test loading prompt from file."""
        prompt_content = "Translate from {source_lang} to {target_lang}:\n\n{content}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(prompt_content)
            temp_path = f.name

        try:
            translator = Translator(
                target_language="zh",
                source_language="en",
                prompt_file=temp_path,
            )

            prompt = translator.generate_prompt("Hello world")
            assert "Hello world" in prompt
            assert "中文" in prompt or "英文" in prompt
        finally:
            Path(temp_path).unlink()

    def test_load_prompt_from_file_not_found(self):
        """Test loading prompt from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            Translator(
                target_language="zh",
                source_language="en",
                prompt_file="/nonexistent/prompt.md",
            )

    def test_prompt_file_overrides_style(self):
        """Test that prompt file overrides style."""
        prompt_content = "Custom prompt: {content}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(prompt_content)
            temp_path = f.name

        try:
            translator = Translator(
                target_language="zh",
                source_language="en",
                style="formal",
                prompt_file=temp_path,
            )

            prompt = translator.generate_prompt("Test")
            assert "Custom prompt" in prompt
            assert "Test" in prompt
        finally:
            Path(temp_path).unlink()

    def test_prompt_file_overrides_custom_template(self):
        """Test that prompt file overrides custom template."""
        prompt_content = "File prompt: {content}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(prompt_content)
            temp_path = f.name

        try:
            translator = Translator(
                target_language="zh",
                source_language="en",
                custom_prompt_template="Template prompt: {content}",
                prompt_file=temp_path,
            )

            prompt = translator.generate_prompt("Test")
            assert "File prompt" in prompt
            assert "Template prompt" not in prompt
        finally:
            Path(temp_path).unlink()

    def test_prompt_file_with_placeholders(self):
        """Test prompt file with all placeholders."""
        prompt_content = (
            "Translate from {source_lang} to {target_lang}.\n\n"
            "Content:\n{content}"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(prompt_content)
            temp_path = f.name

        try:
            translator = Translator(
                target_language="zh",
                source_language="en",
                prompt_file=temp_path,
            )

            prompt = translator.generate_prompt("Hello")
            assert "中文" in prompt or "英文" in prompt
            assert "Hello" in prompt
        finally:
            Path(temp_path).unlink()
