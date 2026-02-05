"""Integration tests for trans command with prompt files."""

import tempfile
from pathlib import Path

import pytest

from ask_llm.core.translator import Translator
from ask_llm.utils.trans_config_loader import TransConfig


class TestTransPromptFileIntegration:
    """Integration tests for prompt file functionality."""

    def test_trans_config_with_prompt_file(self):
        """Test TransConfig with prompt_file field."""
        config = TransConfig(prompt_file="@prompts/tech-paper-trans.md")
        assert config.prompt_file == "@prompts/tech-paper-trans.md"

    def test_translator_with_prompt_file(self):
        """Test Translator with prompt file."""
        prompt_content = """Translate from {source_lang} to {target_lang}:

{content}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(prompt_content)
            temp_path = f.name

        try:
            translator = Translator(
                target_language="zh",
                source_language="en",
                prompt_file=temp_path,
            )

            # Verify prompt is loaded
            assert translator.custom_prompt_template == prompt_content

            # Generate prompt
            prompt = translator.generate_prompt("Hello world")
            assert "Hello world" in prompt
        finally:
            Path(temp_path).unlink()

    def test_prompt_file_priority(self):
        """Test that prompt_file has highest priority."""
        prompt_file_content = "File prompt: {content}"
        custom_template = "Template prompt: {content}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(prompt_file_content)
            temp_path = f.name

        try:
            translator = Translator(
                target_language="zh",
                source_language="en",
                style="formal",
                custom_prompt_template=custom_template,
                prompt_file=temp_path,
            )

            prompt = translator.generate_prompt("Test")
            # prompt_file should override custom_template
            assert "File prompt" in prompt
            assert "Template prompt" not in prompt
        finally:
            Path(temp_path).unlink()
