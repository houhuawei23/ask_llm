"""Tests for translator with LaTeX formulas in prompt templates."""

import tempfile
from pathlib import Path

import pytest

from ask_llm.core.translator import Translator


class TestTranslatorLaTeX:
    """Test Translator with LaTeX formulas in templates."""

    def test_prompt_with_latex_formula(self):
        """Test prompt generation with LaTeX formula in template."""
        # Template with LaTeX formula containing { } braces
        template = (
            "Translate from {source_lang} to {target_lang}.\n\n"
            "Example formula: $L(N, D) = \\alpha N^{\\beta} + \\gamma D^{\\delta}$\n\n"
            "Content:\n{content}"
        )

        translator = Translator(
            target_language="zh",
            source_language="en",
            custom_prompt_template=template,
        )

        # Should not raise KeyError
        prompt = translator.generate_prompt("Hello world")
        assert "Hello world" in prompt
        assert "\\beta" in prompt or "β" in prompt
        assert "中文" in prompt or "英文" in prompt

    def test_prompt_file_with_latex_formula(self):
        """Test loading prompt file with LaTeX formulas."""
        prompt_content = (
            "Translate from {source_lang} to {target_lang}.\n\n"
            "Example:\n"
            "- Inline: $x^{\\alpha} + y^{\\beta}$\n"
            "- Block: $$\\int_{0}^{\\infty} e^{-x} dx = 1$$\n\n"
            "{content}"
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

            # Should not raise KeyError
            prompt = translator.generate_prompt("Test content")
            assert "Test content" in prompt
            assert "\\alpha" in prompt or "α" in prompt
            assert "\\beta" in prompt or "β" in prompt
        finally:
            Path(temp_path).unlink()

    def test_prompt_with_nested_braces(self):
        """Test prompt with nested braces (like LaTeX subscripts)."""
        template = (
            "Translate: {content}\n\n"
            "Note: Use format like $a_{i,j}$ for subscripts."
        )

        translator = Translator(
            target_language="zh",
            source_language="en",
            custom_prompt_template=template,
        )

        # Should not raise KeyError
        prompt = translator.generate_prompt("Test")
        assert "Test" in prompt
        assert "a_{i,j}" in prompt or "a_{i" in prompt

    def test_prompt_with_multiple_placeholders_and_latex(self):
        """Test prompt with both placeholders and LaTeX formulas."""
        template = (
            "From {source_lang} to {target_lang}:\n\n"
            "Formula: $E = mc^{2}$\n\n"
            "Content: {content}"
        )

        translator = Translator(
            target_language="zh",
            source_language="en",
            custom_prompt_template=template,
        )

        prompt = translator.generate_prompt("Energy equation")
        assert "Energy equation" in prompt
        assert "中文" in prompt or "英文" in prompt
        assert "mc^{2}" in prompt or "mc²" in prompt
