"""Tests for @ prefix path resolution in translator."""

import tempfile
from pathlib import Path

import pytest

from ask_llm.core.translator import Translator


class TestTranslatorAtPrefix:
    """Test @ prefix path resolution."""

    def test_at_prefix_resolution(self, tmp_path):
        """Test @ prefix resolves to project root."""
        # Create a mock project structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        prompts_dir = project_root / "prompts"
        prompts_dir.mkdir()

        # Create a prompt file
        prompt_file = prompts_dir / "test-prompt.md"
        prompt_file.write_text("Test prompt: {content}")

        # Create project markers
        (project_root / "pyproject.toml").write_text("[project]")

        # Change to project root
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(project_root)

            # Test @ prefix resolution
            translator = Translator(prompt_file="@prompts/test-prompt.md")
            assert translator.custom_prompt_template == "Test prompt: {content}"

        finally:
            os.chdir(original_cwd)

    def test_at_prefix_with_leading_slash(self, tmp_path):
        """Test @ prefix with leading slash."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        prompts_dir = project_root / "prompts"
        prompts_dir.mkdir()

        prompt_file = prompts_dir / "test-prompt.md"
        prompt_file.write_text("Test prompt: {content}")

        (project_root / "pyproject.toml").write_text("[project]")

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(project_root)

            # Test @/prompts/... format
            translator = Translator(prompt_file="@/prompts/test-prompt.md")
            assert translator.custom_prompt_template == "Test prompt: {content}"

        finally:
            os.chdir(original_cwd)

    def test_at_prefix_fallback_to_current_dir(self, tmp_path):
        """Test @ prefix falls back to current directory if no project root found."""
        # Create a directory without project markers
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        prompt_file = test_dir / "prompt.md"
        prompt_file.write_text("Test prompt: {content}")

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(test_dir)

            # Should fallback to current directory
            translator = Translator(prompt_file="@prompt.md")
            assert translator.custom_prompt_template == "Test prompt: {content}"

        finally:
            os.chdir(original_cwd)

    def test_absolute_path_still_works(self):
        """Test that absolute paths still work without @ prefix."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Test prompt: {content}")
            temp_path = f.name

        try:
            translator = Translator(prompt_file=temp_path)
            assert translator.custom_prompt_template == "Test prompt: {content}"
        finally:
            Path(temp_path).unlink()

    def test_relative_path_without_at_prefix(self):
        """Test relative path without @ prefix."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Test prompt: {content}")
            temp_path = Path(f.name)

        try:
            # Use relative path from current directory
            import os

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_path.parent)
                relative_path = temp_path.name

                translator = Translator(prompt_file=relative_path)
                assert translator.custom_prompt_template == "Test prompt: {content}"
            finally:
                os.chdir(original_cwd)
        finally:
            temp_path.unlink()
