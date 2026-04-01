"""Tests for unified configuration."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ask_llm.config.unified_config import (
    PaperConfig,
    TranslationConfig,
    UnifiedConfig,
)


class TestTranslationConfig:
    """Test TranslationConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TranslationConfig()
        assert config.target_language == "zh"
        assert config.source_language == "auto"
        assert config.style == "formal"
        assert config.threads == 20
        assert config.max_parallel_files == 3
        assert config.retries == 3
        assert config.max_chunk_tokens == 2400
        assert config.max_output_tokens == 8192

    def test_custom_config(self):
        """Test custom configuration."""
        config = TranslationConfig(
            target_language="en",
            source_language="zh",
            style="casual",
            threads=10,
            retries=5,
            max_chunk_tokens=3200,
        )
        assert config.target_language == "en"
        assert config.source_language == "zh"
        assert config.style == "casual"
        assert config.threads == 10
        assert config.retries == 5
        assert config.max_chunk_tokens == 3200


class TestUnifiedConfig:
    """Test UnifiedConfig.from_dict."""

    def test_from_dict_empty(self):
        """Test from_dict with empty data."""
        config = UnifiedConfig.from_dict({})
        assert config.translation.target_language == "zh"
        assert config.general.default_output_filename == "output.txt"
        assert config.batch.threads == 5
        assert config.paper.default_run == "all"
        assert config.paper.output_subdir == "explain"
        assert config.paper.max_output_tokens == 65536
        assert config.paper.full_model == "deepseek-reasoner"


class TestPaperConfig:
    """Paper explanation defaults."""

    def test_paper_defaults(self):
        p = PaperConfig()
        assert p.default_run == "all"
        assert p.output_subdir == "explain"
        assert "prompts/paper" in p.prompt_dir
        assert p.concurrency == 5
        assert p.max_output_tokens == 65536
        assert p.full_model == "deepseek-reasoner"

    def test_from_dict_with_translation(self):
        """Test from_dict with translation section."""
        data = {
            "translation": {
                "target_language": "en",
                "source_language": "zh",
                "style": "casual",
                "threads": 10,
                "retries": 5,
                "max_chunk_size": 3000,
            }
        }
        config = UnifiedConfig.from_dict(data)
        assert config.translation.target_language == "en"
        assert config.translation.source_language == "zh"
        assert config.translation.style == "casual"
        assert config.translation.threads == 10
        assert config.translation.retries == 5
        assert config.translation.max_chunk_tokens == 2400

    def test_from_dict_with_project_root_markers(self):
        """Test from_dict with project root markers."""
        data = {
            "project_root_markers": ["custom.txt", "my.config"],
        }
        config = UnifiedConfig.from_dict(data)
        assert config.project_root_markers == ["custom.txt", "my.config"]
