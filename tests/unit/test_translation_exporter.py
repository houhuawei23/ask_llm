"""Tests for translation exporter."""

import json
import tempfile
from pathlib import Path

import pytest

from ask_llm.core.batch import BatchResult, ModelConfig, TaskStatus
from ask_llm.core.models import RequestMetadata
from ask_llm.core.text_splitter import TextChunk
from ask_llm.utils.translation_exporter import TranslationExporter


class TestTranslationExporter:
    """Test TranslationExporter."""

    def create_test_data(self):
        """Create test chunks and results."""
        chunks = [
            TextChunk(content="First chunk", chunk_id=0, metadata={"type": "paragraph"}),
            TextChunk(content="Second chunk", chunk_id=1, metadata={"type": "paragraph"}),
        ]

        model_config = ModelConfig(provider="test", model="test-model")
        metadata = RequestMetadata(
            provider="test",
            model="test-model",
            temperature=0.7,
            input_tokens=10,
            output_tokens=15,
            latency=0.5,
        )

        results = [
            BatchResult(
                task_id=0,
                prompt="Translate: First chunk",
                content="First chunk",
                model_settings=model_config,
                response="第一个块",
                status=TaskStatus.SUCCESS,
                metadata=metadata,
            ),
            BatchResult(
                task_id=1,
                prompt="Translate: Second chunk",
                content="Second chunk",
                model_settings=model_config,
                response="第二个块",
                status=TaskStatus.SUCCESS,
                metadata=metadata,
            ),
        ]

        return chunks, results

    def test_export_text(self):
        """Test exporting as plain text."""
        chunks, results = self.create_test_data()
        exporter = TranslationExporter(chunks, results)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            exported_path = exporter.export(temp_path, format_type="text")
            assert Path(exported_path).exists()

            content = Path(exported_path).read_text(encoding="utf-8")
            assert "第一个块" in content
            assert "第二个块" in content
        finally:
            Path(temp_path).unlink()

    def test_export_markdown(self):
        """Test exporting as Markdown."""
        chunks = [
            TextChunk(
                content="Content under heading",
                chunk_id=0,
                metadata={"heading_level": 1, "heading_title": "Main Title"},
            )
        ]

        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=0,
                prompt="Translate",
                content="Content under heading",
                model_settings=model_config,
                response="标题下的内容",
                status=TaskStatus.SUCCESS,
            )
        ]

        exporter = TranslationExporter(chunks, results)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = f.name

        try:
            exported_path = exporter.export(temp_path, format_type="markdown")
            assert Path(exported_path).exists()

            content = Path(exported_path).read_text(encoding="utf-8")
            assert "标题下的内容" in content
        finally:
            Path(temp_path).unlink()

    def test_export_json(self):
        """Test exporting as JSON."""
        chunks, results = self.create_test_data()
        exporter = TranslationExporter(chunks, results)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            exported_path = exporter.export(temp_path, format_type="json")
            assert Path(exported_path).exists()

            content = Path(exported_path).read_text(encoding="utf-8")
            data = json.loads(content)

            assert "chunks" in data
            assert "statistics" in data
            assert len(data["chunks"]) == 2
            assert data["statistics"]["total_chunks"] == 2
            assert data["statistics"]["successful_translations"] == 2
        finally:
            Path(temp_path).unlink()

    def test_export_auto_detect_format(self):
        """Test auto-detecting format from file extension."""
        chunks, results = self.create_test_data()
        exporter = TranslationExporter(chunks, results)

        # Test JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            exported_path = exporter.export(json_path)
            content = Path(exported_path).read_text(encoding="utf-8")
            # Should be valid JSON
            json.loads(content)
        finally:
            Path(json_path).unlink()

        # Test Markdown
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            md_path = f.name

        try:
            exported_path = exporter.export(md_path)
            content = Path(exported_path).read_text(encoding="utf-8")
            assert "第一个块" in content
        finally:
            Path(md_path).unlink()

    def test_export_with_failed_translations(self):
        """Test exporting when some translations failed."""
        chunks = [
            TextChunk(content="Success chunk", chunk_id=0),
            TextChunk(content="Failed chunk", chunk_id=1),
        ]

        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=0,
                prompt="Translate",
                content="Success chunk",
                model_settings=model_config,
                response="成功块",
                status=TaskStatus.SUCCESS,
            ),
            BatchResult(
                task_id=1,
                prompt="Translate",
                content="Failed chunk",
                model_settings=model_config,
                response=None,
                status=TaskStatus.FAILED,
                error="Translation failed",
            ),
        ]

        exporter = TranslationExporter(chunks, results)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            exported_path = exporter.export(temp_path, format_type="text")
            content = Path(exported_path).read_text(encoding="utf-8")

            # Should include successful translation
            assert "成功块" in content
            # Should include original text for failed translation
            assert "Failed chunk" in content
        finally:
            Path(temp_path).unlink()

    def test_export_preserve_order(self):
        """Test that chunks are exported in correct order."""
        chunks = [
            TextChunk(content="First", chunk_id=0),
            TextChunk(content="Second", chunk_id=1),
            TextChunk(content="Third", chunk_id=2),
        ]

        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=2,
                prompt="Translate",
                content="Third",
                model_settings=model_config,
                response="第三",
                status=TaskStatus.SUCCESS,
            ),
            BatchResult(
                task_id=0,
                prompt="Translate",
                content="First",
                model_settings=model_config,
                response="第一",
                status=TaskStatus.SUCCESS,
            ),
            BatchResult(
                task_id=1,
                prompt="Translate",
                content="Second",
                model_settings=model_config,
                response="第二",
                status=TaskStatus.SUCCESS,
            ),
        ]

        exporter = TranslationExporter(chunks, results)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            exported_path = exporter.export(temp_path, format_type="text")
            content = Path(exported_path).read_text(encoding="utf-8")

            # Should be in correct order
            first_pos = content.find("第一")
            second_pos = content.find("第二")
            third_pos = content.find("第三")

            assert first_pos < second_pos < third_pos
        finally:
            Path(temp_path).unlink()
