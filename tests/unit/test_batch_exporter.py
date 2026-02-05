"""Unit tests for batch result exporter."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from ask_llm.core.batch import BatchResult, BatchStatistics, ModelConfig, TaskStatus
from ask_llm.core.models import RequestMetadata
from ask_llm.utils.batch_exporter import BatchResultExporter


class TestBatchResultExporter:
    """Test BatchResultExporter."""

    @pytest.fixture
    def sample_results(self):
        """Create sample batch results."""
        model_config = ModelConfig(provider="test", model="test-model")
        metadata = RequestMetadata(
            provider="test",
            model="test-model",
            temperature=0.7,
            input_tokens=10,
            output_tokens=20,
            latency=1.5,
        )

        return [
            BatchResult(
                task_id=1,
                prompt="Test prompt",
                content="Test content",
                model_settings=model_config,
                response="Test response",
                metadata=metadata,
                status=TaskStatus.SUCCESS,
            ),
            BatchResult(
                task_id=2,
                prompt="Test prompt 2",
                content="Test content 2",
                model_settings=model_config,
                response=None,
                metadata=None,
                status=TaskStatus.FAILED,
                error="Test error",
            ),
        ]

    @pytest.fixture
    def sample_statistics(self):
        """Create sample statistics."""
        return BatchStatistics(
            total_tasks=2,
            successful_tasks=1,
            failed_tasks=1,
            total_latency=1.5,
            average_latency=1.5,
            total_input_tokens=10,
            total_output_tokens=20,
        )

    def test_export_json(self, temp_dir, sample_results, sample_statistics):
        """Test exporting to JSON format."""
        exporter = BatchResultExporter(sample_results, sample_statistics)
        output_file = temp_dir / "results.json"

        exporter.export(str(output_file), "json")

        assert output_file.exists()
        data = json.loads(output_file.read_text())

        assert "statistics" in data
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["statistics"]["total_tasks"] == 2

    def test_export_yaml(self, temp_dir, sample_results, sample_statistics):
        """Test exporting to YAML format."""
        exporter = BatchResultExporter(sample_results, sample_statistics)
        output_file = temp_dir / "results.yaml"

        exporter.export(str(output_file), "yaml")

        assert output_file.exists()
        data = yaml.safe_load(output_file.read_text())

        assert "statistics" in data
        assert "results" in data

    def test_export_csv(self, temp_dir, sample_results, sample_statistics):
        """Test exporting to CSV format."""
        exporter = BatchResultExporter(sample_results, sample_statistics)
        output_file = temp_dir / "results.csv"

        exporter.export(str(output_file), "csv")

        assert output_file.exists()
        content = output_file.read_text()
        assert "Task ID" in content
        assert "Status" in content
        assert "1" in content  # Task ID

    def test_export_markdown(self, temp_dir, sample_results, sample_statistics):
        """Test exporting to Markdown format."""
        exporter = BatchResultExporter(sample_results, sample_statistics)
        output_file = temp_dir / "results.md"

        exporter.export(str(output_file), "markdown")

        assert output_file.exists()
        content = output_file.read_text()
        assert "# Batch Processing Results" in content
        assert "## Statistics" in content

    def test_export_unsupported_format(self, temp_dir, sample_results, sample_statistics):
        """Test exporting with unsupported format."""
        exporter = BatchResultExporter(sample_results, sample_statistics)
        output_file = temp_dir / "results.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export(str(output_file), "txt")

    def test_export_multiple_models(self, temp_dir):
        """Test exporting results grouped by model."""
        model_config1 = ModelConfig(provider="test1", model="model1")
        model_config2 = ModelConfig(provider="test2", model="model2")

        results_by_model = {
            "test1/model1": [
                BatchResult(
                    task_id=1,
                    prompt="Prompt 1",
                    content="Content 1",
                    model_settings=model_config1,
                    response="Response 1",
                    status=TaskStatus.SUCCESS,
                )
            ],
            "test2/model2": [
                BatchResult(
                    task_id=2,
                    prompt="Prompt 2",
                    content="Content 2",
                    model_settings=model_config2,
                    response="Response 2",
                    status=TaskStatus.SUCCESS,
                )
            ],
        }

        statistics_by_model = {
            "test1/model1": BatchStatistics(total_tasks=1, successful_tasks=1),
            "test2/model2": BatchStatistics(total_tasks=1, successful_tasks=1),
        }

        exported_files = BatchResultExporter.export_multiple_models(
            results_by_model, statistics_by_model, str(temp_dir), "json"
        )

        assert len(exported_files) == 2
        assert all(Path(f).exists() for f in exported_files)
