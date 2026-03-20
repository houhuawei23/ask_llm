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

    def test_export_split_files(self, temp_dir, sample_results, sample_statistics):
        """Test exporting split files (one file per task)."""
        # Modify sample results to have responses
        sample_results[0].response = "Response 1"
        sample_results[1].response = "Response 2"

        exported_files = BatchResultExporter.export_split_files(
            sample_results, str(temp_dir)
        )

        assert len(exported_files) == 2
        assert all(Path(f).exists() for f in exported_files)

        # Check file contents
        file1 = Path(exported_files[0])
        file2 = Path(exported_files[1])
        assert file1.read_text() in ["Response 1", "Response 2"]
        assert file2.read_text() in ["Response 1", "Response 2"]
        assert file1.read_text() != file2.read_text()

    def test_export_split_files_with_output_filename(self, temp_dir):
        """Test split export with configured output filenames."""
        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=1,
                prompt="Prompt 1",
                content="Content 1",
                output_filename="result1.md",
                model_settings=model_config,
                response="Response 1",
                status=TaskStatus.SUCCESS,
            ),
            BatchResult(
                task_id=2,
                prompt="Prompt 2",
                content="Content 2",
                output_filename="result2.md",
                model_settings=model_config,
                response="Response 2",
                status=TaskStatus.SUCCESS,
            ),
        ]

        exported_files = BatchResultExporter.export_split_files(results, str(temp_dir))

        assert len(exported_files) == 2
        exported_filenames = [Path(f).name for f in exported_files]
        assert "result1.md" in exported_filenames
        assert "result2.md" in exported_filenames

        # Check file contents
        for file_path in exported_files:
            path = Path(file_path)
            if path.name == "result1.md":
                assert path.read_text() == "Response 1"
            elif path.name == "result2.md":
                assert path.read_text() == "Response 2"

    def test_export_split_files_without_output_filename(self, temp_dir):
        """Test split export with default filename generation."""
        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=1,
                prompt="Prompt 1",
                content="Content 1",
                output_filename=None,
                model_settings=model_config,
                response="Response 1",
                status=TaskStatus.SUCCESS,
            ),
            BatchResult(
                task_id=2,
                prompt="Prompt 2",
                content="Content 2",
                output_filename=None,
                model_settings=model_config,
                response="Response 2",
                status=TaskStatus.SUCCESS,
            ),
        ]

        exported_files = BatchResultExporter.export_split_files(results, str(temp_dir))

        assert len(exported_files) == 2
        exported_filenames = [Path(f).name for f in exported_files]
        assert "task_1.md" in exported_filenames
        assert "task_2.md" in exported_filenames

    def test_export_split_files_empty_response(self, temp_dir):
        """Test split export with empty response (failed task)."""
        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=1,
                prompt="Prompt 1",
                content="Content 1",
                output_filename="result1.md",
                model_settings=model_config,
                response=None,
                status=TaskStatus.FAILED,
                error="Test error",
            ),
        ]

        exported_files = BatchResultExporter.export_split_files(results, str(temp_dir))

        assert len(exported_files) == 1
        file_path = Path(exported_files[0])
        assert file_path.exists()
        assert file_path.read_text() == ""  # Empty file for failed task

    def test_export_split_files_duplicate_filename(self, temp_dir):
        """Test handling of duplicate output filenames."""
        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=1,
                prompt="Prompt 1",
                content="Content 1",
                output_filename="result.md",
                model_settings=model_config,
                response="Response 1",
                status=TaskStatus.SUCCESS,
            ),
            BatchResult(
                task_id=2,
                prompt="Prompt 2",
                content="Content 2",
                output_filename="result.md",
                model_settings=model_config,
                response="Response 2",
                status=TaskStatus.SUCCESS,
            ),
        ]

        exported_files = BatchResultExporter.export_split_files(results, str(temp_dir))

        assert len(exported_files) == 2
        exported_filenames = [Path(f).name for f in exported_files]
        assert "result.md" in exported_filenames
        assert "result_1.md" in exported_filenames  # Second file should have suffix

    def test_export_split_files_filename_sanitization(self, temp_dir):
        """Test filename sanitization (removing dangerous characters)."""
        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=1,
                prompt="Prompt 1",
                content="Content 1",
                output_filename="../../result.md",  # Path traversal attempt
                model_settings=model_config,
                response="Response 1",
                status=TaskStatus.SUCCESS,
            ),
        ]

        exported_files = BatchResultExporter.export_split_files(results, str(temp_dir))

        assert len(exported_files) == 1
        file_path = Path(exported_files[0])
        # Should sanitize path separators
        assert ".." not in file_path.name
        assert file_path.parent == temp_dir  # Should be in output dir, not parent
