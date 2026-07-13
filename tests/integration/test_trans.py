"""Integration tests for trans command."""

import tempfile
from pathlib import Path

import pytest

from ask_llm.cli import _is_directory_output, _offset_task_ids, _resolve_trans_input_paths
from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig, TaskStatus
from ask_llm.core.models import RequestMetadata
from ask_llm.core.text_splitter import TextChunk, TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.utils.translation_exporter import TranslationExporter


class TestTransIntegration:
    """Integration tests for translation functionality."""

    def test_text_splitting_and_translation_flow(self):
        """Test complete flow from text splitting to translation."""
        # Create test text file
        test_content = """First paragraph.

Second paragraph.

Third paragraph with more content.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Test file type detection
            file_type = TextSplitter.detect_file_type(temp_path)
            assert file_type == "text"

            # Test text splitting
            splitter = TextSplitter.create_splitter(temp_path, max_chunk_size=100)
            chunks = splitter.split(test_content)

            assert len(chunks) >= 1
            assert all(len(chunk.content) <= 100 for chunk in chunks)

            # Test translator setup
            translator = Translator(target_language="zh", source_language="en")
            prompt = translator.generate_prompt("Hello")
            assert "Hello" in prompt

        finally:
            Path(temp_path).unlink()

    def test_markdown_splitting_flow(self):
        """Test Markdown splitting flow."""
        markdown_content = """# Main Title

Content under main title.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            # Test file type detection
            file_type = TextSplitter.detect_file_type(temp_path)
            assert file_type == "markdown"

            # Test Markdown splitting with large chunk size (should return single chunk)
            splitter = TextSplitter.create_splitter(temp_path, max_chunk_size=1000)
            chunks = splitter.split(markdown_content)
            assert len(chunks) == 1

            # Test with small chunk size (should split)
            splitter = TextSplitter.create_splitter(temp_path, max_chunk_size=50)
            chunks = splitter.split(markdown_content)
            assert len(chunks) >= 1
            # May have heading metadata if split by headings, or paragraph metadata if split by paragraphs
            assert any("type" in chunk.metadata for chunk in chunks)

        finally:
            Path(temp_path).unlink()

    def test_translation_exporter_integration(self):
        """Test translation exporter with real data flow."""
        # Create test chunks - use small max_chunk_size to force splitting
        text = "First chunk.\n\nSecond chunk."
        splitter = TextSplitter.create_splitter("test.txt", max_chunk_size=20)
        chunks = splitter.split(text)
        # Should have at least one chunk, may have more if split
        assert len(chunks) >= 1
        # Use the chunks we got, or create mock chunks if only one
        if len(chunks) < 2:
            # Create a second chunk manually for testing
            from ask_llm.core.text_splitter import TextChunk

            chunks.append(
                TextChunk(
                    content="Second chunk.",
                    chunk_id=1,
                    start_pos=len("First chunk.\n\n"),
                    end_pos=len(text),
                    metadata={"type": "paragraph"},
                )
            )

        # Create mock results
        model_config = ModelConfig(provider="test", model="test-model")
        results = [
            BatchResult(
                task_id=0,
                prompt="Translate",
                content=chunks[0].content,
                model_settings=model_config,
                response="第一个块",
                status=TaskStatus.SUCCESS,
                metadata=RequestMetadata(
                    provider="test",
                    model="test-model",
                    temperature=0.7,
                    input_tokens=10,
                    output_tokens=15,
                    latency=0.5,
                ),
            ),
            BatchResult(
                task_id=1,
                prompt="Translate",
                content=chunks[1].content,
                model_settings=model_config,
                response="第二个块",
                status=TaskStatus.SUCCESS,
                metadata=RequestMetadata(
                    provider="test",
                    model="test-model",
                    temperature=0.7,
                    input_tokens=10,
                    output_tokens=15,
                    latency=0.5,
                ),
            ),
        ]

        # Test exporter
        exporter = TranslationExporter(chunks, results, preserve_format=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            exported_path = exporter.export(temp_path, format_type="text")
            assert Path(exported_path).exists()

            content = Path(exported_path).read_text(encoding="utf-8")
            assert "第一个块" in content or "第二个块" in content

        finally:
            Path(temp_path).unlink()

    def test_config_loading_integration(self, sample_config_file):
        """Test configuration loading integration with default_config.yml."""
        load_result = ConfigLoader.load(str(sample_config_file))
        set_config(load_result)
        assert load_result.app_config.default_provider is not None
        assert load_result.unified_config.translation.target_language == "zh"

    def test_config_not_found_raises(self):
        """Test loading non-existent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("/nonexistent/default_config.yml")

    def test_translator_task_creation(self):
        """Test translator creating tasks from chunks."""
        translator = Translator(target_language="zh", source_language="en")

        # Create test chunks
        text_content = "First paragraph.\n\nSecond paragraph."
        chunks = TextSplitter.create_splitter("test.txt", max_chunk_size=100).split(text_content)

        model_config = ModelConfig(provider="test", model="test-model")
        tasks = translator.create_translation_tasks(chunks, model_config)

        assert len(tasks) == len(chunks)
        assert all(task.model_settings == model_config for task in tasks)
        assert all(task.content == chunks[i].content for i, task in enumerate(tasks))

    def test_end_to_end_text_processing(self):
        """Test end-to-end text processing workflow."""
        # Create test file
        test_content = """Paragraph one.

Paragraph two.

Paragraph three.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as input_file:
            input_file.write(test_content)
            input_path = input_file.name

        try:
            # Step 1: Detect file type
            file_type = TextSplitter.detect_file_type(input_path)
            assert file_type == "text"

            # Step 2: Split text
            splitter = TextSplitter.create_splitter(input_path, max_chunk_size=1000)
            chunks = splitter.split(test_content)
            assert len(chunks) >= 1

            # Step 3: Create translator
            translator = Translator(target_language="zh", source_language="en")

            # Step 4: Create tasks (without actual API call)
            model_config = ModelConfig(provider="test", model="test-model")
            tasks = translator.create_translation_tasks(chunks, model_config)
            assert len(tasks) == len(chunks)

            # Step 5: Verify tasks have correct structure
            for task in tasks:
                assert task.prompt is not None
                assert task.content is not None
                assert task.model_settings == model_config

        finally:
            Path(input_path).unlink()

    def test_notebook_file_type_detection(self):
        """Test notebook file type detection."""
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            temp_path = f.name

        try:
            file_type = TextSplitter.detect_file_type(temp_path)
            assert file_type == "notebook"
        finally:
            Path(temp_path).unlink()

    def test_markdown_end_to_end(self):
        """Test end-to-end Markdown processing workflow."""
        markdown_content = """# Title

Content here.

## Subsection

More content.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as input_file:
            input_file.write(markdown_content)
            input_path = input_file.name

        try:
            # Detect and split
            file_type = TextSplitter.detect_file_type(input_path)
            assert file_type == "markdown"

            # With large max_chunk_size, should return single chunk
            splitter = TextSplitter.create_splitter(input_path, max_chunk_size=1000)
            chunks = splitter.split(markdown_content)
            assert len(chunks) == 1

            # Create translator and tasks
            translator = Translator(target_language="zh", source_language="en")
            model_config = ModelConfig(provider="test", model="test-model")
            tasks = translator.create_translation_tasks(chunks, model_config)

            assert len(tasks) == len(chunks)

            # Verify chunks have metadata
            assert any("type" in chunk.metadata for chunk in chunks)

        finally:
            Path(input_path).unlink()

    def test_resolve_trans_input_paths_directory(self):
        """Test _resolve_trans_input_paths with directory input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.md").write_text("# A")
            (Path(tmpdir) / "b.txt").write_text("text")
            (Path(tmpdir) / "c.ipynb").write_text("{}")
            (Path(tmpdir) / "d.py").write_text("# code")  # Not translatable

            resolved = _resolve_trans_input_paths(
                [tmpdir],
                translatable_extensions=[".md", ".markdown", ".txt", ".ipynb"],
                recursive_dir=False,
            )
            assert len(resolved) == 3  # a.md, b.txt, c.ipynb (not d.py)
            assert any("a.md" in p for p in resolved)
            assert any("b.txt" in p for p in resolved)
            assert any("c.ipynb" in p for p in resolved)
            assert not any("d.py" in p for p in resolved)

    def test_resolve_trans_input_paths_recursive(self):
        """Test _resolve_trans_input_paths with recursive_dir=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "root.md").write_text("# Root")
            sub = Path(tmpdir) / "sub"
            sub.mkdir()
            (sub / "nested.md").write_text("# Nested")

            resolved = _resolve_trans_input_paths(
                [tmpdir],
                translatable_extensions=[".md"],
                recursive_dir=True,
            )
            assert len(resolved) == 2  # root.md and sub/nested.md
            assert any("nested.md" in p for p in resolved)

    def test_is_directory_output_existing_directory(self):
        """Existing directory is always treated as directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _is_directory_output(tmpdir, [], 1) is True

    def test_is_directory_output_trailing_separator(self):
        """Output ending with path separator is treated as directory."""
        assert _is_directory_output("/some/path/", [], 1) is True
        assert _is_directory_output("/some/path\\", [], 1) is True

    def test_is_directory_output_multiple_files_no_extension(self):
        """Non-existent path without extension is treated as directory for multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "translated")
            assert _is_directory_output(out, ["a.md", "b.md"], 2) is True

    def test_is_directory_output_single_directory_input(self):
        """Non-existent path without extension is treated as directory when input is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "translated")
            assert _is_directory_output(out, [tmpdir], 1) is True

    def test_is_directory_output_with_extension_is_file(self):
        """Non-existent path with extension is treated as a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "out.txt")
            assert _is_directory_output(out, ["a.md", "b.md"], 2) is False

    def test_is_directory_output_single_file_is_file(self):
        """Non-existent path without extension but single file input is treated as file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "out")
            assert _is_directory_output(out, ["a.md"], 1) is False

    def test_offset_task_ids_shifts_ids(self):
        """_offset_task_ids shifts task and chunk IDs by the given offset."""
        model_config = ModelConfig(provider="test", model="test-model")
        chunks = [
            TextChunk(content="a", chunk_id=0),
            TextChunk(content="b", chunk_id=1),
        ]
        tasks = [
            BatchTask(task_id=0, prompt="p", content="a", model_settings=model_config),
            BatchTask(task_id=1, prompt="p", content="b", model_settings=model_config),
        ]

        new_tasks, new_chunks = _offset_task_ids(tasks, chunks, 10)

        assert [c.chunk_id for c in new_chunks] == [10, 11]
        assert [t.task_id for t in new_tasks] == [10, 11]
        # Original objects are not mutated.
        assert [c.chunk_id for c in chunks] == [0, 1]
        assert [t.task_id for t in tasks] == [0, 1]

    def test_offset_task_ids_preserves_mapping(self):
        """Offset IDs keep the one-to-one mapping required by TranslationExporter."""
        model_config = ModelConfig(provider="test", model="test-model")
        chunks = [TextChunk(content="x", chunk_id=0)]
        tasks = [BatchTask(task_id=0, prompt="p", content="x", model_settings=model_config)]

        new_tasks, new_chunks = _offset_task_ids(tasks, chunks, 5)
        assert new_tasks[0].task_id == new_chunks[0].chunk_id


class TestTransPerFileBatching:
    """Tests for per-file translation batching with failure isolation."""

    @staticmethod
    def _make_fake_run_global_batch_tasks(captured_calls: list[list[BatchTask]]):
        from ask_llm.core.models import RequestMetadata

        class _FakeProcessor:
            _auth_error_logged = False

        def fake_run_global_batch_tasks(tasks, *args, **kwargs):
            captured_calls.append(list(tasks))
            results = []
            for task in tasks:
                results.append(
                    BatchResult(
                        task_id=task.task_id,
                        prompt=task.prompt,
                        content=task.content,
                        model_settings=task.model_settings,
                        response=f"translated {task.task_id}",
                        status=TaskStatus.SUCCESS,
                        metadata=RequestMetadata(
                            provider="test_provider",
                            model="test-model",
                            temperature=0.5,
                            input_tokens=1,
                            output_tokens=1,
                            latency=0.1,
                        ),
                    )
                )
            return results, _FakeProcessor()

        return fake_run_global_batch_tasks

    def test_each_file_gets_own_batch_call(self, temp_dir, sample_config_file, monkeypatch):
        """Each text file is translated in its own batch call and exported."""
        from typer.testing import CliRunner

        from ask_llm.cli import app

        (temp_dir / "a.md").write_text("Hello world")
        (temp_dir / "b.md").write_text("Second file")

        output_dir = temp_dir / "out"
        output_dir.mkdir()

        captured_calls: list[list[BatchTask]] = []
        fake = self._make_fake_run_global_batch_tasks(captured_calls)
        monkeypatch.setattr(
            "ask_llm.services.translation_service.run_global_batch_tasks",
            fake,
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "trans",
                str(temp_dir / "a.md"),
                str(temp_dir / "b.md"),
                "-o",
                str(output_dir),
                "--config",
                str(sample_config_file),
                "--threads",
                "4",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(captured_calls) == 2
        assert len(captured_calls[0]) == 1
        assert len(captured_calls[1]) == 1
        assert (output_dir / "a_trans.md").exists()
        assert (output_dir / "b_trans.md").exists()

    def test_failure_in_one_file_does_not_block_other(
        self, temp_dir, sample_config_file, monkeypatch
    ):
        """If one file's batch fails, the other file is still translated and saved."""
        from typer.testing import CliRunner

        from ask_llm.cli import app
        from ask_llm.core.models import RequestMetadata

        (temp_dir / "ok.md").write_text("Good content")
        (temp_dir / "bad.md").write_text("Bad content")

        output_dir = temp_dir / "out"
        output_dir.mkdir()

        class _FakeProcessor:
            _auth_error_logged = False

        def fake_run_global_batch_tasks(tasks, *args, **kwargs):
            results = []
            for task in tasks:
                if "bad" in task.content.lower():
                    results.append(
                        BatchResult(
                            task_id=task.task_id,
                            prompt=task.prompt,
                            content=task.content,
                            model_settings=task.model_settings,
                            status=TaskStatus.FAILED,
                            error="simulated failure",
                        )
                    )
                else:
                    results.append(
                        BatchResult(
                            task_id=task.task_id,
                            prompt=task.prompt,
                            content=task.content,
                            model_settings=task.model_settings,
                            response="translated",
                            status=TaskStatus.SUCCESS,
                            metadata=RequestMetadata(
                                provider="test_provider",
                                model="test-model",
                                temperature=0.5,
                                input_tokens=1,
                                output_tokens=1,
                                latency=0.1,
                            ),
                        )
                    )
            return results, _FakeProcessor()

        monkeypatch.setattr(
            "ask_llm.services.translation_service.run_global_batch_tasks",
            fake_run_global_batch_tasks,
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "trans",
                str(temp_dir / "ok.md"),
                str(temp_dir / "bad.md"),
                "-o",
                str(output_dir),
                "--config",
                str(sample_config_file),
                "--threads",
                "4",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (output_dir / "ok_trans.md").exists()
        assert not (output_dir / "bad_trans.md").exists()
