"""Jupyter Notebook translation - translate markdown cells only, preserve code cells."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import nbformat
from loguru import logger
from nbformat import NotebookNode

from ask_llm.core.batch import BatchTask, GlobalBatchProcessor, ModelConfig
from ask_llm.core.text_splitter import MarkdownSplitter
from ask_llm.core.translator import Translator


def _split_markdown_cell(text: str, max_chunk_size: int) -> List[str]:
    """
    Split long markdown text into chunks for translation.

    Args:
        text: Markdown text to split
        max_chunk_size: Maximum chunk size in characters

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    if len(text) <= max_chunk_size:
        return [text]

    splitter = MarkdownSplitter(max_chunk_size=max_chunk_size)
    chunks = splitter.split(text)
    return [c.content for c in chunks]


def _is_markdown_cell(cell: NotebookNode) -> bool:
    """Check if a cell is a markdown cell."""
    return cell.cell_type == "markdown"


class NotebookTranslator:
    """
    Translate Jupyter notebook markdown cells using LLM API.

    Only translates markdown cells; code cells are preserved unchanged.
    """

    def __init__(
        self,
        translator: Translator,
        model_config: ModelConfig,
        max_chunk_size: int = 2000,
    ):
        """
        Initialize notebook translator.

        Args:
            translator: Translator instance for generating prompts
            model_config: Model configuration for API calls
            max_chunk_size: Maximum chunk size for splitting long markdown cells
        """
        self.translator = translator
        self.model_config = model_config
        self.max_chunk_size = max_chunk_size

    def translate_notebook(
        self,
        input_path: str,
        output_path: str,
        config_manager: Any,
        max_workers: int = 5,
        max_retries: int = 3,
        show_progress: bool = True,
    ) -> Tuple[int, int]:
        """
        Translate a Jupyter notebook.

        Args:
            input_path: Path to input notebook
            output_path: Path to output translated notebook
            config_manager: ConfigManager instance for provider/model
            max_workers: Number of concurrent workers
            max_retries: Maximum retry attempts
            show_progress: Whether to show progress

        Returns:
            Tuple of (successful_count, failed_count)
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input notebook not found: {input_path}")
        if input_file.suffix != ".ipynb":
            raise ValueError(f"Input file must be a Jupyter notebook (.ipynb): {input_path}")

        logger.info(f"Reading notebook: {input_path}")
        with open(input_path, encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        # Build translation tasks: (cell_index, chunk_content) for markdown cells
        tasks_data: List[Tuple[int, str]] = []  # (cell_index, chunk_content)
        for i, cell in enumerate(notebook.cells):
            if not _is_markdown_cell(cell):
                continue
            original_text = cell.source
            if isinstance(original_text, list):
                original_text = "".join(original_text)
            if not original_text.strip():
                continue

            chunks = _split_markdown_cell(original_text, self.max_chunk_size)
            for chunk in chunks:
                tasks_data.append((i, chunk))

        if not tasks_data:
            logger.info("No markdown cells to translate")
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
            return 0, 0

        # Create BatchTasks
        tasks: List[BatchTask] = []
        for task_id, (_, chunk_content) in enumerate(tasks_data):
            prompt = self.translator.generate_prompt(chunk_content)
            tasks.append(
                BatchTask(
                    task_id=task_id,
                    prompt=prompt,
                    content=chunk_content,
                    task_model_config=self.model_config,
                )
            )

        # Process with GlobalBatchProcessor
        processor = GlobalBatchProcessor(
            max_workers=max_workers,
            max_retries=max_retries,
        )
        results = processor.process_global_tasks(tasks, config_manager, show_progress=show_progress)

        # Build cell_index -> list of translated chunks (in order)
        # tasks_data[task_id] = (cell_index, chunk_content)
        result_map = {r.task_id: r for r in results}
        cell_translations: Dict[int, List[str]] = {}
        for task_id, (cell_idx, _) in enumerate(tasks_data):
            result = result_map.get(task_id)
            if result and result.response and result.status.value == "success":
                translated = result.response.strip()
            else:
                # On failure, keep original
                translated = tasks_data[task_id][1]
                logger.warning(f"Translation failed for cell {cell_idx} chunk, keeping original")

            if cell_idx not in cell_translations:
                cell_translations[cell_idx] = []
            cell_translations[cell_idx].append(translated)

        # Merge chunks per cell and update notebook
        translated_cells = list(notebook.cells)
        for cell_idx, translated_chunks in cell_translations.items():
            merged_text = "\n\n".join(translated_chunks)
            cell = notebook.cells[cell_idx]
            translated_cells[cell_idx] = NotebookNode(
                {
                    "cell_type": cell.cell_type,
                    "metadata": cell.metadata.copy(),
                    "source": merged_text,
                }
            )

        # Create output notebook
        translated_notebook = NotebookNode(
            {
                "cells": translated_cells,
                "metadata": notebook.metadata.copy(),
                "nbformat": notebook.nbformat,
                "nbformat_minor": notebook.nbformat_minor,
            }
        )

        # Write output
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(translated_notebook, f)

        successful = sum(1 for r in results if r.status.value == "success")
        failed = len(results) - successful
        logger.info(f"Translated notebook saved to: {output_path}")
        logger.info(f"Statistics: {successful} chunks translated, {failed} failed")

        return successful, failed
