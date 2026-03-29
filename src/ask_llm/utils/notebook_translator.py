"""Jupyter Notebook translation - translate markdown cells only, preserve code cells."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import nbformat
from loguru import logger
from nbformat import NotebookNode

from ask_llm.core.batch import BatchTask, GlobalBatchProcessor, ModelConfig
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.text_splitter import TextChunk
from ask_llm.core.translator import Translator
from ask_llm.utils.chunk_balance import rebalance_translation_chunks


def _split_markdown_cell_tokens(text: str, model: str, max_chunk_tokens: int) -> List[str]:
    """Split long markdown cell text by token budget (structure-aware)."""
    if not text.strip():
        return []
    splitter = MarkdownTokenSplitter(model, max_chunk_tokens)
    return [c.content for c in splitter.split(text)]


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
    ):
        self.translator = translator
        self.model_config = model_config

    def translate_notebook(
        self,
        input_path: str,
        output_path: str,
        config_manager: Any,
        max_workers: int = 5,
        max_retries: int = 3,
        show_progress: bool = True,
        *,
        balance_chunks: bool = True,
        max_chunk_tokens: int = 2400,
        min_chunk_merge_tokens: int = 400,
    ) -> Tuple[int, int, int, int]:
        """
        Translate a Jupyter notebook.

        Args:
            input_path: Path to input notebook
            output_path: Path to output translated notebook
            config_manager: ConfigManager instance for provider/model
            max_workers: Number of concurrent workers
            max_retries: Maximum retry attempts
            show_progress: Whether to show progress
            balance_chunks: Rebalance markdown sub-chunks by token estimate (per cell)
            max_chunk_tokens: Token cap for splitting and rebalance
            min_chunk_merge_tokens: Unused (API compat); rebalance merges greedily up to max_chunk_tokens

        Returns:
            Tuple of (successful_count, failed_count, total_input_tokens, total_output_tokens)
            Token counts aggregate metadata from successful tasks only (same as batch statistics).
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
        tasks_data: List[Tuple[int, str]] = []
        model = self.model_config.model
        for i, cell in enumerate(notebook.cells):
            if not _is_markdown_cell(cell):
                continue
            original_text = cell.source
            if isinstance(original_text, list):
                original_text = "".join(original_text)
            if not original_text.strip():
                continue

            raw_chunks = _split_markdown_cell_tokens(original_text, model, max_chunk_tokens)
            tmp_chunks = [
                TextChunk(content=s, chunk_id=j, start_pos=0, end_pos=len(s), metadata={})
                for j, s in enumerate(raw_chunks)
            ]
            balanced = rebalance_translation_chunks(
                tmp_chunks,
                model,
                max_chunk_tokens=max_chunk_tokens,
                min_merge_tokens=min_chunk_merge_tokens,
                enabled=balance_chunks,
            )
            for part in balanced:
                tasks_data.append((i, part.content))

        if not tasks_data:
            logger.info("No markdown cells to translate")
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
            return 0, 0, 0, 0

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

        successful = sum(1 for r in results if r.status.value == "success")
        failed = len(results) - successful
        if successful == 0 and failed > 0 and getattr(processor, "_auth_error_logged", False):
            raise RuntimeError("API authentication failed; no translated output.")

        # Build cell_index -> list of translated chunks (in order)
        result_map = {r.task_id: r for r in results}
        cell_translations: Dict[int, List[str]] = {}
        for task_id, (cell_idx, _) in enumerate(tasks_data):
            result = result_map.get(task_id)
            if result and result.response and result.status.value == "success":
                translated = result.response.strip()
            else:
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

        ok = [r for r in results if r.status.value == "success" and r.metadata]
        total_in = sum(r.metadata.input_tokens for r in ok if r.metadata)
        total_out = sum(r.metadata.output_tokens for r in ok if r.metadata)
        logger.info(f"Translated notebook saved to: {output_path}")
        logger.info(f"Statistics: {successful} chunks translated, {failed} failed")

        return successful, failed, total_in, total_out
