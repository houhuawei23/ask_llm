"""Jupyter Notebook translation - translate markdown cells only, preserve code cells."""

from pathlib import Path
from typing import Any

import nbformat
from loguru import logger
from nbformat import NotebookNode

from ask_llm.core.batch_models import BatchResult, BatchTask, ModelConfig, TaskStatus
from ask_llm.core.binary_splitter import create_markdown_splitter, locate_pieces
from ask_llm.core.command_runner import compute_checkpoint_digest, run_with_checkpoint
from ask_llm.core.text_splitter import TextChunk, join_chunks_position_aware
from ask_llm.core.translator import Translator
from ask_llm.utils.chunk_balance import rebalance_translation_chunks
from ask_llm.utils.token_counter import TokenCounter


class NotebookAuthError(RuntimeError):
    """Raised when every notebook chunk failed on API authentication.

    Dedicated type (P1): callers used to string-match the RuntimeError message.
    """


def _split_markdown_cell_tokens(
    text: str, model: str, max_chunk_tokens: int, prompt_overhead: int = 0
) -> list[str]:
    """Split long markdown cell text by token budget (structure-aware)."""
    if not text.strip():
        return []
    splitter = create_markdown_splitter(
        model, max_chunk_tokens, prompt_overhead_tokens=prompt_overhead
    )
    return [c.content for c in splitter.split(text)]


def _is_markdown_cell(cell: NotebookNode) -> bool:
    """Check if a cell is a markdown cell."""
    return bool(cell.cell_type == "markdown")


def plan_notebook_translation(
    input_path: str,
    model: str,
    *,
    prompt_template: str,
    max_chunk_tokens: int = 2400,
    balance_chunks: bool = True,
    notebook: NotebookNode | None = None,
) -> list[tuple[int, str]]:
    """Build the (cell_index, chunk_content) translation plan for a notebook.

    M10/2.25: extracted from ``NotebookTranslator.translate_notebook`` so the
    ``trans --dry-run`` estimate walks the *same* chunking pipeline (same cell
    extraction, splitter, prompt-overhead reservation, rebalance) instead of
    silently skipping notebooks. ``prompt_template`` is measured here for
    chunk sizing (D2) exactly as the paid run does.

    ``notebook`` (P0): pass an already-loaded notebook to avoid a second disk
    read racing with the first (and to keep the plan consistent with the
    notebook that will actually be written).
    """
    if notebook is None:
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input notebook not found: {input_path}")
        if input_file.suffix != ".ipynb":
            raise ValueError(f"Input file must be a Jupyter notebook (.ipynb): {input_path}")
        with open(input_path, encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

    tasks_data: list[tuple[int, str, int, int]] = []
    prompt_overhead = TokenCounter.count_tokens(prompt_template, model)
    for i, cell in enumerate(notebook.cells):
        if not _is_markdown_cell(cell):
            continue
        original_text = cell.source
        if isinstance(original_text, list):
            original_text = "".join(original_text)
        if not original_text.strip():
            continue

        raw_chunks = _split_markdown_cell_tokens(
            original_text, model, max_chunk_tokens, prompt_overhead
        )
        # Real spans (locate_pieces) so the per-cell reassembly can restore the
        # original inter-chunk separators instead of forcing blank lines.
        located = locate_pieces(original_text, raw_chunks)
        tmp_chunks = [
            TextChunk(
                content=s,
                chunk_id=j,
                start_pos=located[j][0],
                end_pos=located[j][0] + located[j][1],
                metadata={},
            )
            for j, s in enumerate(raw_chunks)
        ]
        balanced = rebalance_translation_chunks(
            tmp_chunks,
            model,
            max_chunk_tokens=max_chunk_tokens,
            enabled=balance_chunks,
            prompt_overhead=prompt_overhead,
        )
        for part in balanced:
            tasks_data.append((i, part.content, part.start_pos, part.end_pos))
    return tasks_data


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
        self.last_results: list[BatchResult] = []

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
        stream_api: bool = True,
        resume: bool = False,
    ) -> tuple[int, int, int, int]:
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
            resume: Continue from an existing checkpoint (completed chunks kept)

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

        # Build translation tasks via the shared planner (M10/2.25) — the same
        # pipeline the dry-run estimator walks. The batch prompt template is
        # measured once here for chunk sizing (D2) and reused for task
        # construction below.
        model = self.model_config.model
        prompt_template = self.translator.prompt_template_for_batch()
        tasks_data = plan_notebook_translation(
            input_path,
            model,
            prompt_template=prompt_template,
            max_chunk_tokens=max_chunk_tokens,
            balance_chunks=balance_chunks,
            notebook=notebook,
        )

        if not tasks_data:
            logger.info("No markdown cells to translate")
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
            return 0, 0, 0, 0

        # Create BatchTasks (template keeps {content}; processor merges once).
        # ``prompt_template`` was measured above for chunk sizing (D2).
        tasks: list[BatchTask] = []
        for task_id, (_, chunk_content, _, _) in enumerate(tasks_data):
            tasks.append(
                BatchTask(
                    task_id=task_id,
                    prompt=prompt_template,
                    content=chunk_content,
                    model_settings=self.model_config,
                )
            )

        # Shared checkpoint lifecycle (P4.1): the same resume filtering,
        # incremental saves and unlink-on-success semantics the text/batch
        # paths use — a long notebook no longer loses all paid work on
        # interrupt. Checkpoints live next to the output.
        checkpoint_path = f"{output_path}.trans_checkpoint.json"
        outcome = run_with_checkpoint(
            command="notebook",
            config_digest=compute_checkpoint_digest(input_path, tasks),
            checkpoint_path=checkpoint_path,
            tasks=tasks,
            config_manager=config_manager,
            resume=resume,
            max_retries=max_retries,
            max_workers=max_workers,
            show_progress=show_progress,
            clamp_workers_to_task_count=True,
            stream_api=stream_api,
        )
        results = sorted(outcome.results, key=lambda r: r.task_id)
        self.last_results = list(results)

        processor = outcome.processor
        if outcome.interrupted:
            logger.warning(
                f"Notebook translation interrupted: progress saved to {checkpoint_path}; "
                "resume to continue."
            )

        successful = sum(1 for r in results if r.status == TaskStatus.SUCCESS)
        failed = len(results) - successful
        if successful == 0 and failed > 0 and processor and processor.auth_error_logged:
            raise NotebookAuthError("API authentication failed; no translated output.")

        # Build cell_index -> list of (translated, span) in order
        result_map = {r.task_id: r for r in results}
        cell_translations: dict[int, list[tuple[str, tuple[int, int]]]] = {}
        for task_id, (cell_idx, _, chunk_start, chunk_end) in enumerate(tasks_data):
            result = result_map.get(task_id)
            if result and result.response and result.status == TaskStatus.SUCCESS:
                translated = result.response.strip()
            else:
                translated = tasks_data[task_id][1]
                logger.warning(f"Translation failed for cell {cell_idx} chunk, keeping original")

            cell_translations.setdefault(cell_idx, []).append(
                (translated, (chunk_start, chunk_end))
            )

        # Merge chunks per cell and update notebook in place: mutating the
        # existing cell preserves attachments (embedded images) and the cell
        # ``id`` required by nbformat >= 4.5 — rebuilding the node from a
        # 3-key dict silently dropped both. The position-aware joiner restores
        # the cell's original inter-chunk separators; ``"\n\n"`` is the
        # fallback when spans are unusable.
        for cell_idx, translated_items in cell_translations.items():
            cell_source = notebook.cells[cell_idx].source
            if isinstance(cell_source, list):
                cell_source = "".join(cell_source)
            parts = [t for t, _ in translated_items]
            spans = [s for _, s in translated_items]
            joined = join_chunks_position_aware(parts, spans, cell_source)
            notebook.cells[cell_idx].source = joined if joined is not None else "\n\n".join(parts)
        translated_cells = notebook.cells

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

        # Audit 2.8: totals reflect ALL attempts (what the user actually paid),
        # matching ExecutionReport — not just successful chunks.
        with_meta = [r for r in results if r.metadata]
        total_in = sum(r.metadata.input_tokens for r in with_meta if r.metadata)
        total_out = sum(r.metadata.output_tokens for r in with_meta if r.metadata)
        logger.info(f"Translated notebook saved to: {output_path}")
        logger.info(f"Statistics: {successful} chunks translated, {failed} failed")

        return successful, failed, total_in, total_out
