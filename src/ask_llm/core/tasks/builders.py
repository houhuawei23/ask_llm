"""Factories for :class:`~ask_llm.core.batch.BatchTask` (paper explain, etc.)."""

from __future__ import annotations

from ask_llm.core.batch import BatchTask, ModelConfig


def build_paper_explain_task(
    task_id: int,
    prompt: str,
    *,
    task_model_config: ModelConfig,
    output_filename: str,
    return_reasoning: bool,
) -> BatchTask:
    """Single paper-explain job (streaming API, optional reasoning channel)."""
    return BatchTask(
        task_id=task_id,
        prompt=prompt,
        content="",
        output_filename=output_filename,
        task_model_config=task_model_config,
        task_kind="paper_explain",
        return_reasoning=return_reasoning,
    )
