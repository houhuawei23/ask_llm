"""Factories for :class:`~ask_llm.core.batch.BatchTask` (paper explain, etc.)."""

from __future__ import annotations

from ask_llm.core.batch import BatchTask, ModelConfig


def build_paper_explain_task(
    task_id: int,
    prompt: str,
    *,
    model_settings: ModelConfig,
    output_filename: str,
    return_reasoning: bool,
    fallback_model_configs: list[ModelConfig] | None = None,
) -> BatchTask:
    """Single paper-explain job (streaming API, optional reasoning channel)."""
    return BatchTask(
        task_id=task_id,
        prompt=prompt,
        content="",
        output_filename=output_filename,
        model_settings=model_settings,
        task_kind="paper_explain",
        return_reasoning=return_reasoning,
        fallback_model_configs=fallback_model_configs or [],
    )
