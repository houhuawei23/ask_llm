"""Single streaming implementation for batch task execution.

Consumes a provider's streaming iterator, accumulates the response (and
optional reasoning), counts output tokens, and throttles progress-bar updates.

Extracted from ``GlobalBatchProcessor`` (ARCHITECTURE_REVIEW.md §7.2 / P1.3,
``StreamCollector``) so the streaming + token-collection logic has a single
owner and is independently testable without a processor instance.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from rich.progress import Progress, TaskID

from ask_llm.core.batch_models import ModelConfig
from ask_llm.core.constants import PROGRESS_UPDATE_INTERVAL
from ask_llm.core.protocols import ReasoningChunk
from ask_llm.utils.token_counter import TokenCounter


def stream_and_collect(
    stream_iter: Iterator[str | ReasoningChunk],
    model_config: ModelConfig,
    progress: Progress | None,
    progress_task_id: TaskID | None,
    description_prefix: str,
    *,
    return_reasoning: bool = False,
) -> tuple[str, str | None, int, float]:
    """Run a streaming iterator, counting tokens and updating progress.

    Args:
        stream_iter: Iterator of text chunks (or :class:`ReasoningChunk` when
            ``return_reasoning`` is set).
        model_config: Config whose model selects the output-token encoding.
        progress: Optional ``rich.Progress`` to update (throttled).
        progress_task_id: Task id within ``progress``.
        description_prefix: Prefix for the progress-bar description.
        return_reasoning: When True, ``stream_iter`` yields
            :class:`ReasoningChunk`; reasoning text is accumulated separately.

    Returns:
        ``(response, reasoning, output_token_count, latency_seconds)``.
    """
    start_time = time.time()
    response_parts: list[str] = []
    reasoning_parts: list[str] = []
    output_token_count = 0

    encoding = TokenCounter.get_encoding(model_config.model)
    last_progress_update = 0.0
    for chunk in stream_iter:
        if return_reasoning:
            assert isinstance(chunk, ReasoningChunk)
            if chunk.content:
                response_parts.append(chunk.content)
                if encoding is not None:
                    output_token_count += len(encoding.encode(chunk.content))
                else:
                    output_token_count += TokenCounter.count_words(chunk.content)
            if chunk.reasoning:
                reasoning_parts.append(chunk.reasoning)
                if encoding is not None:
                    output_token_count += len(encoding.encode(chunk.reasoning))
                else:
                    output_token_count += TokenCounter.count_words(chunk.reasoning)
        else:
            text_chunk = chunk if isinstance(chunk, str) else str(chunk)
            response_parts.append(text_chunk)
            if encoding is not None:
                output_token_count += len(encoding.encode(text_chunk))
            else:
                output_token_count += TokenCounter.count_words(text_chunk)

        # Throttle progress updates to avoid UI flicker on high-frequency streams.
        now = time.time()
        if (
            progress
            and progress_task_id is not None
            and now - last_progress_update >= PROGRESS_UPDATE_INTERVAL
        ):
            progress.update(
                progress_task_id,
                completed=output_token_count,
                description=f"{description_prefix}: {output_token_count} tok",
            )
            last_progress_update = now

    response = "".join(response_parts).strip()
    reasoning = "".join(reasoning_parts).strip() if reasoning_parts else None
    latency = time.time() - start_time
    return response, reasoning, output_token_count, latency
