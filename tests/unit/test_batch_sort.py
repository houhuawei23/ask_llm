"""Batch task token estimation shared with longest-first ordering (M5/2.25)."""

from ask_llm.core.batch_models import (
    BatchTask,
    ModelConfig,
    estimate_batch_task_tokens,
)


def test_estimates_tokens_per_task() -> None:
    mc = ModelConfig(provider="deepseek", model="deepseek-chat")
    short = BatchTask(
        task_id=0,
        prompt="Translate:\n\n{content}",
        content="hi",
        model_settings=mc,
    )
    long = BatchTask(
        task_id=1,
        prompt="Translate:\n\n{content}",
        content="word " * 400,
        model_settings=mc,
    )
    estimates = estimate_batch_task_tokens([short, long], "deepseek-chat")
    assert [t.task_id for t, _ in estimates] == [0, 1]
    (short_est, long_est) = (est for _, est in estimates)
    assert long_est > short_est > 0


def test_longest_first_sort_matches_batch_processor() -> None:
    """The sort applied in process_global_tasks puts heavy tasks first."""
    mc = ModelConfig(provider="deepseek", model="deepseek-chat")
    short = BatchTask(
        task_id=0,
        prompt="Translate:\n\n{content}",
        content="hi",
        model_settings=mc,
    )
    long = BatchTask(
        task_id=1,
        prompt="Translate:\n\n{content}",
        content="word " * 400,
        model_settings=mc,
    )
    task_estimates = estimate_batch_task_tokens([short, long], "deepseek-chat")
    task_estimates.sort(key=lambda pair: pair[1], reverse=True)
    assert [t.task_id for t, _ in task_estimates] == [1, 0]
