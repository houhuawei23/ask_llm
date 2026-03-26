"""Batch task ordering by estimated input size."""

from ask_llm.core.batch import BatchTask, ModelConfig, sort_batch_tasks_by_estimated_input


def test_sort_batch_tasks_descending_by_tokens() -> None:
    mc = ModelConfig(provider="deepseek", model="deepseek-chat")
    short = BatchTask(
        task_id=0,
        prompt="Translate:\n\n{content}",
        content="hi",
        task_model_config=mc,
    )
    long = BatchTask(
        task_id=1,
        prompt="Translate:\n\n{content}",
        content="word " * 400,
        task_model_config=mc,
    )
    out = sort_batch_tasks_by_estimated_input([short, long], "deepseek-chat")
    assert out[0].task_id == 1
    assert out[1].task_id == 0
