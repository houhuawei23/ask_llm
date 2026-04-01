"""Tests for cli_session bootstrap helpers and BatchTask kind coercion."""

from ask_llm.core.batch import BatchTask, ModelConfig
from ask_llm.core.tasks.builders import build_paper_explain_task


def test_batch_task_legacy_paper_mode_sets_kind() -> None:
    mc = ModelConfig(provider="x", model="m")
    t = BatchTask(
        task_id=0,
        prompt="p",
        content="",
        task_model_config=mc,
        paper_mode=True,
    )
    assert t.task_kind == "paper_explain"


def test_build_paper_explain_task() -> None:
    mc = ModelConfig(provider="p", model="m", max_tokens=100)
    t = build_paper_explain_task(
        1,
        "full prompt",
        task_model_config=mc,
        output_filename="paper:full",
        return_reasoning=True,
    )
    assert t.task_kind == "paper_explain"
    assert t.return_reasoning is True
    assert t.content == ""
