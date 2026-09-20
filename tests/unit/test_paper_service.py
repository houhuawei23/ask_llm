"""Unit tests for PaperService orchestration (H6 partial-success semantics)."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.batch_models import BatchResult, BatchTask, ModelConfig, TaskStatus
from ask_llm.core.models import AppConfig, ProviderConfig, RequestMetadata
from ask_llm.core.paper_explain_pipeline import PaperExplainPipelineConfig
from ask_llm.services.paper_service import PaperExplainOptions, PaperService


def _result(task_id: int, status: TaskStatus, body: str = "answer") -> BatchResult:
    task = BatchTask(
        task_id=task_id,
        prompt="explain",
        content="",
        output_filename=f"paper:section{task_id}",
        model_settings=ModelConfig(provider="openai", model="gpt-4"),
        task_kind="paper_explain",
    )
    return BatchResult(
        task_id=task_id,
        prompt=task.prompt,
        content=task.content,
        model_settings=task.model_settings,
        output_filename=task.output_filename,
        response=body if status == TaskStatus.SUCCESS else None,
        error=None if status == TaskStatus.SUCCESS else "429 exhausted",
        metadata=RequestMetadata(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            input_tokens=10,
            output_tokens=10,
            latency=0.2,
        ),
        status=status,
    )


def _make_service() -> PaperService:
    config_manager = MagicMock()
    config_manager.current_provider_name = "openai"
    paper_cfg = MagicMock(
        prompt_dir="prompts",
        pipeline_config="",
        output_subdir="explain",
        max_output_tokens=512,
        full_model="deepseek-reasoner",
        retries=1,
    )
    unified_config = MagicMock(paper=paper_cfg)
    return PaperService(
        config_manager,
        unified_config,
        provider="openai",
        model="gpt-4",
        pricing_map={},
    )


def _options() -> PaperExplainOptions:
    return PaperExplainOptions(
        run_mode="sections",
        section_filter=None,
        temperature=None,
        force=True,
        include_metadata=False,
        concurrency=4,
        dry_run=False,
        resume=False,
        pipeline_path=None,
        use_fallback=False,
        retries=1,
    )


class TestPartialSuccessPersistence:
    def test_successes_written_despite_failures(self, tmp_path: Path):
        """H6 regression: one failed job must not discard the paid-for
        successful jobs — their outputs are written before the failure is
        reported, and the result carries partial-success counts."""
        paper_md = tmp_path / "paper.md"
        paper_md.write_text(
            "# My Paper\n\n## Abstract\n\nShort abstract.\n\n## Methods\n\nWe did things.\n",
            encoding="utf-8",
        )
        service = _make_service()

        with (
            patch(
                "ask_llm.services.paper_service.load_paper_explain_pipeline",
                return_value=PaperExplainPipelineConfig.builtin(),
            ),
            patch(
                "ask_llm.services.paper_service.load_providers_model_limits",
                return_value=({}, None),
            ),
            patch.object(service, "_render_job_prompt", return_value=("tpl", "prompt")),
            patch(
                "ask_llm.services.paper_service.run_global_batch_tasks",
                return_value=(
                    [
                        _result(0, TaskStatus.SUCCESS, "abstract answer"),
                        _result(1, TaskStatus.FAILED),
                    ],
                    MagicMock(last_metrics=MagicMock(interrupted=False)),
                ),
            ),
        ):
            session = service.explain_paper(paper_md, _options())

        assert session.status == "failed"
        assert session.succeeded_count == 1
        assert session.failed_count == 1
        # The successful job's output must be on disk.
        explain_dir = tmp_path / "explain"
        written = [p for p in explain_dir.rglob("*.md") if p.is_file()]
        assert written, "successful job output missing from explain dir"
        assert any("answer" in p.read_text(encoding="utf-8") for p in written)

    def test_all_success_reports_ok(self, tmp_path: Path):
        paper_md = tmp_path / "paper.md"
        paper_md.write_text(
            "# My Paper\n\n## Abstract\n\nShort abstract.\n\n## Methods\n\nWe did things.\n",
            encoding="utf-8",
        )
        service = _make_service()
        with (
            patch(
                "ask_llm.services.paper_service.load_paper_explain_pipeline",
                return_value=PaperExplainPipelineConfig.builtin(),
            ),
            patch(
                "ask_llm.services.paper_service.load_providers_model_limits",
                return_value=({}, None),
            ),
            patch.object(service, "_render_job_prompt", return_value=("tpl", "prompt")),
            patch(
                "ask_llm.services.paper_service.run_global_batch_tasks",
                return_value=(
                    [
                        _result(0, TaskStatus.SUCCESS, "abstract answer"),
                        _result(1, TaskStatus.SUCCESS, "methods answer"),
                    ],
                    MagicMock(last_metrics=MagicMock(interrupted=False)),
                ),
            ),
        ):
            session = service.explain_paper(paper_md, _options())

        assert session.status == "ok"
        assert session.succeeded_count == 0  # only set on partial failure
        assert (tmp_path / "explain").exists()
