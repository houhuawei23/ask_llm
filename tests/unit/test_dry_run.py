"""Unit tests for the zero-network dry-run estimators (plan 5.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ask_llm.services.dry_run import (
    DryRunReport,
    estimate_batch_run,
    estimate_translation_file,
    estimate_translation_run,
)


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(
        "# Title\n\n" + ("Some paragraph content to be translated. " * 200),
        encoding="utf-8",
    )
    return p


class TestTranslationDryRun:
    def test_estimates_chunks_and_tokens_no_network(self, md_file: Path):
        est = estimate_translation_file(
            md_file,
            "gpt-4",
            target_language="en",
            max_chunk_tokens=100,
            balance_chunks=True,
        )
        assert est is not None
        assert est.chunks > 1
        assert est.input_tokens > 0

    def test_unsupported_and_empty_files_are_skipped(self, tmp_path: Path):
        binary = tmp_path / "img.png"
        binary.write_bytes(b"\x89PNG\r\n")
        empty = tmp_path / "empty.md"
        empty.write_text("   \n", encoding="utf-8")

        assert (
            estimate_translation_file(binary, "gpt-4", target_language="en", max_chunk_tokens=100)
            is None
        )
        assert (
            estimate_translation_file(empty, "gpt-4", target_language="en", max_chunk_tokens=100)
            is None
        )

    def test_run_report_totals_and_cost(self, md_file: Path, tmp_path: Path):
        pricing = {
            ("deepseek", "deepseek-chat"): {"input": 2.0, "output": 8.0, "input_cache_hit": 0.0}
        }
        report = estimate_translation_run(
            [md_file],
            "deepseek-chat",
            "deepseek",
            target_language="en",
            max_chunk_tokens=200,
            pricing_map=pricing,
        )
        assert report.kind == "translation"
        assert report.files and report.input_tokens > 0
        assert report.est_output_tokens >= report.input_tokens
        assert report.est_cost_cny is not None and report.est_cost_cny > 0
        lines = report.render()
        assert any("Dry run" in line for line in lines)
        assert any("¥" in line for line in lines)

    def test_missing_pricing_leaves_cost_none(self, md_file: Path):
        report = estimate_translation_run(
            [md_file],
            "unknown-model",
            "unknown",
            target_language="en",
            max_chunk_tokens=200,
            pricing_map={},
        )
        assert report.est_cost_cny is None


class TestBatchDryRun:
    def test_request_count_matches_tasks(self):
        pricing = {("p", "m"): {"input": 1.0, "output": 2.0, "input_cache_hit": 0.0}}
        tasks = [("Translate: {content}", "hello"), ("Translate: {content}", "world")]
        report = estimate_batch_run(tasks, "p", "m", pricing_map=pricing)
        assert report.kind == "batch"
        assert report.task_count == 2
        assert report.input_tokens > 0
        assert report.est_cost_cny is not None

    def test_render_mentions_tasks(self):
        report = DryRunReport(provider="p", model="m", kind="batch", task_count=7)
        lines = report.render()
        assert any("Tasks: 7" in line for line in lines)
