"""Unit tests for the ChunkedLLMJob base orchestration skeleton.

A tiny concrete subclass drives ``_run_units`` through the real bounded runner;
checkpoint writing runs against ``tmp_path``. The processor is the only mock
(external LLM boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.chunked_llm_job import ChunkedLLMJob
from ask_llm.core.format_checkpoint import (
    CHECKPOINT_VERSION,
    FailedChunkInfo,
    FormatCheckpoint,
    SuccessfulChunkInfo,
    generate_checkpoint_path,
)


@dataclass
class Unit:
    unit_id: int
    payload: str


@dataclass
class JobResult:
    unit_id: int
    value: str
    failed: bool = False
    error: str = ""
    retry_count: int = 0


class EchoJob(ChunkedLLMJob):
    """Concrete job: worker echoes the unit payload (or fails on demand)."""

    def __init__(self, *, concurrency=2, retries=1, fail_payloads=()):
        super().__init__(
            MagicMock(),
            concurrency=concurrency,
            retries=retries,
            retry_delay=0.0,
            retry_delay_max=0.0,
        )
        self.fail_payloads = set(fail_payloads)

    def worker(self, unit: Unit, retry_count: int) -> JobResult:
        if unit.payload in self.fail_payloads:
            return JobResult(
                unit.unit_id,
                "",
                failed=True,
                error="rate limit exceeded",
                retry_count=retry_count,
            )
        return JobResult(unit.unit_id, f"done:{unit.payload}", retry_count=retry_count)


def make_units(n):
    return [Unit(i, f"payload-{i}") for i in range(n)]


class TestRunUnits:
    def test_results_sorted_by_order_key(self):
        job = EchoJob(concurrency=4)

        results = job._run_units(
            make_units(6),
            job.worker,
            is_failed=lambda r: r.failed,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: -r.unit_id,  # deliberately not the natural order
        )

        assert [r.unit_id for r in results] == [5, 4, 3, 2, 1, 0]
        assert all(not r.failed for r in results)
        assert {r.value for r in results} == {f"done:payload-{i}" for i in range(6)}

    def test_transient_failures_are_retried_then_kept_as_failures(self):
        job = EchoJob(concurrency=2, retries=1, fail_payloads={"payload-1"})
        attempts = []

        def counting_worker(unit, retry_count):
            attempts.append((unit.unit_id, retry_count))
            return job.worker(unit, retry_count)

        results = job._run_units(
            make_units(3),
            counting_worker,
            is_failed=lambda r: r.failed,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.unit_id,
        )

        by_id = {r.unit_id: r for r in results}
        # The transient failure was retried once (retry_count 0 -> 1), still
        # failed after exhausting retries, and stayed in the sorted results.
        assert sorted(rc for uid, rc in attempts if uid == 1) == [0, 1]
        assert by_id[1].failed and by_id[1].retry_count == 1
        assert not by_id[0].failed and not by_id[2].failed
        assert len(results) == 3

    def test_non_retryable_error_is_not_retried(self):
        job = EchoJob(concurrency=2, retries=3, fail_payloads=set())
        attempts = []

        def worker(unit, retry_count):
            attempts.append((unit.unit_id, retry_count))
            return JobResult(unit.unit_id, "", failed=True, error="invalid request: bad model")

        results = job._run_units(
            make_units(2),
            worker,
            is_failed=lambda r: r.failed,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.unit_id,
        )

        # Terminal validation error: one attempt each, no retry scheduling.
        assert sorted(attempts) == [(0, 0), (1, 0)]
        assert all(r.failed for r in results)

    def test_routes_concurrency_and_retry_settings_to_runner(self):
        job = EchoJob(concurrency=8, retries=2)
        with patch("ask_llm.core.chunked_llm_job.run_bounded_with_retries") as mock_run:
            mock_run.return_value = []

            job._run_units(
                make_units(3),
                job.worker,
                is_failed=lambda r: r.failed,
                error_message=lambda r: r.error,
                retry_count_from_result=lambda r: r.retry_count,
                order_key=lambda r: r.unit_id,
            )
            kwargs = mock_run.call_args.kwargs
            assert kwargs["max_workers"] == 3  # min(concurrency, len(units))
            assert kwargs["max_retries"] == 2
            assert kwargs["retry_delay"] == 0.0
            assert kwargs["retry_delay_max"] == 0.0
            assert mock_run.call_args.args[1] == job.worker  # bound-method equality

            # Empty unit list still yields a valid (single) worker pool.
            job._run_units(
                [],
                job.worker,
                is_failed=lambda r: r.failed,
                error_message=lambda r: r.error,
                retry_count_from_result=lambda r: r.retry_count,
                order_key=lambda r: r.unit_id,
            )
            assert mock_run.call_args.kwargs["max_workers"] == 1


class TestSaveCheckpoint:
    def make_failed(self):
        return [FailedChunkInfo(1, "chunk text", "tmpl", "rate limit", 2)]

    def test_noop_without_failures_or_source_file(self, tmp_path):
        job = EchoJob()
        cp = tmp_path / "cp.json"

        assert (
            job._save_checkpoint(
                source_file=str(tmp_path / "doc.md"),
                format_type="body",
                model="m",
                prompt_template="t",
                max_chunk_tokens=None,
                failed_chunks=[],
                successful_chunks=[],
                checkpoint_path=str(cp),
            )
            is None
        )
        # Failures but no source file: nothing to resume against.
        assert (
            job._save_checkpoint(
                source_file=None,
                format_type="body",
                model="m",
                prompt_template="t",
                max_chunk_tokens=None,
                failed_chunks=self.make_failed(),
                successful_chunks=[],
                checkpoint_path=str(cp),
            )
            is None
        )
        assert not cp.exists()

    def test_writes_checkpoint_that_round_trips(self, tmp_path):
        job = EchoJob()
        cp = tmp_path / "cp.json"
        failed = self.make_failed()
        successful = [SuccessfulChunkInfo(0, "formatted-0")]

        path = job._save_checkpoint(
            source_file=str(tmp_path / "doc.md"),
            format_type="body",
            model="gpt-4o",
            prompt_template="PROMPT",
            max_chunk_tokens=2400,
            failed_chunks=failed,
            successful_chunks=successful,
            checkpoint_path=str(cp),
            original_text="ORIGINAL",
            chunk_spans={0: (0, 9, "body"), 1: (9, 20, "body")},
            frontmatter="---\ntitle: x\n---\n",
        )

        assert path == str(cp)
        loaded = FormatCheckpoint.load(cp)
        assert loaded.version == CHECKPOINT_VERSION
        assert loaded.config_digest  # v4: digest always present
        assert loaded.format_type == "body"
        assert loaded.model == "gpt-4o"
        assert loaded.prompt_template == "PROMPT"
        assert loaded.max_chunk_tokens == 2400
        assert loaded.original_text == "ORIGINAL"
        assert loaded.frontmatter.startswith("---")
        assert [(fc.chunk_id, fc.error, fc.retry_count) for fc in loaded.failed_chunks] == [
            (1, "rate limit", 2)
        ]
        assert [sc.formatted_content for sc in loaded.successful_chunks] == ["formatted-0"]
        # chunk_spans dict is converted to span dicts sorted by insertion order.
        assert loaded.chunk_spans == [
            {"chunk_id": 0, "start": 0, "end": 9, "type": "body"},
            {"chunk_id": 1, "start": 9, "end": 20, "type": "body"},
        ]

    def test_default_path_is_generated_next_to_source(self, tmp_path):
        job = EchoJob()
        source = tmp_path / "doc.md"
        source.write_text("# x\n", encoding="utf-8")

        path = job._save_checkpoint(
            source_file=str(source),
            format_type="title",
            model="m",
            prompt_template="t",
            max_chunk_tokens=None,
            failed_chunks=self.make_failed(),
            successful_chunks=[],
        )

        expected = generate_checkpoint_path(str(source), "title")
        assert path == str(expected)
        assert expected.name == "doc.md.title_checkpoint.json"
        assert expected.exists()


class TestTemplateAndPickHelpers:
    def test_pick_prefers_override_only_when_not_none(self):
        assert ChunkedLLMJob._pick(None, 5) == 5
        assert ChunkedLLMJob._pick(3, 5) == 3
        assert ChunkedLLMJob._pick(0, 5) == 0  # falsy-but-not-None still wins

    def test_resolve_template_loads_from_file_and_caches(self, tmp_path):
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("  TEMPLATE BODY  \n", encoding="utf-8")
        job = EchoJob()
        job.prompt_file = str(prompt_file)

        assert job._resolve_template("need a template") == "TEMPLATE BODY"
        assert job.prompt_template == "TEMPLATE BODY"  # cached for later calls

    def test_resolve_template_raises_without_any_template(self):
        job = EchoJob()

        with pytest.raises(ValueError, match="no prompt available"):
            job._resolve_template("no prompt available")
