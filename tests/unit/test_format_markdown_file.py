"""Unit tests for ask_llm.core.format_markdown_file.

BodyFormatter / HeadingFormatter execution is mocked; file I/O runs against
``tmp_path`` so output-path resolution and on-disk behavior are real.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.format_checkpoint import FailedChunkInfo
from ask_llm.core.format_markdown_file import (
    format_body_markdown_file,
    format_one_markdown_file,
)
from ask_llm.core.md_body_formatter import BodyFormatStats

MODULE = "ask_llm.core.format_markdown_file"


def make_md_file(tmp_path, name="doc.md", content="# Title\n\nSome body text.\n"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def make_body_result(
    text="FORMATTED BODY",
    *,
    failed_chunks=None,
    checkpoint_path=None,
):
    result = MagicMock()
    result.text = text
    result.stats = BodyFormatStats(total_input_tokens=11, total_output_tokens=7, total_latency=1.5)
    result.failed_chunks = failed_chunks or []
    result.checkpoint_path = checkpoint_path
    return result


@pytest.fixture
def processor():
    return MagicMock()


class TestFormatBodyMarkdownFile:
    def test_happy_path_writes_default_output_and_stats(self, processor, tmp_path):
        src = make_md_file(tmp_path)
        with patch(f"{MODULE}.BodyFormatter") as mock_bf:
            mock_bf.return_value.format_body.return_value = make_body_result()

            outcome = format_body_markdown_file(
                str(src),
                processor=processor,
                model="gpt-4o",
                prompt_file_resolved="/tmp/prompt.md",
                body_max_chunk_tokens=1234,
                body_concurrency=6,
                output=None,
                inplace=False,
                force=False,
            )

        assert outcome.ok and not outcome.skipped
        # No config set -> built-in "_formatted" suffix next to the source.
        expected_out = tmp_path / "doc_formatted.md"
        assert outcome.output_path == str(expected_out)
        assert expected_out.read_text(encoding="utf-8") == "FORMATTED BODY"
        assert (outcome.total_input_tokens, outcome.total_output_tokens) == (11, 7)
        assert outcome.message == "OK"
        assert outcome.checkpoint_path is None
        # Formatter received the routed parameters.
        kwargs = mock_bf.call_args.kwargs
        assert kwargs["processor"] is processor
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["prompt_file"] == "/tmp/prompt.md"
        assert kwargs["max_chunk_tokens"] == 1234
        assert kwargs["concurrency"] == 6
        mock_bf.return_value.format_body.assert_called_once()
        assert mock_bf.return_value.format_body.call_args.kwargs["source_file"] == str(src)

    def test_output_option_accepts_file_or_directory(self, processor, tmp_path):
        src = make_md_file(tmp_path)
        out_dir = tmp_path / "outdir"
        out_dir.mkdir()
        explicit = tmp_path / "nested" / "renamed.md"

        with patch(f"{MODULE}.BodyFormatter") as mock_bf:
            mock_bf.return_value.format_body.return_value = make_body_result()

            to_dir = format_body_markdown_file(
                str(src),
                processor=processor,
                model="m",
                prompt_file_resolved="p",
                body_max_chunk_tokens=None,
                body_concurrency=None,
                output=str(out_dir),
                inplace=False,
                force=False,
            )
            explicit.parent.mkdir()
            to_file = format_body_markdown_file(
                str(src),
                processor=processor,
                model="m",
                prompt_file_resolved="p",
                body_max_chunk_tokens=None,
                body_concurrency=None,
                output=str(explicit),
                inplace=False,
                force=False,
            )

        # Directory output: <stem>_formatted<suffix> inside the directory.
        assert to_dir.output_path == str(out_dir / "doc_formatted.md")
        assert (out_dir / "doc_formatted.md").exists()
        # Explicit file output: used verbatim (parent dirs created).
        assert to_file.output_path == str(explicit)
        assert explicit.read_text(encoding="utf-8") == "FORMATTED BODY"

    def test_existing_output_blocks_without_force_and_overwrites_with_force(
        self, processor, tmp_path
    ):
        src = make_md_file(tmp_path)
        out = tmp_path / "out.md"
        out.write_text("PRE-EXISTING", encoding="utf-8")
        common = {
            "processor": processor,
            "model": "m",
            "prompt_file_resolved": "p",
            "body_max_chunk_tokens": None,
            "body_concurrency": None,
        }

        with patch(f"{MODULE}.BodyFormatter") as mock_bf:
            mock_bf.return_value.format_body.return_value = make_body_result()

            blocked = format_body_markdown_file(
                str(src), output=str(out), inplace=False, force=False, **common
            )
            assert not blocked.ok
            assert "--force" in blocked.message
            assert blocked.output_path is None
            assert out.read_text(encoding="utf-8") == "PRE-EXISTING"

            forced = format_body_markdown_file(
                str(src), output=str(out), inplace=False, force=True, **common
            )

        assert forced.ok
        assert out.read_text(encoding="utf-8") == "FORMATTED BODY"

    def test_inplace_overwrites_source(self, processor, tmp_path):
        src = make_md_file(tmp_path)

        with patch(f"{MODULE}.BodyFormatter") as mock_bf:
            mock_bf.return_value.format_body.return_value = make_body_result()

            outcome = format_body_markdown_file(
                str(src),
                processor=processor,
                model="m",
                prompt_file_resolved="p",
                body_max_chunk_tokens=None,
                body_concurrency=None,
                output=None,
                inplace=True,
                force=False,  # even without --force, inplace may overwrite
            )

        assert outcome.ok
        assert outcome.output_path == str(src)
        assert src.read_text(encoding="utf-8") == "FORMATTED BODY"

    def test_formatter_failure_returns_error_without_writing(self, processor, tmp_path):
        src = make_md_file(tmp_path)
        with patch(f"{MODULE}.BodyFormatter") as mock_bf:
            mock_bf.return_value.format_body.side_effect = RuntimeError("api down")

            outcome = format_body_markdown_file(
                str(src),
                processor=processor,
                model="m",
                prompt_file_resolved="p",
                body_max_chunk_tokens=None,
                body_concurrency=None,
                output=None,
                inplace=False,
                force=False,
            )

        assert not outcome.ok and not outcome.skipped
        assert outcome.message == "api down"
        assert outcome.output_path is None
        assert not (tmp_path / "doc_formatted.md").exists()

    def test_partial_failure_carries_failed_chunks_and_checkpoint_path(self, processor, tmp_path):
        src = make_md_file(tmp_path)
        failed = [FailedChunkInfo(1, "chunk", "tmpl", "boom", 2)]
        with patch(f"{MODULE}.BodyFormatter") as mock_bf:
            mock_bf.return_value.format_body.return_value = make_body_result(
                failed_chunks=failed, checkpoint_path="/cp/doc.md.body_checkpoint.json"
            )

            outcome = format_body_markdown_file(
                str(src),
                processor=processor,
                model="m",
                prompt_file_resolved="p",
                body_max_chunk_tokens=None,
                body_concurrency=None,
                output=None,
                inplace=False,
                force=False,
            )

        assert outcome.ok
        assert outcome.message.startswith("Partial success: 1 chunk(s) failed")
        assert outcome.failed_chunks == failed
        assert outcome.checkpoint_path == "/cp/doc.md.body_checkpoint.json"


class TestFormatOneMarkdownFile:
    def test_title_dispatch_formatters_and_write(self, processor, tmp_path):
        src = make_md_file(tmp_path, content="# intro\n\nparagraph\n\n## details\n\nmore text\n")
        with (
            patch(f"{MODULE}.HeadingFormatter") as mock_hf,
            patch(f"{MODULE}.HeadingApplier") as mock_ha,
        ):
            formatter = mock_hf.return_value
            formatter.format_headings.return_value.formatted_headings = [
                "# Intro",
                "## Details",
            ]
            formatter.format_headings.return_value.failed_batches = []
            formatter.format_headings.return_value.checkpoint_path = None
            mock_ha.return_value.apply.return_value = "TITLE-FORMATTED DOC"

            outcome = format_one_markdown_file(
                str(src),
                processor=processor,
                prompt_file_resolved="prompts/t.md",
                heading_batch_size=42,
                heading_concurrency=3,
                output=None,
                inplace=False,
                force=False,
            )

        assert outcome.ok
        assert outcome.heading_count == 2
        assert outcome.message == "OK"
        assert (tmp_path / "doc_formatted.md").read_text(encoding="utf-8") == "TITLE-FORMATTED DOC"
        # Real extractor found both headings; formatter got them + the source.
        headings = formatter.format_headings.call_args.args[0]
        assert [h.title for h in headings] == ["intro", "details"]
        assert formatter.format_headings.call_args.kwargs["source_file"] == str(src)
        kwargs = mock_hf.call_args.kwargs
        assert kwargs["processor"] is processor
        assert kwargs["prompt_file"] == "prompts/t.md"
        assert kwargs["batch_size"] == 42
        assert kwargs["concurrency"] == 3
        # Applier receives original text + original headings + formatted list.
        apply_args = mock_ha.return_value.apply.call_args.args
        assert apply_args[0].startswith("# intro")
        assert len(apply_args[1]) == 2
        assert apply_args[2] == ["# Intro", "## Details"]

    def test_title_partial_failure_reports_batches_and_checkpoint(self, processor, tmp_path):
        src = make_md_file(tmp_path, content="# only\n\ntext\n")
        failed = [FailedChunkInfo(0, "batch", "tmpl", "err", 0)]
        with (
            patch(f"{MODULE}.HeadingFormatter") as mock_hf,
            patch(f"{MODULE}.HeadingApplier") as mock_ha,
        ):
            formatter = mock_hf.return_value
            formatter.format_headings.return_value.formatted_headings = ["# Only"]
            formatter.format_headings.return_value.failed_batches = failed
            formatter.format_headings.return_value.checkpoint_path = "/cp/title.json"
            mock_ha.return_value.apply.return_value = "PARTIAL DOC"

            outcome = format_one_markdown_file(
                str(src),
                processor=processor,
                prompt_file_resolved="p",
                heading_batch_size=None,
                heading_concurrency=None,
                output=None,
                inplace=False,
                force=False,
            )

        assert outcome.ok
        assert outcome.message.startswith("Partial success: 1 batch(es) failed")
        assert outcome.failed_chunks == failed
        assert outcome.checkpoint_path == "/cp/title.json"

    def test_skip_paths_no_headings_unsupported_suffix_empty_file(self, processor, tmp_path):
        no_headings = make_md_file(tmp_path, name="plain.md", content="just text\n\nno heads")
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        empty = make_md_file(tmp_path, name="empty.md", content="   \n")
        kwargs = {
            "processor": processor,
            "prompt_file_resolved": "p",
            "heading_batch_size": None,
            "heading_concurrency": None,
        }

        with patch(f"{MODULE}.HeadingFormatter") as mock_hf:
            r_no_headings = format_one_markdown_file(
                str(no_headings), output=None, inplace=False, force=False, **kwargs
            )
            r_bad_type = format_one_markdown_file(
                str(tmp_path / "notes.txt"), output=None, inplace=False, force=False, **kwargs
            )
            r_empty = format_one_markdown_file(
                str(empty), output=None, inplace=False, force=False, **kwargs
            )

        assert not r_no_headings.ok and r_no_headings.skipped
        assert r_no_headings.message == "No headings found"
        assert not r_bad_type.ok and r_bad_type.skipped
        assert ".txt" in r_bad_type.message
        assert not r_empty.ok and r_empty.skipped
        assert r_empty.message == "File is empty"
        # The LLM formatter was never constructed on any skip path.
        mock_hf.assert_not_called()
