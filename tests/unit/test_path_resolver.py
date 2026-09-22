"""Tests for path resolution and output-target preflight (audit 2.1)."""

from __future__ import annotations

import pytest

from ask_llm.utils.path_resolver import (
    OutputTargetError,
    resolve_trans_input_paths,
    validate_multi_input_output,
    validate_output_targets,
)


@pytest.fixture
def md_tree(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("b", encoding="utf-8")
    return tmp_path


class TestResolveTransInputPaths:
    def test_tilde_glob_expands(self, md_tree, monkeypatch):
        monkeypatch.setenv("HOME", str(md_tree))
        (md_tree / "docs").mkdir(exist_ok=True)
        out = resolve_trans_input_paths(["~/docs/*.md"], [".md"], recursive_dir=False)
        assert len(out) == 2
        assert all(o.startswith(str(md_tree)) for o in out)

    def test_tilde_file_path_expands(self, md_tree, monkeypatch):
        monkeypatch.setenv("HOME", str(md_tree))
        out = resolve_trans_input_paths(["~/docs/a.md"], [".md"], recursive_dir=False)
        assert out == [str((md_tree / "docs" / "a.md").resolve())]

    def test_directory_entries_are_resolved(self, md_tree):
        out = resolve_trans_input_paths([str(md_tree / "docs")], [".md"], recursive_dir=False)
        assert out == sorted(set(out))
        assert all(not o.startswith("docs/") for o in out)  # absolute, uniform


class TestValidateOutputTargets:
    def test_multi_input_single_file_output_rejected(self, md_tree):
        targets = [
            str(md_tree / "out_a.md"),
            str(md_tree / "out_a.md"),  # both inputs map to the same file
        ]
        with pytest.raises(OutputTargetError, match="same output path"):
            validate_output_targets(targets, force=False)

    def test_overwrite_requires_force(self, md_tree):
        existing = md_tree / "exists.md"
        existing.write_text("x", encoding="utf-8")
        with pytest.raises(OutputTargetError, match="--force"):
            validate_output_targets([str(existing)], force=False)

    def test_force_allows_existing(self, md_tree):
        existing = md_tree / "exists.md"
        existing.write_text("x", encoding="utf-8")
        validate_output_targets([str(existing)], force=True)

    def test_resume_allows_existing(self, md_tree):
        existing = md_tree / "partial.md"
        existing.write_text("x", encoding="utf-8")
        validate_output_targets([str(existing)], force=False, resume=True)

    def test_duplicate_detection_survives_relative_forms(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "x.md"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(OutputTargetError):
            validate_output_targets(["out.md", str(tmp_path / "out.md")], force=True)


class TestValidateMultiInputOutput:
    def test_rejects_single_file_output_for_many_inputs(self, md_tree):
        with pytest.raises(OutputTargetError):
            validate_multi_input_output(str(md_tree / "combined.md"), 3, inplace=False)

    def test_allows_single_input(self, md_tree):
        validate_multi_input_output(str(md_tree / "combined.md"), 1, inplace=False)

    def test_allows_directory_output(self, md_tree):
        d = md_tree / "outdir"
        d.mkdir()
        validate_multi_input_output(str(d), 3, inplace=False)

    def test_inplace_bypasses(self, md_tree):
        validate_multi_input_output(str(md_tree / "combined.md"), 3, inplace=True)
