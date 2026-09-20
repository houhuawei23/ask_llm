"""Tests for prompt file resolution (@-paths, project-root detection, package fallback)."""

from pathlib import Path

import pytest

from ask_llm.utils.prompt_resolver import (
    _PACKAGE_PROMPTS_DIR,
    expand_prompt,
    load_prompt_template,
    resolve_prompt_file,
    resolve_prompt_or_template,
)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test in an empty directory: no project root, no stray files."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestResolvePromptFile:
    def test_plain_path_resolved_against_cwd(self, tmp_path):
        assert resolve_prompt_file("prompts/x.md") == (tmp_path / "prompts" / "x.md").resolve()

    def test_atpath_uses_project_root(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        (tmp_path / "prompts").mkdir()
        target = tmp_path / "prompts" / "custom.md"
        target.write_text("template", encoding="utf-8")
        assert resolve_prompt_file("@prompts/custom.md") == target.resolve()

    def test_atpath_falls_back_to_packaged_prompts(self):
        """K2 regression: with no project root (simulated pip install), @-paths
        must resolve to the packaged ask_llm/prompts copy instead of a missing
        cwd-relative path."""
        resolved = resolve_prompt_file("@prompts/md-heading-format.md")
        assert resolved.is_file(), f"expected packaged fallback, got {resolved}"
        assert resolved == (_PACKAGE_PROMPTS_DIR / "md-heading-format.md").resolve()

    def test_project_root_wins_over_packaged_copy(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "md-heading-format.md").write_text("user override", encoding="utf-8")
        assert load_prompt_template("@prompts/md-heading-format.md") == "user override"

    def test_missing_everywhere_still_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt_template("@prompts/definitely-not-here.md")


class TestLoadPromptTemplate:
    def test_loads_and_strips(self, tmp_path):
        target = tmp_path / "p.md"
        target.write_text("  hello \n", encoding="utf-8")
        assert load_prompt_template(str(target)) == "hello"


class TestExpandPrompt:
    def test_replaces_content_placeholder(self):
        assert expand_prompt("Sum: {content}", "abc") == "Sum: abc"

    def test_appends_when_no_placeholder(self):
        assert expand_prompt("Sum:", "abc") == "Sum:\n\nabc"

    def test_literal_braces_preserved(self):
        assert expand_prompt("JSON {key} {content}", "v") == "JSON {key} v"


class TestResolvePromptOrLiteral:
    """M4: ask/chat --prompt path resolution must error on missing path-like
    inputs instead of silently sending the path string to the LLM."""

    def test_missing_home_path_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            resolve_prompt_or_template("~/prompts/missing.md")

    def test_home_path_reads_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "p.md"
        target.write_text("tpl {content}", encoding="utf-8")
        assert resolve_prompt_or_template("~/p.md") == "tpl {content}"

    def test_literal_template_passthrough(self):
        assert resolve_prompt_or_template("Translate: {content}") == "Translate: {content}"

    def test_atpath_missing_everywhere_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_prompt_or_template("@prompts/definitely-not-here.md")

    def test_none_passthrough(self):
        assert resolve_prompt_or_template(None) is None
