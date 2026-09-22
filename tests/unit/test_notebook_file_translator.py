"""Tests for the notebook file translation collaborator (previously untested).

Focus (audit 2.2): a notebook with failed cell chunks must not be reported as
a clean success.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ask_llm.services.notebook_file_translator import NotebookFileTranslator


def _make_translator(tmp_path) -> NotebookFileTranslator:
    return NotebookFileTranslator(
        MagicMock(),
        provider="openai",
        model="gpt-4",
    )


class _FakeNotebookTranslator:
    def __init__(self, *, failed: int, **_kwargs):
        self._failed = failed
        self.last_results = []

    def translate_notebook(self, **_kwargs):
        successful = 2
        return successful, self._failed, 100, 50


def test_partial_chunk_failure_counts_file_failed(tmp_path, monkeypatch):
    translator = _make_translator(tmp_path)
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("ask_llm.services.notebook_file_translator.Translator", MagicMock())
    monkeypatch.setattr(
        "ask_llm.services.notebook_file_translator.NotebookTranslator",
        lambda **kw: _FakeNotebookTranslator(failed=1, **kw),
    )
    result = translator.translate(
        str(nb),
        MagicMock(),
        output=str(tmp_path / "out"),
        output_is_dir=True,
        effective_suffix="_trans",
        force=True,
        stream=False,
        stream_api=True,
    )
    assert result.success is False
    assert result.partial is True


def test_clean_notebook_is_success(tmp_path, monkeypatch):
    translator = _make_translator(tmp_path)
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("ask_llm.services.notebook_file_translator.Translator", MagicMock())
    monkeypatch.setattr(
        "ask_llm.services.notebook_file_translator.NotebookTranslator",
        lambda **kw: _FakeNotebookTranslator(failed=0, **kw),
    )
    result = translator.translate(
        str(nb),
        MagicMock(),
        output=str(tmp_path / "out"),
        output_is_dir=True,
        effective_suffix="_trans",
        force=True,
        stream=False,
        stream_api=True,
    )
    assert result.success is True
    assert result.partial is False
