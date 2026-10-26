"""Integration: notebook translation through the real execution chain.

Locks in the P1 unification (notebook path must use the shared checkpoint
lifecycle) and the P0 cell-merge fix (in-place mutation keeps attachments and
cell ids).
"""

from __future__ import annotations

from typing import ClassVar

import nbformat
import pytest

from ask_llm.config.manager import ConfigManager
from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.models import ProviderConfig
from ask_llm.core.translator import Translator
from ask_llm.utils.notebook_translator import NotebookTranslator
from ask_llm.utils.provider_cache import ProviderAdapterCache


class _FakeAdapter:
    provider = "test"
    name = "test"
    available_models: ClassVar[list[str]] = []

    def __init__(self, config, default_model=None):
        self.config = config
        self.default_model = default_model
        self.calls: list[str] = []

    def test_connection(self):
        return True, "ok", 0.01

    def call(self, prompt=None, messages=None, temperature=None, model=None, stream=False, **kw):
        content = ""
        if messages:
            content = messages[-1].get("content", "")
        elif prompt:
            content = prompt
        self.calls.append(content)
        reply = f"translated[{content[-12:]}]"
        if stream:
            yield reply
        else:
            return reply


@pytest.fixture
def fake_adapter(monkeypatch):
    created = []

    def _create(config, default_model=None, **kw):
        adapter = _FakeAdapter(config, default_model)
        created.append(adapter)
        return adapter

    monkeypatch.setattr("ask_llm.utils.provider_cache.create_engine_adapter", _create)
    ProviderAdapterCache.clear()
    return created


def _config_manager() -> ConfigManager:
    unified_config = UnifiedConfig(
        default_provider="test",
        default_model="test-model",
        providers={
            "test": ProviderConfig(
                api_provider="test",
                api_key="sk-real-key-123",
                api_base="https://test.example.com/v1",
                models=["test-model"],
            )
        },
    )
    return ConfigManager(unified_config)


def _notebook(tmp_path, name="nb.ipynb"):
    nb = nbformat.v4.new_notebook()
    md = nbformat.v4.new_markdown_cell("## Section one\n\nHello world content.")
    md["id"] = "cell-md-1"
    code = nbformat.v4.new_code_cell("print(1)")
    code["id"] = "cell-code-1"
    nb.cells = [md, code]
    path = tmp_path / name
    nbformat.write(nb, str(path))
    return path


def _model_config():
    from ask_llm.core.batch_models import ModelConfig

    return ModelConfig(provider="test", model="test-model", temperature=0.2, max_tokens=1024)


def test_notebook_translation_writes_checkpoint_and_preserves_cells(
    tmp_path, fake_adapter, monkeypatch
):
    monkeypatch.syspath_prepend(str(tmp_path))
    from ask_llm.config.context import set_config
    from ask_llm.config.loader import ConfigLoader

    set_config(ConfigLoader.load())

    src = _notebook(tmp_path)
    out = tmp_path / "nb_trans.ipynb"
    translator_obj = Translator(
        target_language="中文",
        source_language="英文",
        style="formal",
        custom_prompt_template=None,
        prompt_file=None,
        glossary_pairs=[],
    )
    translator = NotebookTranslator(translator=translator_obj, model_config=_model_config())

    successful, failed, _total_in, _ = translator.translate_notebook(
        str(src),
        str(out),
        _config_manager(),
        max_workers=2,
        max_retries=1,
        show_progress=False,
        balance_chunks=False,
    )

    # The markdown cell was translated; checkpoint lifecycle ran to completion
    # and removed its own checkpoint on full success.
    assert successful >= 1
    assert failed == 0
    checkpoint_file = tmp_path / "nb_trans.ipynb.trans_checkpoint.json"
    assert not checkpoint_file.exists(), "checkpoint must be unlinked on full success"

    written = nbformat.read(str(out), as_version=4)
    md_cell = written.cells[0]
    # P0: in-place merge — the cell keeps its explicit id and type.
    assert md_cell["id"] == "cell-md-1"
    assert md_cell.cell_type == "markdown"
    assert "translated" in md_cell.source
    # The code cell is untouched.
    assert written.cells[1]["id"] == "cell-code-1"
