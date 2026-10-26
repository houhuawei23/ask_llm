"""Regression: llm_engine per-model flags must not depend on the process cwd.

``ask-llm trans`` appeared to hang when launched outside the repo, while the
same command inside the repo streamed instantly. Root cause: ``llm_engine``
resolves per-model metadata by calling ``load_providers_config()`` with no
path, and that loader's search order starts at ``Path("providers.yml")`` —
i.e. the *cwd*. In this deployment the other candidates
(``site-packages/providers.yml``, ``~/.config/llm-engine/providers.yml``) do
not exist, so outside the repo the lookup failed silently and
``reasoning: false`` was never seen:

* ``thinking: {"type": "disabled"}`` was not sent, so DeepSeek thinking models
  streamed ``reasoning_content`` for 30-170s+ before any content — the
  translation pipeline ignores reasoning, so the progress bar sat at
  "Processing..." with 0 tokens (and ``max_tokens`` could be burned into an
  empty body).
* ``token_per_character`` / ``functions.json_output`` also fell back to
  defaults.

``engine_facade.bind_engine_providers_yml`` pins that lookup to ask_llm's
already-resolved runtime catalog, making behavior cwd-independent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr

from ask_llm.core.models import ProviderConfig
from ask_llm.utils import engine_facade
from ask_llm.utils.engine_facade import (
    bind_engine_providers_yml,
    create_engine_adapter,
)

_MODEL = "deepseek-v4-flash"


def _write_catalog(path: Path, *, reasoning: bool, marker: str) -> Path:
    """Write a minimal providers.yml carrying one model's ``reasoning`` flag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "deepseek": {
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key": "sk-test",
                        "models": [{"name": _MODEL, "reasoning": reasoning}],
                        "marker": marker,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _model_flags(catalog: dict) -> dict:
    return catalog["providers"]["deepseek"]["models"][0]


@pytest.fixture(autouse=True)
def _restore_engine_loader():
    """Every test starts and ends with the engine's own (unbound) loader."""
    engine_facade._reset_engine_providers_yml_binding()
    yield
    engine_facade._reset_engine_providers_yml_binding()


@pytest.fixture
def catalogs(tmp_path, monkeypatch):
    """Runtime catalog (reasoning off) + a conflicting one in the cwd."""
    runtime = _write_catalog(
        tmp_path / "user_config" / "providers.yml", reasoning=False, marker="runtime"
    )
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    _write_catalog(workdir / "providers.yml", reasoning=True, marker="cwd")
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(engine_facade, "resolved_runtime_providers_yml", lambda: runtime.resolve())
    return runtime, workdir


class TestBindEngineProvidersYml:
    def test_without_binding_the_engine_follows_the_cwd(self, catalogs):
        """The bug this module guards against: cwd wins with no binding."""
        from llm_engine.config_loader import load_providers_config

        assert _model_flags(load_providers_config())["reasoning"] is True

    def test_binding_pins_lookup_to_the_runtime_catalog(self, catalogs):
        runtime, _workdir = catalogs

        assert bind_engine_providers_yml() == runtime.resolve()

        from llm_engine.config_loader import load_providers_config

        catalog = load_providers_config()
        assert catalog["providers"]["deepseek"]["marker"] == "runtime"
        assert _model_flags(catalog)["reasoning"] is False

    def test_explicit_path_still_wins_over_the_binding(self, catalogs):
        _runtime, workdir = catalogs
        bind_engine_providers_yml()

        from llm_engine.config_loader import load_providers_config

        catalog = load_providers_config(workdir / "providers.yml")
        assert catalog["providers"]["deepseek"]["marker"] == "cwd"

    def test_binding_is_idempotent_and_resettable(self, catalogs):
        from llm_engine import config_loader

        original = config_loader.load_providers_config
        bind_engine_providers_yml()
        bound = config_loader.load_providers_config
        assert getattr(bound, "__ask_llm_bound__", False) is True

        bind_engine_providers_yml()  # second bind must not stack wrappers
        assert config_loader.load_providers_config is bound

        engine_facade._reset_engine_providers_yml_binding()
        assert config_loader.load_providers_config is original


class TestAdapterUsesBoundCatalog:
    @staticmethod
    def _provider_config() -> ProviderConfig:
        return ProviderConfig(
            api_provider="deepseek",
            api_key=SecretStr("sk-test"),
            api_base="https://api.deepseek.com/v1",
            models=[_MODEL],
            timeout=30.0,
        )

    def test_thinking_is_disabled_from_any_directory(self, catalogs):
        adapter = create_engine_adapter(self._provider_config(), default_model=_MODEL)

        inner = adapter._provider
        assert inner._reasoning_disabled is True
        payload = inner._build_payload([{"role": "user", "content": "hi"}], stream=True)
        assert payload["thinking"] == {"type": "disabled"}
