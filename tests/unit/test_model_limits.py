"""Tests for providers.yml model limits and max_tokens resolution."""

import textwrap
from pathlib import Path

import pytest

from ask_llm.utils.model_limits import (
    ModelLimits,
    load_providers_model_limits,
    resolve_paper_max_tokens,
)


def test_load_providers_model_limits_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    yml = textwrap.dedent(
        """\
        providers:
          deepseek:
            models:
              - name: deepseek-chat
                context_length: 128000
                max_output:
                  default: 4096
                  maximum: 8192
              - name: deepseek-reasoner
                context_length: 128000
                max_output:
                  default: 4096
                  maximum: 64000
        """
    )
    p = tmp_path / "providers.yml"
    p.write_text(yml, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)

    limits, used = load_providers_model_limits()
    assert used == p.resolve()
    assert limits["deepseek-chat"] == ModelLimits(128000, 4096, 8192)
    assert limits["deepseek-reasoner"] == ModelLimits(128000, 4096, 64000)


def test_resolve_paper_max_tokens_from_catalog():
    limits = {
        "deepseek-chat": ModelLimits(128000, 4096, 8192),
        "deepseek-reasoner": ModelLimits(128000, 4096, 64000),
    }
    assert resolve_paper_max_tokens("deepseek-chat", 65536, limits) == 8192
    assert resolve_paper_max_tokens("deepseek-reasoner", 100000, limits) == 64000
    reasoner_hi = {
        "deepseek-reasoner": ModelLimits(128000, 4096, 65536),
    }
    assert resolve_paper_max_tokens("deepseek-reasoner", 100000, reasoner_hi) == 65536
    assert resolve_paper_max_tokens("deepseek-chat", 4000, limits) == 4000
    assert resolve_paper_max_tokens(None, 16000, limits) == 16000
    assert resolve_paper_max_tokens("gpt-4o", 32000, limits) == 32000
    kimi_limits = {
        "kimi-k2-0711-preview": ModelLimits(256000, 8000, 32000),
    }
    assert resolve_paper_max_tokens("kimi-k2-0711-preview", 65536, kimi_limits) == 32000


def test_resolve_paper_max_tokens_unknown_deepseek_fallback():
    assert resolve_paper_max_tokens("deepseek-unknown", 65536, {}) == 8192


def test_bundled_providers_yml_loads():
    """Bundled ``providers.yml`` (package root) parses and lists known models."""
    limits, path = load_providers_model_limits()
    assert path is not None
    assert "deepseek-chat" in limits
    assert limits["deepseek-chat"].max_output_maximum == 8192
    assert "deepseek-reasoner" in limits
    assert limits["deepseek-reasoner"].max_output_maximum == 65536


class TestAudit44DeepseekCapPrecedence:
    """Audit 4.4: the substring 8192 clamp yields to an explicit catalog maximum."""

    def test_unknown_deepseek_with_declared_maximum_not_clamped(self):
        from ask_llm.utils.model_limits import ModelLimits

        limits = {"my-deepseek-proxy-v3": ModelLimits(128000, 8192, 32768)}
        assert resolve_paper_max_tokens("my-deepseek-proxy-v3", 65536, limits) == 32768

    def test_unknown_deepseek_without_declared_maximum_still_clamped(self):
        assert resolve_paper_max_tokens("deepseek-unknown", 65536, {}) == 8192

    def test_exact_known_ids_keep_hard_api_cap(self):
        """Verified API caps apply even when the catalog says otherwise."""
        from ask_llm.utils.model_limits import ModelLimits

        limits = {"deepseek-chat": ModelLimits(128000, 8192, 999999)}
        assert resolve_paper_max_tokens("deepseek-chat", 65536, limits) == 8192
