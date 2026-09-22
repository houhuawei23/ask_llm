"""Unit tests for provider fallback chain resolution (audit 4.7)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from ask_llm.core.batch_models import ModelConfig
from ask_llm.core.models import AppConfig, FallbackConfig, ProviderConfig
from ask_llm.utils.fallback_chain import build_fallback_chain, model_config_with_fallback


def _app_config_with_fallbacks(primary: str, fallbacks: list[FallbackConfig]) -> AppConfig:
    return AppConfig(
        default_provider=primary,
        providers={
            primary: ProviderConfig(
                api_provider=primary,
                api_key=SecretStr("sk-test"),
                api_base=f"https://{primary}.example.com/v1",
                models=["m1"],
                fallback_to=fallbacks,
            ),
            # Other referenced providers need config entries too (never
            # overwrite the primary entry itself).
            **{
                fb.provider: ProviderConfig(
                    api_provider=fb.provider,
                    api_key=SecretStr("sk-test"),
                    api_base=f"https://{fb.provider}.example.com/v1",
                    models=[fb.model],
                )
                for fb in fallbacks
                if fb.provider != primary
            },
        },
    )


class TestFallbackChain:
    def test_chain_built_in_order_with_primary_sampling_defaults(self):
        cfg = _app_config_with_fallbacks(
            "alpha",
            [
                FallbackConfig(provider="beta", model="bm"),
                FallbackConfig(provider="gamma", model="gm", temperature=0.2),
            ],
        )
        chain = build_fallback_chain(
            cfg, ModelConfig(provider="alpha", model="m1", temperature=0.9)
        )

        assert [(c.provider, c.model) for c in chain] == [("beta", "bm"), ("gamma", "gm")]
        assert chain[0].temperature == 0.9  # inherits primary
        assert chain[1].temperature == 0.2  # explicit wins

    def test_unknown_provider_yields_empty_chain(self):
        cfg = _app_config_with_fallbacks("alpha", [])
        assert build_fallback_chain(cfg, ModelConfig(provider="missing", model="m")) == []

    def test_no_fallbacks_yields_empty_chain(self):
        cfg = _app_config_with_fallbacks("alpha", [])
        assert build_fallback_chain(cfg, ModelConfig(provider="alpha", model="m1")) == []


class TestAudit47ChainHygiene:
    """Audit 4.7: the chain excludes the primary endpoint and dedupes."""

    def test_primary_as_own_fallback_is_dropped(self):
        cfg = _app_config_with_fallbacks(
            "alpha",
            [
                FallbackConfig(provider="alpha", model="m1"),  # == primary
                FallbackConfig(provider="beta", model="bm"),
            ],
        )
        chain = build_fallback_chain(cfg, ModelConfig(provider="alpha", model="m1"))

        assert [(c.provider, c.model) for c in chain] == [("beta", "bm")]

    def test_duplicate_entries_keep_first_occurrence(self):
        cfg = _app_config_with_fallbacks(
            "alpha",
            [
                FallbackConfig(provider="beta", model="bm", temperature=0.2),
                FallbackConfig(provider="beta", model="bm", temperature=0.8),  # dup
                FallbackConfig(provider="beta", model="bm2"),
            ],
        )
        chain = build_fallback_chain(cfg, ModelConfig(provider="alpha", model="m1"))

        assert [(c.provider, c.model) for c in chain] == [("beta", "bm"), ("beta", "bm2")]
        assert chain[0].temperature == 0.2  # first entry's params win

    def test_all_primary_entries_collapses_to_empty(self):
        cfg = _app_config_with_fallbacks(
            "alpha",
            [FallbackConfig(provider="alpha", model="m1")],
        )
        chain = build_fallback_chain(cfg, ModelConfig(provider="alpha", model="m1"))
        assert chain == []

    def test_model_config_with_fallback_respects_use_fallback_flag(self):
        cfg = _app_config_with_fallbacks("alpha", [FallbackConfig(provider="beta", model="bm")])

        primary, chain = model_config_with_fallback(
            "alpha",
            "m1",
            temperature=None,
            max_tokens=None,
            app_config=cfg,
            use_fallback=False,
        )
        assert chain == []
        assert (primary.provider, primary.model) == ("alpha", "m1")

        _primary, chain_on = model_config_with_fallback(
            "alpha", "m1", temperature=None, max_tokens=None, app_config=cfg
        )
        assert [(c.provider, c.model) for c in chain_on] == [("beta", "bm")]

    def test_none_app_config_yields_empty_chain(self):
        _primary, chain = model_config_with_fallback(
            "alpha", "m1", temperature=None, max_tokens=None, app_config=None
        )
        assert chain == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
