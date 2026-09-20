"""H1 regression tests: ConfigManager overrides must be per-provider.

A single global override slot made ``get_provider_config(other)`` inherit the
current provider's api_key / sampling overrides — e.g. an API key pasted into
the interactive gate leaked into every fallback or batch provider.
"""

import pytest

from ask_llm.config.manager import ConfigManager
from ask_llm.core.models import AppConfig, ProviderConfig


@pytest.fixture
def two_providers() -> AppConfig:
    return AppConfig(
        default_provider="alpha",
        default_model="alpha-model",
        providers={
            "alpha": ProviderConfig(
                api_provider="alpha",
                api_key="alpha-key",
                api_base="https://alpha.example.com/v1",
                models=["alpha-model"],
                api_temperature=0.3,
            ),
            "beta": ProviderConfig(
                api_provider="beta",
                api_key="beta-key",
                api_base="https://beta.example.com/v1",
                models=["beta-model"],
                api_temperature=0.8,
            ),
        },
    )


class TestPerProviderOverrides:
    def test_api_key_override_does_not_leak_to_other_provider(self, two_providers):
        manager = ConfigManager(two_providers)
        manager.set_provider("alpha")
        manager.apply_overrides(api_key="pasted-interactive-key")

        assert manager.get_provider_config("alpha").api_key.get_secret_value() == (
            "pasted-interactive-key"
        )
        assert manager.get_provider_config("beta").api_key.get_secret_value() == "beta-key"

    def test_sampling_overrides_do_not_leak_across_batch_providers(self, two_providers):
        """batch_service flow: set_provider(B) then apply_overrides must not
        inherit A's temperature/max_tokens leftovers."""
        manager = ConfigManager(two_providers)
        manager.set_provider("alpha")
        manager.apply_overrides(temperature=0.1, max_tokens=123)
        assert manager.get_provider_config("alpha").api_temperature == 0.1

        manager.set_provider("beta")
        manager.apply_overrides(model="beta-model")  # beta passes no temperature
        beta_cfg = manager.get_provider_config("beta")
        assert beta_cfg.api_temperature == 0.8
        assert beta_cfg.max_tokens is None

    def test_model_override_stays_global_but_out_of_provider_payload(self, two_providers):
        manager = ConfigManager(two_providers)
        manager.set_provider("alpha")
        manager.apply_overrides(model="cli-model")

        assert manager.get_model_override() == "cli-model"
        # The fake "_model_override" key must not be smuggled into the
        # validated provider config.
        alpha_dump = manager.get_provider_config("alpha").model_dump()
        assert "_model_override" not in alpha_dump

    def test_explicit_override_applies_to_named_provider_it_was_set_for(self, two_providers):
        manager = ConfigManager(two_providers)
        manager.set_provider("beta")
        manager.apply_overrides(api_base="https://override.example.com/v1")

        assert manager.get_provider_config("beta").api_base == "https://override.example.com/v1"
        assert manager.get_provider_config("alpha").api_base == "https://alpha.example.com/v1"

    def test_clear_overrides_resets_everything(self, two_providers):
        manager = ConfigManager(two_providers)
        manager.set_provider("alpha")
        manager.apply_overrides(api_key="k", temperature=0.0, model="m")
        manager.clear_overrides()

        assert manager.get_model_override() is None
        assert manager.get_provider_config("alpha").api_key.get_secret_value() == "alpha-key"
        assert manager.get_override_sources() == {}
