"""Tests for interactive API-key handling (write target, atomicity, env leak).

Covers the previously untested ``utils/interactive_config.py`` (audit 1.3):
secrets must land in the user config dir only, atomically, mode 0600, and the
chat ``!shell`` escape must not inherit the injected key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ask_llm.core.chat import ChatSession
from ask_llm.utils.interactive_config import InteractiveConfigHelper

# _save_api_key_to_config does not touch config_manager; None keeps the unit
# test free of config-loading scaffolding.
_HELPER = InteractiveConfigHelper(None)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # Clear the catalog env override so seeding tests control the catalog.
    monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)


def _user_providers_yml(tmp_path: Path) -> Path:
    return tmp_path / ".config" / "ask_llm" / "providers.yml"


class TestSaveApiKeyToConfig:
    def test_key_written_to_user_config_not_cwd(self, tmp_path):
        """A providers.yml in cwd must never receive the secret (audit 1.2/1.3)."""
        cwd_yml = tmp_path / "providers.yml"
        cwd_yml.write_text("providers:\n  deepseek:\n    base_url: x\n", encoding="utf-8")

        _HELPER._save_api_key_to_config("deepseek", "sk-test")

        assert "sk-test" not in cwd_yml.read_text(encoding="utf-8")
        saved = _user_providers_yml(tmp_path)
        assert saved.exists()

    def test_key_file_mode_0600(self, tmp_path):
        _HELPER._save_api_key_to_config("deepseek", "sk-test")
        saved = _user_providers_yml(tmp_path)
        assert (saved.stat().st_mode & 0o777) == 0o600

    def test_key_stored_under_provider(self, tmp_path):
        _HELPER._save_api_key_to_config("deepseek", "sk-test")
        data = yaml.safe_load(_user_providers_yml(tmp_path).read_text(encoding="utf-8"))
        assert data["providers"]["deepseek"]["api_key"] == "sk-test"

    def test_user_file_seeded_from_catalog(self, tmp_path, monkeypatch):
        """No user file yet: it is seeded from the resolved catalog copy."""
        catalog = tmp_path / "catalog" / "providers.yml"
        catalog.parent.mkdir(parents=True)
        catalog.write_text(
            yaml.safe_dump(
                {
                    "providers": {
                        "deepseek": {"base_url": "https://api.deepseek.com/v1"},
                    },
                    "default_provider": "deepseek",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "ask_llm.utils.interactive_config.load_first_providers_yml",
            lambda explicit_path=None: ({"providers": {}}, catalog.resolve()),
        )

        _HELPER._save_api_key_to_config("deepseek", "sk-test")

        data = yaml.safe_load(_user_providers_yml(tmp_path).read_text(encoding="utf-8"))
        assert data["providers"]["deepseek"]["api_key"] == "sk-test"
        assert data["default_provider"] == "deepseek"  # catalog content preserved

    def test_no_catalog_and_no_user_file_leaves_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ask_llm.utils.interactive_config.load_first_providers_yml",
            lambda explicit_path=None: (None, None),
        )
        _HELPER._save_api_key_to_config("deepseek", "sk-test")
        assert not _user_providers_yml(tmp_path).exists()

    def test_interrupted_write_leaves_prior_config_intact(self, tmp_path, monkeypatch):
        """A crash mid-save must not truncate the existing user file."""
        saved = _user_providers_yml(tmp_path)
        saved.parent.mkdir(parents=True)
        saved.write_text(
            yaml.safe_dump({"providers": {"deepseek": {"api_key": "sk-old"}}}),
            encoding="utf-8",
        )

        def exploding_chmod(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("ask_llm.core.checkpoint.os.chmod", exploding_chmod)
        _HELPER._save_api_key_to_config("deepseek", "sk-new")  # must not raise

        data = yaml.safe_load(saved.read_text(encoding="utf-8"))
        assert data["providers"]["deepseek"]["api_key"] == "sk-old"
        assert not list(saved.parent.glob("*.tmp"))


class TestShellEnvScrub:
    def test_shell_spawn_env_strips_provider_key(self, monkeypatch):
        """The !shell escape must not leak the interactively injected key."""

        class _Provider:
            name = "deepseek"
            default_model = "deepseek-chat"

        session = ChatSession(_Provider())
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
        env = session._scrubbed_shell_env()
        assert "DEEPSEEK_API_KEY" not in env
        # Other env is untouched.
        assert env.get("HOME") is not None
