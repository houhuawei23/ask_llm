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


class TestAudit53SetConfigValue:
    """Plan 5.3: comment-preserving config set, atomic + mode honored."""

    @staticmethod
    def _write(tmp_path: Path, content: str) -> Path:
        p = tmp_path / "providers.yml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_set_preserves_comments_and_other_providers(self, tmp_path):
        from ask_llm.utils.interactive_config import set_config_value

        original = (
            "# my providers\n"
            "providers:\n"
            "  deepseek:\n"
            "    api_key: ${DEEPSEEK_API_KEY}  # keep this comment\n"
            "    api_base: https://api.deepseek.com/v1\n"
            "  openai:\n"
            "    # inline note above the key\n"
            "    api_key: sk-openai\n"
        )
        path = self._write(tmp_path, original)

        preserved = set_config_value(path, "providers.deepseek.api_key", "sk-new")

        assert preserved is True
        text = path.read_text(encoding="utf-8")
        assert "# my providers" in text
        assert "# keep this comment" in text
        assert "# inline note above the key" in text
        assert "sk-new" in text
        assert "sk-openai" in text
        assert "api_base: https://api.deepseek.com/v1" in text

    def test_set_quotes_numeric_secret_as_string(self, tmp_path):
        import yaml

        from ask_llm.utils.interactive_config import set_config_value

        path = self._write(tmp_path, "providers:\n  x:\n    api_key: abc\n")
        set_config_value(path, "providers.x.api_key", "123456")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["providers"]["x"]["api_key"] == "123456"
        assert "'123456'" in path.read_text(encoding="utf-8")

    def test_set_parses_scalars_in_place(self, tmp_path):
        import yaml

        from ask_llm.utils.interactive_config import set_config_value

        path = self._write(tmp_path, "translation:\n  max_chunk_tokens: 2000  # budget\n")
        set_config_value(path, "translation.max_chunk_tokens", "3000")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["translation"]["max_chunk_tokens"] == 3000
        assert "# budget" in path.read_text(encoding="utf-8")

    def test_missing_key_falls_back_to_dump_rewrite_with_mode(self, tmp_path):
        import yaml

        from ask_llm.utils.interactive_config import set_config_value

        path = self._write(tmp_path, "# header comment\nproviders:\n  x:\n    api_key: a\n")
        preserved = set_config_value(path, "providers.new.api_key", "sk-n", mode=0o600)

        assert preserved is False
        assert (path.stat().st_mode & 0o777) == 0o600
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["providers"]["new"]["api_key"] == "sk-n"
        assert data["providers"]["x"]["api_key"] == "a"

    def test_save_api_key_preserves_comments(self, tmp_path, monkeypatch):
        """_save_api_key_to_config delegates to the comment-preserving setter."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        target = tmp_path / ".config" / "ask_llm" / "providers.yml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "# comment survives\nproviders:\n  deepseek:\n    api_key: old\n",
            encoding="utf-8",
        )

        _HELPER._save_api_key_to_config("deepseek", "sk-rotated")

        text = target.read_text(encoding="utf-8")
        assert "sk-rotated" in text
        assert "old" not in text
        assert "# comment survives" in text
        assert (target.stat().st_mode & 0o777) == 0o600
