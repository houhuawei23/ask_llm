"""Tests for providers.yml catalog discovery (search order, packaged fallback)."""

from pathlib import Path

import yaml

from ask_llm.config import providers_catalog
from ask_llm.config.providers_catalog import (
    _candidate_providers_yml_paths,
    load_first_providers_yml,
)

_PACKAGED_YML = Path(providers_catalog.__file__).resolve().parent / "providers.yml"


def _write_providers_yml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "prov_a": {
                        "base_url": "https://a.example.com/v1",
                        "api_key": "${PROV_A_KEY}",
                        "models": [{"name": "model-a1"}],
                    }
                },
                "default_provider": "prov_a",
            }
        ),
        encoding="utf-8",
    )


class TestCandidatePaths:
    def test_packaged_copy_is_a_candidate(self):
        """K3 regression: the wheel-packaged providers.yml must be on the search
        path so pip installs keep provider fallback + pricing working."""
        assert _PACKAGED_YML.is_file()
        assert _PACKAGED_YML in _candidate_providers_yml_paths()

    def test_dev_repo_root_guarded_by_marker(self, tmp_path, monkeypatch):
        """A site-packages-like parent (no pyproject.toml/providers.yml) must not
        be treated as the repo root."""
        fake_module = (
            tmp_path
            / "venv"
            / "lib"
            / "py3"
            / "site-packages"
            / "ask_llm"
            / ("config")
            / "providers_catalog.py"
        )
        fake_module.parent.mkdir(parents=True)
        fake_module.write_text("", encoding="utf-8")
        monkeypatch.setattr(providers_catalog, "__file__", str(fake_module))
        candidates = _candidate_providers_yml_paths()
        assert (tmp_path / "venv" / "providers.yml") not in candidates


class TestLoadFirstProvidersYml:
    def test_env_var_wins(self, tmp_path, monkeypatch):
        env_yml = tmp_path / "env" / "providers.yml"
        _write_providers_yml(env_yml)
        monkeypatch.setenv("ASK_LLM_PROVIDERS_YML", str(env_yml))
        _data, source = load_first_providers_yml()
        assert source == env_yml.resolve()

    def test_cwd_wins_over_packaged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)
        _write_providers_yml(tmp_path / "providers.yml")
        _data, source = load_first_providers_yml()
        assert source == (tmp_path / "providers.yml").resolve()

    def test_packaged_copy_found_in_wheel_layout(self, tmp_path, monkeypatch):
        """K3 end-to-end: with a wheel-like layout (module inside site-packages,
        no repo root above it), the packaged catalog is discovered."""
        site_packages = tmp_path / "venv" / "lib" / "py3" / "site-packages"
        pkg_config_dir = site_packages / "ask_llm" / "config"
        pkg_config_dir.mkdir(parents=True)
        fake_module = pkg_config_dir / "providers_catalog.py"
        fake_module.write_text("", encoding="utf-8")
        packaged = pkg_config_dir / "providers.yml"
        _write_providers_yml(packaged)
        monkeypatch.setattr(providers_catalog, "__file__", str(fake_module))
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        _data, source = load_first_providers_yml()
        assert source == packaged.resolve()
        assert "providers" in _data

    def test_returns_none_when_nothing_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)
        monkeypatch.setattr(
            providers_catalog,
            "_candidate_providers_yml_paths",
            lambda: [tmp_path / "nowhere" / "providers.yml"],
        )
        data, source = load_first_providers_yml()
        assert data is None and source is None
