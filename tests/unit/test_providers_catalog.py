"""Tests for providers.yml catalog discovery (search order, packaged fallback)."""

from pathlib import Path

import yaml

from ask_llm.config import providers_catalog
from ask_llm.config.providers_catalog import (
    _candidate_providers_yml_paths,
    _load_providers_yml,
    load_first_providers_yml,
    runtime_providers_yml_paths,
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


class TestRuntimeProviderPaths:
    """Audit 1.2: cwd providers.yml must never feed *runtime* provider config."""

    def test_runtime_paths_exclude_cwd_and_repo_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)
        paths = runtime_providers_yml_paths()
        assert (tmp_path / "providers.yml") not in paths
        assert Path.cwd() / "providers.yml" not in paths
        assert _PACKAGED_YML in paths

    def test_runtime_env_override_wins(self, tmp_path, monkeypatch):
        env_yml = tmp_path / "env" / "providers.yml"
        _write_providers_yml(env_yml)
        monkeypatch.setenv("ASK_LLM_PROVIDERS_YML", str(env_yml))
        assert runtime_providers_yml_paths()[0] == env_yml

    def test_cwd_providers_yml_not_merged_into_runtime_config(self, tmp_path, monkeypatch):
        """A hostile providers.yml in cwd must not be picked up by the runtime
        merge (resolved API keys must not be sent to its base_url)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        _write_providers_yml(tmp_path / "providers.yml")  # hostile cwd copy

        data, source = _load_providers_yml()
        assert source != (tmp_path / "providers.yml").resolve()
        # Dev checkout: the packaged copy (symlink to repo providers.yml) takes
        # over; a wheel layout without it yields (empty, None). Both are safe.
        assert source is None or source == _PACKAGED_YML.resolve()
        if source is None:
            assert data == {}

    def test_runtime_merge_uses_user_and_packaged_only(self, tmp_path, monkeypatch):
        """With no cwd copy, the runtime merge still finds the user config dir."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)
        home = tmp_path / "home"
        _write_providers_yml(home / ".config" / "ask_llm" / "providers.yml")
        monkeypatch.setenv("HOME", str(home))

        data, source = _load_providers_yml()
        assert source == (home / ".config" / "ask_llm" / "providers.yml").resolve()
        assert data["providers"]["prov_a"]["base_url"] == "https://a.example.com/v1"

    def test_catalog_reads_still_use_cwd(self, tmp_path, monkeypatch):
        """Pricing/model-limit catalog readers keep the cwd search path."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ASK_LLM_PROVIDERS_YML", raising=False)
        _write_providers_yml(tmp_path / "providers.yml")
        _data, source = load_first_providers_yml()
        assert source == (tmp_path / "providers.yml").resolve()


class TestMemoizedParse:
    def test_parsed_once_per_process(self, tmp_path):
        yml = tmp_path / "providers.yml"
        _write_providers_yml(yml)
        providers_catalog._load_yaml_cached.cache_clear()

        _data1, _ = load_first_providers_yml(explicit_path=yml)
        misses_after_first = providers_catalog._load_yaml_cached.cache_info().misses
        assert misses_after_first == 1

        _data2, _ = load_first_providers_yml(explicit_path=yml)
        assert providers_catalog._load_yaml_cached.cache_info().misses == 1  # no re-parse

    def test_cache_invalidated_on_file_change(self, tmp_path):
        yml = tmp_path / "providers.yml"
        _write_providers_yml(yml)
        providers_catalog._load_yaml_cached.cache_clear()
        data1, _ = load_first_providers_yml(explicit_path=yml)
        # Rewrite with different content (mtime/size change busts the cache).
        yml.write_text(
            yaml.safe_dump(
                {
                    "providers": {
                        "prov_b": {
                            "base_url": "https://b.example.com/v1",
                            "models": ["model-b1"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        data2, _ = load_first_providers_yml(explicit_path=yml)
        assert "prov_a" in data1["providers"]
        assert "prov_b" in data2["providers"]

    def test_env_resolution_stays_fresh(self, tmp_path, monkeypatch):
        """Only the raw parse is memoized; ${VAR} must resolve per call."""
        yml = tmp_path / "providers.yml"
        _write_providers_yml(yml)
        providers_catalog._load_yaml_cached.cache_clear()
        monkeypatch.setenv("PROV_A_KEY", "key-one")
        data1, _ = load_first_providers_yml(explicit_path=yml)
        assert data1["providers"]["prov_a"]["api_key"] == "key-one"
        monkeypatch.setenv("PROV_A_KEY", "key-two")
        data2, _ = load_first_providers_yml(explicit_path=yml)
        assert data2["providers"]["prov_a"]["api_key"] == "key-two"

    def test_callers_receive_independent_copies(self, tmp_path):
        yml = tmp_path / "providers.yml"
        _write_providers_yml(yml)
        providers_catalog._load_yaml_cached.cache_clear()
        data1, _ = load_first_providers_yml(explicit_path=yml)
        data1["providers"]["prov_a"]["base_url"] = "https://mutated.example.com"
        data2, _ = load_first_providers_yml(explicit_path=yml)
        assert data2["providers"]["prov_a"]["base_url"] == "https://a.example.com/v1"
