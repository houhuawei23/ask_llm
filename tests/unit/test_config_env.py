"""Unit tests for ask_llm.config.env.

Covers ``${VAR}`` resolution in YAML values and ``ASK_LLM_*`` environment
overrides (ENV_TO_CONFIG mapping, type coercion, provenance recording).
"""

from __future__ import annotations

import pytest

from ask_llm.config.env import (
    ENV_TO_CONFIG,
    _apply_env_overrides,
    resolve_env_vars,
)

# Every env var the module reads, plus the deprecated one it only warns about.
_ALL_ENV_VARS = [*ENV_TO_CONFIG, "ASK_LLM_TRANSLATION_MAX_CHUNK_SIZE"]


@pytest.fixture(autouse=True)
def _clean_ask_llm_env(monkeypatch):
    """Remove all ASK_LLM_* variables so the host environment cannot leak in."""
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestResolveEnvVars:
    def test_substitutes_set_var_in_nested_structures(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-abc123")
        monkeypatch.setenv("MY_BASE", "https://api.example.com/v1")

        data = {
            "providers": {
                "p1": {
                    "api_key": "${MY_KEY}",
                    "api_base": "${MY_BASE}",
                    "note": "prefix-${MY_KEY}-suffix",
                }
            },
            "keys": ["${MY_KEY}", "literal"],
            "plain": "no references here",
        }
        result = resolve_env_vars(data)

        assert result["providers"]["p1"]["api_key"] == "sk-abc123"
        assert result["providers"]["p1"]["api_base"] == "https://api.example.com/v1"
        # Substitution works inside a larger string (verbatim replacement).
        assert result["providers"]["p1"]["note"] == "prefix-sk-abc123-suffix"
        assert result["keys"] == ["sk-abc123", "literal"]
        assert result["plain"] == "no references here"

    def test_keeps_placeholder_when_var_unset_or_empty(self, monkeypatch):
        monkeypatch.delenv("MY_MISSING", raising=False)
        monkeypatch.setenv("MY_EMPTY", "")

        assert resolve_env_vars("${MY_MISSING}") == "${MY_MISSING}"
        # An empty-but-set value is falsy, so the placeholder survives too.
        assert resolve_env_vars("${MY_EMPTY}") == "${MY_EMPTY}"
        # Nested value in a structure: unresolved reference stays literal.
        assert resolve_env_vars({"k": ["${MY_MISSING}"]}) == {"k": ["${MY_MISSING}"]}

    def test_passthrough_non_strings_and_plain_strings(self):
        # Non-str scalars are returned unchanged; strings without references
        # (including ':'/'=') come back untouched.
        assert resolve_env_vars(42) == 42
        assert resolve_env_vars(None) is None
        assert resolve_env_vars(True) is True
        assert resolve_env_vars("a:b=c$d") == "a:b=c$d"
        assert resolve_env_vars(":{}= ") == ":{}= "


class TestApplyEnvOverrides:
    def test_maps_env_vars_onto_config_with_type_coercion(self, monkeypatch):
        base = {"translation": {"retries": 1}, "batch": {}}
        monkeypatch.setenv("ASK_LLM_DEFAULT_PROVIDER", "openai:prod")  # colon kept
        monkeypatch.setenv("ASK_LLM_TRANSLATION_TARGET_LANGUAGE", "fr")
        monkeypatch.setenv("ASK_LLM_TRANSLATION_THREADS", "4")  # int
        monkeypatch.setenv("ASK_LLM_TRANSLATION_TEMPERATURE", "0.15")  # float
        monkeypatch.setenv("ASK_LLM_TRANSLATION_PRESERVE_FORMAT", "yes")  # bool
        monkeypatch.setenv("ASK_LLM_BATCH_RETRY_DELAY", "2.5")  # float

        provenance: dict[str, str] = {}
        result = _apply_env_overrides(base, provenance)

        assert result["default_provider"] == "openai:prod"
        assert result["translation"]["target_language"] == "fr"
        assert result["translation"]["max_concurrent_api_calls"] == 4
        assert result["translation"]["temperature"] == 0.15
        assert result["translation"]["preserve_format"] is True
        assert result["batch"]["retry_delay"] == 2.5
        # Existing base values survive; the input dict is not mutated.
        assert result["translation"]["retries"] == 1
        assert base["translation"] == {"retries": 1}
        # Provenance records the winning env var per dotted config path.
        assert provenance["default_provider"] == "env:ASK_LLM_DEFAULT_PROVIDER"
        assert (
            provenance["translation.max_concurrent_api_calls"] == "env:ASK_LLM_TRANSLATION_THREADS"
        )

    def test_invalid_null_and_empty_values(self, monkeypatch):
        base = {"translation": {"max_chunk_tokens": 99, "style": "plain"}}
        monkeypatch.setenv("ASK_LLM_TRANSLATION_MAX_CHUNK_TOKENS", "abc")  # bad int
        monkeypatch.setenv("ASK_LLM_TRANSLATION_RETRIES", "null")  # explicit None
        monkeypatch.setenv("ASK_LLM_TRANSLATION_STYLE", "")  # empty: ignored

        result = _apply_env_overrides(base)

        # Invalid int: warning logged, base value untouched.
        assert result["translation"]["max_chunk_tokens"] == 99
        # "null"/"none" parse to an explicit None override.
        assert result["translation"]["retries"] is None
        # Empty string is treated as "not set".
        assert result["translation"]["style"] == "plain"

    def test_conflicting_targets_last_var_wins_and_legacy_is_ignored(self, monkeypatch):
        base = {"translation": {"max_concurrent_api_calls": 2, "max_chunk_tokens": 10}}
        # THREADS and MAX_CONCURRENT_API_CALLS both target the same key; the
        # last entry in ENV_TO_CONFIG order wins.
        monkeypatch.setenv("ASK_LLM_TRANSLATION_THREADS", "3")
        monkeypatch.setenv("ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS", "7")
        # Deprecated var: warned about, never applied.
        monkeypatch.setenv("ASK_LLM_TRANSLATION_MAX_CHUNK_SIZE", "5000")

        result = _apply_env_overrides(base)

        assert result["translation"]["max_concurrent_api_calls"] == 7
        assert result["translation"]["max_chunk_tokens"] == 10

    def test_missing_section_is_created(self, monkeypatch):
        monkeypatch.setenv("ASK_LLM_TRANSLATION_RECURSIVE_DIR", "true")

        result = _apply_env_overrides({})

        assert result == {"translation": {"recursive_dir": True}}
