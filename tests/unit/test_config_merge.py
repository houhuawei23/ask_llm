"""Unit tests for ask_llm.config.merge (_deep_merge and record_leaves)."""

from __future__ import annotations

from ask_llm.config.merge import _deep_merge, record_leaves


class TestDeepMerge:
    def test_nested_dicts_merge_and_scalars_override(self):
        base = {
            "default_provider": "openai",
            "translation": {"target_language": "en", "retries": 1},
        }
        overlay = {"default_provider": "deepseek", "translation": {"retries": 3}}

        result = _deep_merge(base, overlay)

        assert result["default_provider"] == "deepseek"
        # Nested dict: overlay key wins, sibling base keys survive.
        assert result["translation"] == {"target_language": "en", "retries": 3}

    def test_providers_merge_per_provider_not_wholesale(self):
        base = {
            "providers": {
                "openai": {"base_url": "https://api.openai.com/v1", "api_key": "old"},
                "deepseek": {"base_url": "https://api.deepseek.com", "api_key": "ds"},
            }
        }
        overlay = {
            "providers": {
                "openai": {"api_key": "new", "models": ["gpt-4o"]},
                "moonshot": {"base_url": "https://api.moonshot.cn/v1"},
            }
        }

        result = _deep_merge(base, overlay)

        # Existing provider: deep-merged (base_url kept, api_key overridden).
        assert result["providers"]["openai"] == {
            "base_url": "https://api.openai.com/v1",
            "api_key": "new",
            "models": ["gpt-4o"],
        }
        # Untouched provider survives the merge.
        assert result["providers"]["deepseek"]["api_key"] == "ds"
        # New provider is added.
        assert result["providers"]["moonshot"] == {"base_url": "https://api.moonshot.cn/v1"}

    def test_explicitly_empty_providers_clears_everything(self):
        base = {"providers": {"openai": {"api_key": "k"}}, "default_provider": "openai"}

        result = _deep_merge(base, {"providers": {}})

        assert result["providers"] == {}
        assert result["default_provider"] == "openai"

    def test_non_dict_overlay_value_replaces_base_value(self):
        base = {"translation": {"retries": 1}, "providers": {"openai": {"api_key": "k"}}}

        # A scalar / list overlay replaces a dict base value wholesale.
        result = _deep_merge(base, {"translation": 5, "providers": ["flat"]})

        assert result["translation"] == 5
        assert result["providers"] == ["flat"]

    def test_merge_does_not_mutate_its_inputs(self):
        base = {"translation": {"retries": 1}}
        overlay = {"translation": {"retries": 2}}

        _deep_merge(base, overlay)

        assert base == {"translation": {"retries": 1}}
        assert overlay == {"translation": {"retries": 2}}


class TestRecordLeaves:
    def test_records_dotted_leaf_paths_with_source(self):
        provenance: dict[str, str] = {}
        record_leaves(
            {"translation": {"retries": 3, "style": "plain"}, "models": ["m1"], "flag": True},
            "user",
            provenance,
        )

        assert provenance == {
            "translation.retries": "user",
            "translation.style": "user",
            "models": "user",  # a list counts as one leaf
            "flag": "user",
        }

    def test_empty_dict_and_empty_containers_are_leaves(self):
        provenance: dict[str, str] = {}
        record_leaves({"providers": {}, "translation": {}}, "default", provenance)

        # An empty dict is recorded as a leaf (nothing inside it to recurse into).
        assert provenance == {"providers": "default", "translation": "default"}

    def test_later_call_overwrites_earlier_source_labels(self):
        provenance: dict[str, str] = {}
        record_leaves({"a": 1, "nested": {"b": 2}}, "default", provenance)
        record_leaves({"a": 9, "nested": {"c": 3}}, "user", provenance)

        assert provenance == {
            "a": "user",  # overwritten by the higher-precedence layer
            "nested.b": "default",  # untouched by the later layer
            "nested.c": "user",
        }

    def test_toplevel_scalar_is_not_recorded(self):
        provenance: dict[str, str] = {}
        record_leaves(42, "default", provenance)

        assert provenance == {}
