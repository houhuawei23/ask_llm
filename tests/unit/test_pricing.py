"""Tests for providers.yml pricing helpers."""

from pathlib import Path

import pytest

from ask_llm.utils.pricing import estimate_cost_cny, load_providers_pricing, lookup_pricing


def test_estimate_cost_cny_deepseek_rates() -> None:
    row = {"input": 2.0, "output": 3.0, "input_cache_hit": 0.2}
    # 1M in + 1M out = 2 + 3 = 5 CNY
    assert estimate_cost_cny(row, 1_000_000, 1_000_000) == pytest.approx(5.0)
    # 0 tokens
    assert estimate_cost_cny(row, 0, 0) == 0.0


def test_load_providers_pricing_explicit(tmp_path: Path) -> None:
    p = tmp_path / "providers.yml"
    p.write_text(
        """
providers:
  deepseek:
    base_url: "https://example.com"
    api_key: "x"
    models:
      - name: deepseek-chat
        pricing_per_million_tokens:
          input: 2
          output: 3
""",
        encoding="utf-8",
    )
    m, used = load_providers_pricing(p)
    assert used == p.resolve()
    row = lookup_pricing(m, "deepseek", "deepseek-chat")
    assert row is not None
    assert row["input"] == 2.0
    assert row["output"] == 3.0


class TestAudit44CacheHitPricing:
    """Audit 4.4: cache-hit tokens price at the hit rate, not the input rate."""

    def test_format_cost_estimate_forwards_cache_hit_tokens(self):
        from ask_llm.utils.pricing import format_cost_estimate

        pricing_map = {
            ("deepseek", "deepseek-chat"): {
                "input": 2.0,
                "output": 8.0,
                "input_cache_hit": 0.2,
            }
        }

        full_rate = format_cost_estimate("deepseek", "deepseek-chat", 1_000_000, 0, pricing_map)
        hit_rate = format_cost_estimate(
            "deepseek",
            "deepseek-chat",
            1_000_000,
            0,
            pricing_map,
            input_cache_hit_tokens=1_000_000,
        )

        assert "¥2.0000" in full_rate
        assert "¥0.2000" in hit_rate
        # The pricing note surfaces the cache-hit rate.
        assert "0.20/M cached-in" in hit_rate

    def test_estimate_cost_cny_hit_tokens_capped_at_input(self):
        from ask_llm.utils.pricing import estimate_cost_cny

        row = {"input": 2.0, "output": 8.0, "input_cache_hit": 0.2}
        # Hit portion can't exceed input; 1M input with 1.2M "hit" clamps to 1M.
        cost = estimate_cost_cny(row, 1_000_000, 0, input_cache_hit_tokens=1_200_000)
        assert abs(cost - 0.2) < 1e-9
