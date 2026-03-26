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
