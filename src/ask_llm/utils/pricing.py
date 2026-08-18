"""Load per-model pricing from providers.yml and estimate API cost."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ask_llm.config.providers_catalog import load_first_providers_yml


def load_providers_pricing(
    explicit_path: str | Path | None = None,
) -> tuple[dict[tuple[str, str], dict[str, float]], Path | None]:
    """
    Load pricing_per_million_tokens from the first existing providers.yml.

    Returns:
        (pricing_map, path_used) where pricing_map maps (provider_id, model_name) to
        {"input", "output", "input_cache_hit"} in CNY per 1M tokens.
    """
    pricing: dict[tuple[str, str], dict[str, float]] = {}
    data, used = load_first_providers_yml(explicit_path)
    if data is None:
        return pricing, None

    for prov_id, prov_cfg in data["providers"].items():
        if not isinstance(prov_cfg, dict):
            continue
        for m in prov_cfg.get("models") or []:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            if not name:
                continue
            ppm = m.get("pricing_per_million_tokens")
            if not isinstance(ppm, dict):
                continue
            inp = float(ppm.get("input", 0) or 0)
            out = float(ppm.get("output", 0) or 0)
            hit = float(ppm.get("input_cache_hit", 0) or 0)
            pricing[(str(prov_id), str(name))] = {
                "input": inp,
                "output": out,
                "input_cache_hit": hit,
            }
    logger.debug(f"Loaded API pricing from {used} ({len(pricing)} model entries)")
    return pricing, used


def estimate_cost_cny(
    pricing_row: dict[str, float],
    input_tokens: int,
    output_tokens: int,
    *,
    input_cache_hit_tokens: int = 0,
) -> float:
    """
    Estimate cost in CNY from per-million token prices.

    input_cache_hit_tokens: portion of input billed at cache-hit rate (rest at input rate).
    """
    inp_rate = pricing_row.get("input", 0.0)
    out_rate = pricing_row.get("output", 0.0)
    hit_rate = pricing_row.get("input_cache_hit", 0.0)

    inp = max(0, int(input_tokens))
    out = max(0, int(output_tokens))
    hit = max(0, min(int(input_cache_hit_tokens), inp))
    miss = inp - hit

    cost = (
        (miss / 1_000_000.0) * inp_rate
        + (hit / 1_000_000.0) * hit_rate
        + (out / 1_000_000.0) * out_rate
    )
    return cost


def lookup_pricing(
    pricing_map: dict[tuple[str, str], dict[str, float]],
    provider: str,
    model: str,
) -> dict[str, float] | None:
    return pricing_map.get((provider, model))


def format_cost_estimate(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing_map: dict[tuple[str, str], dict[str, float]],
    *,
    pricing_source: Path | None = None,
) -> str:
    """Human-readable lines for console (no leading/trailing newlines)."""
    total = input_tokens + output_tokens
    lines = [
        f"  Tokens: input={input_tokens:,}  output={output_tokens:,}  total={total:,}",
    ]
    row = lookup_pricing(pricing_map, provider, model)
    src = f" ({pricing_source.name})" if pricing_source else ""
    if row is not None:
        cny = estimate_cost_cny(row, input_tokens, output_tokens)
        lines.append(
            f"  Estimated cost (CNY){src}: ¥{cny:.4f}  (pricing: ¥{row['input']:.2f}/M in, ¥{row['output']:.2f}/M out)"
        )
    else:
        lines.append(
            f"  Estimated cost: unavailable — no pricing_per_million_tokens for "
            f"{provider}/{model} in providers.yml{src or ''}"
        )
    return "\n".join(lines)
