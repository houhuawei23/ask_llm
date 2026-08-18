"""Load per-model limits (context, max_output) from providers.yml."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ask_llm.config.providers_catalog import load_first_providers_yml

# DeepSeek ``/chat/completions`` per-model ``max_tokens`` caps (HTTP API), applied after
# ``providers.yml``. Verified 2026-04: ``deepseek-chat`` [1,8192], ``deepseek-reasoner`` [1,65536].
_DEEPSEEK_API_MAX_TOKENS: dict[str, int] = {
    "deepseek-chat": 8192,
    "deepseek-reasoner": 65536,
}


def _deepseek_http_max_tokens_cap(model: str | None) -> int | None:
    """Return API ``max_tokens`` ceiling for known DeepSeek model ids; ``None`` if not DeepSeek."""
    if not model:
        return None
    key = model.strip().lower()
    if "deepseek" not in key:
        return None
    if key in _DEEPSEEK_API_MAX_TOKENS:
        return _DEEPSEEK_API_MAX_TOKENS[key]
    return 8192


@dataclass(frozen=True)
class ModelLimits:
    """Per-model limits from ``providers.yml``."""

    context_length: int
    max_output_default: int
    max_output_maximum: int


def load_providers_model_limits(
    explicit_path: str | Path | None = None,
) -> tuple[dict[str, ModelLimits], Path | None]:
    """
    Load ``context_length`` and ``max_output`` (default / maximum) for each model name.

    Model names are keyed by the API model id (e.g. ``deepseek-chat``). If the same name
    appears under multiple providers, the last occurrence wins (with a warning).

    Returns:
        (limits_by_model_name, path_used)
    """
    limits: dict[str, ModelLimits] = {}
    data, used = load_first_providers_yml(explicit_path)
    if data is None:
        return limits, None

    for prov_id, prov_cfg in data["providers"].items():
        if not isinstance(prov_cfg, dict):
            continue
        for m in prov_cfg.get("models") or []:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            if not name:
                continue
            name = str(name).strip()
            ctx = int(m.get("context_length") or 0)
            if ctx <= 0:
                ctx = 128_000
            mo = m.get("max_output")
            default_out, max_out = _parse_max_output(mo)
            if max_out < default_out:
                max_out = default_out
            if name in limits:
                logger.warning(
                    f"Duplicate model name {name!r} in providers.yml "
                    f"(provider {prov_id}); overwriting earlier entry"
                )
            limits[name] = ModelLimits(ctx, default_out, max_out)
    logger.debug(f"Loaded model limits from {used} ({len(limits)} model entries)")

    return limits, used


def _parse_max_output(mo: object) -> tuple[int, int]:
    if isinstance(mo, dict):
        d = int(mo.get("default") or 4096)
        mx = int(mo.get("maximum") or d)
        return d, mx
    return 4096, 4096


def resolve_paper_max_tokens(
    model: str | None,
    requested: int,
    limits_by_model: Mapping[str, ModelLimits] | None = None,
) -> int:
    """
    Effective ``max_tokens`` for a completion:

    - ``min(requested, max_output.maximum)`` when the model is listed in ``providers.yml``.
    - For DeepSeek models, ``min(..., API cap)``: ``deepseek-chat`` ≤ 8192, ``deepseek-reasoner`` ≤ 65536.
    """
    try:
        r = int(requested)
    except (TypeError, ValueError):
        r = 8192
    r = max(1, r)
    if model and limits_by_model:
        key = model.strip()
        if key in limits_by_model:
            r = min(r, limits_by_model[key].max_output_maximum)
    cap = _deepseek_http_max_tokens_cap(model)
    if cap is not None:
        r = min(r, cap)
    return r


def get_model_limits(
    model: str | None,
    limits_by_model: Mapping[str, ModelLimits] | None,
) -> ModelLimits | None:
    """Return limits for ``model`` if present in the catalog."""
    if not model or not limits_by_model:
        return None
    return limits_by_model.get(model.strip())
