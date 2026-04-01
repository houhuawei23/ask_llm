"""Load per-model limits (context, max_output) from providers.yml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import yaml
from loguru import logger

from ask_llm.config.loader import resolve_env_vars
from ask_llm.utils.pricing import _candidate_providers_yml_paths

# DeepSeek ``/chat/completions`` per-model ``max_tokens`` caps (HTTP API), applied after
# ``providers.yml``. Verified 2026-04: ``deepseek-chat`` [1,8192], ``deepseek-reasoner`` [1,65536].
_DEEPSEEK_API_MAX_TOKENS: dict[str, int] = {
    "deepseek-chat": 8192,
    "deepseek-reasoner": 65536,
}


def _deepseek_http_max_tokens_cap(model: Optional[str]) -> Optional[int]:
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
    used: Path | None = None
    for p in _candidate_providers_yml_paths(explicit_path):
        if not p.is_file():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            data = resolve_env_vars(data)
            providers = data.get("providers") or {}
            for prov_id, prov_cfg in providers.items():
                if not isinstance(prov_cfg, dict):
                    continue
                models = prov_cfg.get("models") or []
                for m in models:
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
            used = p.resolve()
            logger.debug(
                f"Loaded model limits from {used} ({len(limits)} model entries)"
            )
            break
        except OSError as e:
            logger.warning(f"Could not read providers.yml at {p}: {e}")
        except (yaml.YAMLError, TypeError, ValueError) as e:
            logger.warning(f"Invalid YAML or model specs in {p}: {e}")

    return limits, used


def _parse_max_output(mo: object) -> tuple[int, int]:
    if isinstance(mo, dict):
        d = int(mo.get("default") or 4096)
        mx = int(mo.get("maximum") or d)
        return d, mx
    return 4096, 4096


def resolve_paper_max_tokens(
    model: Optional[str],
    requested: int,
    limits_by_model: Optional[Mapping[str, ModelLimits]] = None,
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
    model: Optional[str],
    limits_by_model: Optional[Mapping[str, ModelLimits]],
) -> Optional[ModelLimits]:
    """Return limits for ``model`` if present in the catalog."""
    if not model or not limits_by_model:
        return None
    return limits_by_model.get(model.strip())
