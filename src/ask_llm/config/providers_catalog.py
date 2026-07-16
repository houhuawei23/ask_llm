"""providers.yml catalog loading (provider runtime config fallback).

``providers.yml`` carries the provider catalog (base URLs, models, pricing).
Only the runtime fields needed for API calls are extracted here; pricing/spec
fields are parsed separately by ``ask_llm.utils.pricing``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from ask_llm.config.env import resolve_env_vars

# Fields needed for API calls; pricing/spec fields (context_length, max_output,
# pricing_per_million_tokens, etc.) are intentionally ignored here.
_RUNTIME_FIELDS = {
    "base_url",
    "api_key",
    "default_model",
    "models",
    "api_temperature",
    "api_top_p",
    "max_tokens",
    "timeout",
}


def _candidate_providers_yml_paths() -> list[Path]:
    """Return candidate paths for providers.yml (provider specs / pricing catalog)."""
    paths: list[Path] = []
    env_path = os.getenv("ASK_LLM_PROVIDERS_YML")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.append(Path.cwd() / "providers.yml")
    # Package: .../ask_llm/config/providers_catalog.py -> ask_llm repo root often 3 levels up
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    paths.append(pkg_root / "providers.yml")
    paths.append(Path.home() / ".config" / "ask_llm" / "providers.yml")
    return paths


def _load_providers_yml() -> dict[str, Any]:
    """
    Load provider runtime config from the first existing providers.yml.

    Extracts fields needed for API calls: base_url, api_key, default_model, models,
    api_temperature, api_top_p, max_tokens, timeout. Ignores pricing/spec fields
    (context_length, max_output, pricing_per_million_tokens, etc.).

    Returns:
        Dict with shape {"providers": {...}, "default_provider": ..., "default_model": ...}
        or empty dict if no providers.yml found.
    """
    for p in _candidate_providers_yml_paths():
        if not p.is_file():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data or not isinstance(data, dict):
                continue
            data = resolve_env_vars(data)
            providers = data.get("providers") or {}
            if not providers:
                continue

            cleaned_providers: dict[str, Any] = {}
            for prov_id, prov_cfg in providers.items():
                if not isinstance(prov_cfg, dict):
                    continue
                cleaned = {k: v for k, v in prov_cfg.items() if k in _RUNTIME_FIELDS}
                # Normalize models list: extract "name" from dict entries
                models = cleaned.get("models")
                if isinstance(models, list):
                    model_names = []
                    for m in models:
                        if isinstance(m, dict):
                            name = m.get("name")
                            if name:
                                model_names.append(name)
                        elif isinstance(m, str):
                            model_names.append(m)
                    cleaned["models"] = model_names
                if cleaned.get("base_url"):
                    cleaned_providers[prov_id] = cleaned

            if not cleaned_providers:
                continue

            # Determine default_provider / default_model from providers.yml
            default_provider = data.get("default_provider")
            default_model = data.get("default_model")
            if not default_provider:
                default_provider = next(iter(cleaned_providers.keys()))
            if not default_model:
                first_cfg = cleaned_providers[default_provider]
                default_model = first_cfg.get("default_model")
                if not default_model and first_cfg.get("models"):
                    default_model = first_cfg["models"][0]

            logger.debug(
                f"Loaded provider runtime config from {p.resolve()} "
                f"({len(cleaned_providers)} providers)"
            )
            return {
                "providers": cleaned_providers,
                "default_provider": default_provider,
                "default_model": default_model,
            }
        except OSError as e:
            logger.warning(f"Could not read providers.yml at {p}: {e}")
        except (yaml.YAMLError, TypeError, ValueError) as e:
            logger.warning(f"Invalid YAML in providers.yml at {p}: {e}")

    return {}
