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


def load_first_providers_yml(
    explicit_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load the first parseable providers.yml with a non-empty ``providers`` mapping.

    Single entry point for reading providers.yml (runtime catalog, pricing,
    model limits). Resolves ``${VAR}`` environment placeholders.

    Returns:
        ``(data, source_path)``; ``(None, None)`` when no usable file exists.
    """
    paths: list[Path] = []
    if explicit_path:
        paths.append(Path(explicit_path).expanduser())
    paths.extend(_candidate_providers_yml_paths())

    for p in paths:
        if not p.is_file():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data or not isinstance(data, dict):
                continue
            data = resolve_env_vars(data)
            providers = data.get("providers")
            if not providers or not isinstance(providers, dict):
                continue
            return data, p.resolve()
        except OSError as e:
            logger.warning(f"Could not read providers.yml at {p}: {e}")
        except (yaml.YAMLError, TypeError, ValueError) as e:
            logger.warning(f"Invalid YAML in providers.yml at {p}: {e}")
    return None, None


def _load_providers_yml() -> tuple[dict[str, Any], Path | None]:
    """
    Load provider runtime config from the first existing providers.yml.

    Extracts fields needed for API calls: base_url, api_key, default_model, models,
    api_temperature, api_top_p, max_tokens, timeout. Ignores pricing/spec fields
    (context_length, max_output, pricing_per_million_tokens, etc.).

    Returns:
        Tuple of (data, source_path). ``data`` has shape
        ``{"providers": {...}, "default_provider": ..., "default_model": ...}``;
        ``({}, None)`` when no providers.yml was found.
    """
    data, _source = load_first_providers_yml()
    if data is None:
        return {}, None
    providers = data["providers"]

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
        return {}, None

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

    logger.debug(f"Loaded provider runtime config ({len(cleaned_providers)} providers)")
    return (
        {
            "providers": cleaned_providers,
            "default_provider": default_provider,
            "default_model": default_model,
        },
        _source,
    )
