"""providers.yml catalog loading (provider runtime config fallback).

``providers.yml`` carries the provider catalog (base URLs, models, pricing).
Only the runtime fields needed for API calls are extracted here; pricing/spec
fields are parsed separately by ``ask_llm.utils.pricing``.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
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
    """Return candidate paths for providers.yml (provider specs / pricing catalog).

    Order matters (first match wins): explicit env override, cwd, the repo root
    of a development checkout, the user's config directory, and finally the copy
    packaged inside the wheel. The packaged copy must stay last so a dev
    checkout and user overrides always shadow the (potentially stale) shipped
    catalog.

    Catalog readers (pricing, model limits) may safely consult cwd/repo copies —
    they only read reference data. Runtime provider config must use
    :func:`runtime_providers_yml_paths` instead.
    """
    paths: list[Path] = []
    env_path = os.getenv("ASK_LLM_PROVIDERS_YML")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.append(Path.cwd() / "providers.yml")
    # Dev checkout: .../ask_llm/config/providers_catalog.py -> repo root is 4
    # levels up. Only trust it when it actually looks like the repo; under a
    # wheel install this path lands in ``.../site-packages/..`` where the file
    # never exists, and the packaged copy below takes over instead.
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    if (pkg_root / "pyproject.toml").is_file() or (pkg_root / "providers.yml").is_file():
        paths.append(pkg_root / "providers.yml")
    paths.append(Path.home() / ".config" / "ask_llm" / "providers.yml")
    # Packaged copy (pyproject.toml package-data), used by pip installs.
    paths.append(Path(__file__).resolve().parent / "providers.yml")
    return paths


def runtime_providers_yml_paths() -> list[Path]:
    """Return candidate paths for providers.yml used as *runtime* provider config.

    The runtime merge sends the resolved ``api_key`` to the ``base_url`` found
    here, so cwd and repo-root copies are deliberately excluded: running
    ask-llm in a directory that happens to contain a ``providers.yml`` must not
    redirect credentials to that file's endpoints. Order (first match wins):
    explicit ``ASK_LLM_PROVIDERS_YML`` override, the user's config directory,
    then the packaged copy (which in a dev checkout is a symlink to the repo
    ``providers.yml``, preserving development behavior).
    """
    paths: list[Path] = []
    env_path = os.getenv("ASK_LLM_PROVIDERS_YML")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.append(Path.home() / ".config" / "ask_llm" / "providers.yml")
    # Packaged copy (pyproject.toml package-data for ask_llm.config).
    paths.append(Path(__file__).resolve().parent / "providers.yml")
    return paths


@lru_cache(maxsize=8)
def _load_yaml_cached(path_str: str, mtime_ns: int, size: int) -> dict[str, Any] | None:
    """Parse a providers.yml once per (path, mtime, size); ``None`` if unusable.

    The file is read and YAML-parsed here only; ``${VAR}`` resolution stays with
    the callers so env changes within a process are still honored. Loggers fire
    once per (mtime, size) instead of once per consumer read.
    """
    del mtime_ns, size  # cache-key only
    try:
        with open(path_str, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        logger.warning(f"Could not read providers.yml at {path_str}: {e}")
        return None
    except (yaml.YAMLError, TypeError, ValueError) as e:
        logger.warning(f"Invalid YAML in providers.yml at {path_str}: {e}")
        return None
    return data if isinstance(data, dict) else None


def _read_providers_yml(p: Path) -> dict[str, Any] | None:
    """Return a fresh deep copy of the parsed providers.yml at ``p``, or ``None``."""
    try:
        st = p.stat()
    except OSError:
        return None
    data = _load_yaml_cached(str(p), st.st_mtime_ns, st.st_size)
    return copy.deepcopy(data) if data else None


def load_first_providers_yml(
    explicit_path: str | Path | None = None,
    paths: list[Path] | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load the first parseable providers.yml with a non-empty ``providers`` mapping.

    Single entry point for reading providers.yml (runtime catalog, pricing,
    model limits). Resolves ``${VAR}`` environment placeholders. Parses are
    memoized per (path, mtime, size); each caller receives its own deep copy.

    Args:
        explicit_path: Optional path tried before the search list.
        paths: Search list override. Runtime config loading passes
            :func:`runtime_providers_yml_paths`; catalog readers keep the
            default (which includes cwd).

    Returns:
        ``(data, source_path)``; ``(None, None)`` when no usable file exists.
    """
    search: list[Path] = []
    if explicit_path:
        search.append(Path(explicit_path).expanduser())
    search.extend(paths if paths is not None else _candidate_providers_yml_paths())

    for p in search:
        if not p.is_file():
            continue
        data = _read_providers_yml(p)
        if not data:
            continue
        data = resolve_env_vars(data)
        providers = data.get("providers")
        if not providers or not isinstance(providers, dict):
            continue
        return data, p.resolve()
    return None, None


def _load_providers_yml() -> tuple[dict[str, Any], Path | None]:
    """
    Load provider runtime config (base_url/api_key/models) for API calls.

    Reads only :func:`runtime_providers_yml_paths` — cwd and repo-root copies
    are excluded so a stray ``providers.yml`` in the working directory cannot
    redirect resolved API keys to its own endpoints. Extracts fields needed for
    API calls; ignores pricing/spec fields (context_length, max_output,
    pricing_per_million_tokens, etc.).

    Returns:
        Tuple of (data, source_path). ``data`` has shape
        ``{"providers": {...}, "default_provider": ..., "default_model": ...}``;
        ``({}, None)`` when no providers.yml was found.
    """
    data, _source = load_first_providers_yml(paths=runtime_providers_yml_paths())
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
