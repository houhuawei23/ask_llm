"""Environment-variable configuration layer.

Two responsibilities:
  1. ``resolve_env_vars`` — expand ``${VAR}`` references inside YAML values.
  2. ``apply_env_overrides`` — overlay ``ASK_LLM_*`` variables onto config data.

Parameter priority: CLI args > environment variables > user config > package default.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any

from loguru import logger

# Log each missing ${VAR} at most once per process (avoid duplicate warnings).
_WARNED_UNSET_ENV_VARS: set[str] = set()

# Environment variable to config path mapping. Env vars overlay user config.
# Format: "ASK_LLM_<SECTION>_<KEY>" -> ("section", "key") or "ASK_LLM_<KEY>" -> ("key",)
ENV_TO_CONFIG: dict[str, tuple[str, ...]] = {
    "ASK_LLM_DEFAULT_PROVIDER": ("default_provider",),
    "ASK_LLM_DEFAULT_MODEL": ("default_model",),
    "ASK_LLM_TRANSLATION_TARGET_LANGUAGE": ("translation", "target_language"),
    "ASK_LLM_TRANSLATION_SOURCE_LANGUAGE": ("translation", "source_language"),
    "ASK_LLM_TRANSLATION_STYLE": ("translation", "style"),
    # ASK_LLM_TRANSLATION_THREADS controls the per-file chunk concurrency used by trans.
    # It maps to the active field max_concurrent_api_calls; "threads" is a legacy alias.
    "ASK_LLM_TRANSLATION_THREADS": ("translation", "max_concurrent_api_calls"),
    "ASK_LLM_TRANSLATION_MAX_PARALLEL_FILES": ("translation", "max_parallel_files"),
    "ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS": (
        "translation",
        "max_concurrent_api_calls",
    ),
    "ASK_LLM_TRANSLATION_RETRIES": ("translation", "retries"),
    "ASK_LLM_TRANSLATION_BALANCE_CHUNK_TOKENS": ("translation", "balance_translation_chunks"),
    "ASK_LLM_TRANSLATION_MAX_CHUNK_TOKENS": ("translation", "max_chunk_tokens"),
    "ASK_LLM_TRANSLATION_MAX_OUTPUT_TOKENS": ("translation", "max_output_tokens"),
    "ASK_LLM_TRANSLATION_MIN_CHUNK_MERGE_TOKENS": ("translation", "min_chunk_merge_tokens"),
    "ASK_LLM_TRANSLATION_PRESERVE_FORMAT": ("translation", "preserve_format"),
    "ASK_LLM_TRANSLATION_INCLUDE_ORIGINAL": ("translation", "include_original"),
    "ASK_LLM_TRANSLATION_TEMPERATURE": ("translation", "temperature"),
    "ASK_LLM_TRANSLATION_DEFAULT_PROMPT_FILE": ("translation", "default_prompt_file"),
    "ASK_LLM_TRANSLATION_RECURSIVE_DIR": ("translation", "recursive_dir"),
    "ASK_LLM_BATCH_THREADS": ("batch", "threads"),
    "ASK_LLM_BATCH_RETRIES": ("batch", "retries"),
    "ASK_LLM_BATCH_RETRY_DELAY": ("batch", "retry_delay"),
    "ASK_LLM_BATCH_RETRY_DELAY_MAX": ("batch", "retry_delay_max"),
}


def _parse_env_value(value: str, key_path: tuple[str, ...]) -> Any:
    """Parse env var string to appropriate type for the config key."""
    if value.lower() in ("null", "none", ""):
        return None
    last_key = key_path[-1].lower()
    if (
        "threads" in last_key
        or "retries" in last_key
        or "max_chunk_size" in last_key
        or "max_chunk_tokens" in last_key
        or "max_output_tokens" in last_key
        or "min_chunk_merge_tokens" in last_key
    ):
        return int(value)
    if "retry_delay" in last_key or "temperature" in last_key:
        return float(value)
    if (
        "preserve_format" in last_key
        or "include_original" in last_key
        or "recursive" in last_key
        or "balance_translation_chunks" in last_key
    ):
        return value.lower() in ("true", "1", "yes")
    return value


def _duplicate_env_targets() -> dict[tuple[str, ...], list[str]]:
    """Config keys targeted by more than one env var (P2.7).

    Returns ``{config_key_path: [env_var_names in ENV_TO_CONFIG order]}`` for
    keys with two or more env vars. The apply loop overwrites in iteration
    order, so only the last such env var wins; the others are silently ignored.
    """
    targets: dict[tuple[str, ...], list[str]] = {}
    for env_var, key_path in ENV_TO_CONFIG.items():
        targets.setdefault(key_path, []).append(env_var)
    return {key: vars_ for key, vars_ in targets.items() if len(vars_) > 1}


def _warn_conflicting_env_overrides() -> None:
    """Warn when multiple SET env vars target the same config key (P2.7).

    Surfaces the previously silent last-writer-wins ambiguity -- e.g.
    ``ASK_LLM_TRANSLATION_THREADS`` and ``ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS``
    both map to ``translation.max_concurrent_api_calls``.
    """
    for key_path, env_vars in _duplicate_env_targets().items():
        set_vars = [v for v in env_vars if (os.getenv(v) or "").strip()]
        if len(set_vars) > 1:
            winner = set_vars[-1]  # last in ENV_TO_CONFIG order wins
            logger.warning(
                f"Conflicting env overrides for config key "
                f"{'.'.join(key_path)}: {set_vars} are all set. Using "
                f"{winner!r} (last wins); unset the others to silence this."
            )


def _apply_env_overrides(
    data: dict[str, Any], provenance: dict[str, str] | None = None
) -> dict[str, Any]:
    """Apply ASK_LLM_* environment variable overrides to config data.

    When *provenance* is given, each applied override is recorded as
    ``{"<dotted.key>": "env:<VAR_NAME>"}``.
    """
    result = copy.deepcopy(data)
    _warn_conflicting_env_overrides()
    legacy_chunk = os.getenv("ASK_LLM_TRANSLATION_MAX_CHUNK_SIZE")
    if legacy_chunk is not None and legacy_chunk != "":
        logger.warning(
            "ASK_LLM_TRANSLATION_MAX_CHUNK_SIZE is deprecated and ignored; use "
            "ASK_LLM_TRANSLATION_MAX_CHUNK_TOKENS for token-based chunking."
        )
    for env_var, key_path in ENV_TO_CONFIG.items():
        env_val = os.getenv(env_var)
        if env_val is not None and env_val != "":
            try:
                parsed = _parse_env_value(env_val, key_path)
                _set_nested(result, key_path, parsed)
                if provenance is not None:
                    provenance[".".join(key_path)] = f"env:{env_var}"
                logger.debug(f"Config override from {env_var}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid env {env_var}={env_val!r}: {e}")
    return result


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set a nested key in data. Creates intermediate dicts as needed."""
    current = data
    for part in path[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[path[-1]] = value


def resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variable references.

    Supports ${VAR_NAME} format environment variable references.

    Args:
        value: Value that may contain environment variable references

    Returns:
        Resolved value
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} format
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)

        if matches:
            for var_name in matches:
                env_value = os.getenv(var_name)
                if env_value:
                    value = value.replace(f"${{{var_name}}}", env_value)
                else:
                    if var_name not in _WARNED_UNSET_ENV_VARS:
                        _WARNED_UNSET_ENV_VARS.add(var_name)
                        logger.debug(f"Environment variable {var_name} not set")

        return value
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    else:
        return value
