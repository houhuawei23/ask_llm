"""Shared CLI bootstrap: load config, ConfigManager, overrides, API key gate."""

from __future__ import annotations

from pathlib import Path

import typer

from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader, LoadResult
from ask_llm.config.manager import ConfigManager
from ask_llm.utils.api_key_gate import ensure_api_key_for_provider, require_resolved_api_key
from ask_llm.utils.console import console


def load_cli_session(
    config_path: str | Path | None = None,
) -> tuple[LoadResult, ConfigManager]:
    """
    Load default_config.yml (search paths per ConfigLoader), set global config context,
    and return LoadResult plus a fresh ConfigManager for the loaded app config.
    """
    load_result = ConfigLoader.load(config_path)
    set_config(load_result)
    config_manager = ConfigManager(load_result.app_config)
    return load_result, config_manager


def apply_cli_overrides_and_gate_api_key(
    config_manager: ConfigManager,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    skip_api_key_check: bool = False,
) -> None:
    """
    Optionally set active provider, apply model/temperature overrides, then run
    ensure_api_key_for_provider + require_resolved_api_key when the strict gate applies.
    """
    if provider:
        config_manager.set_provider(provider)
    config_manager.apply_overrides(model=model, temperature=temperature)
    pname = config_manager.current_provider_name
    strict_gate = ensure_api_key_for_provider(
        config_manager,
        pname,
        skip_api_key_check=skip_api_key_check,
    )
    if strict_gate:
        require_resolved_api_key(config_manager, pname)


def resolve_default_model_or_exit(config_manager: ConfigManager) -> str:
    """Resolve default model after overrides; exit with a clear error if unset."""
    default_model = config_manager.get_model_override() or config_manager.get_default_model()
    if not default_model:
        console.print_error(
            "No model specified. Use --model or set a default model for the provider."
        )
        raise typer.Exit(1)
    return default_model
