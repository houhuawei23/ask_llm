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
    config_manager = ConfigManager(load_result.app_config, load_result.unified_config)
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


def load_pricing_with_hint(
    explicit_path: str | Path | None = None,
) -> tuple[dict, Path | None]:
    """Load providers.yml pricing and print the standard CLI hint (P4.4).

    Single home for the previously byte-identical 6-line pricing block in
    the batch/trans/paper commands.

    Returns:
        Tuple of (pricing_map, pricing_source_path_or_None).
    """
    from ask_llm.utils.pricing import load_providers_pricing

    pricing_map, pricing_source = load_providers_pricing(explicit_path)
    if pricing_source:
        console.print_info(f"API pricing loaded from: {pricing_source}")
    else:
        console.print_info(
            "No providers.yml with pricing found; token counts will still be shown, "
            "cost estimate unavailable (add pricing_per_million_tokens or use --providers-pricing)"
        )
    return pricing_map, pricing_source


def bootstrap_command(
    config_path: str | Path | None = None,
    *,
    pricing_path: str | Path | None = None,
    cli_provider: str | None = None,
    cli_model: str | None = None,
) -> tuple[LoadResult, ConfigManager, dict, Path | None, str, str]:
    """One-call CLI bootstrap (P4.4).

    Composes :func:`load_cli_session` + :func:`load_pricing_with_hint` +
    :func:`resolve_provider_and_model_or_exit` into the standard command
    preamble shared by trans/paper (and future commands).

    Returns:
        ``(load_result, config_manager, pricing_map, pricing_source,
        provider, model)``.
    """
    load_result, config_manager = load_cli_session(config_path)
    pricing_map, pricing_source = load_pricing_with_hint(pricing_path)
    provider, model = resolve_provider_and_model_or_exit(
        config_manager,
        cli_provider=cli_provider,
        cli_model=cli_model,
    )
    return load_result, config_manager, pricing_map, pricing_source, provider, model


def resolve_provider_and_model_or_exit(
    config_manager: ConfigManager,
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
) -> tuple[str, str]:
    """Resolve provider and model from CLI args and config, exiting if either is missing.

    Priority:
        1. CLI --provider / --model
        2. Configured default provider / model

    Args:
        config_manager: Active config manager.
        cli_provider: Provider passed on the CLI.
        cli_model: Model passed on the CLI.

    Returns:
        Tuple of (resolved_provider, resolved_model).

    Raises:
        typer.Exit: If provider or model cannot be resolved.
    """
    provider = cli_provider or config_manager.config.default_provider
    if not provider:
        console.print_error(
            "No provider specified. Use --provider or configure a default provider."
        )
        raise typer.Exit(1)

    try:
        config_manager.set_provider(provider)
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e

    try:
        model = cli_model or config_manager.get_default_model()
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e

    if not model:
        console.print_error(
            "No model specified. Use --model or configure a default model for the provider."
        )
        raise typer.Exit(1)

    return provider, model
