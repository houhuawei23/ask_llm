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


def resolve_and_prepare(
    config_manager: ConfigManager,
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    temperature: float | None = None,
) -> tuple[str, str]:
    """
    Single provider/model resolution entry for CLI commands.

    Priority:
        1. CLI --provider / --model
        2. Configured default provider / model

    Validates provider membership, applies the model/temperature overrides
    exactly once, and returns the effective pair. Follow up with
    :func:`gate_api_key_or_exit` (commands with a pre-network dry-run path
    gate after that path instead).

    Args:
        config_manager: Active config manager.
        cli_provider: Provider passed on the CLI.
        cli_model: Model passed on the CLI.
        temperature: Effective temperature (CLI flag or command config).

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

    config_manager.apply_overrides(model=model, temperature=temperature)
    return provider, model


def gate_api_key_or_exit(
    config_manager: ConfigManager,
    provider: str,
    *,
    skip_api_key_check: bool = False,
) -> None:
    """
    Run the interactive API-key gate, then the resolved-key check when strict.

    Args:
        config_manager: Active config manager.
        provider: Resolved provider name.
        skip_api_key_check: Skip the gate when the user passed the flag.
    """
    strict_gate = ensure_api_key_for_provider(
        config_manager,
        provider,
        skip_api_key_check=skip_api_key_check,
    )
    if strict_gate:
        require_resolved_api_key(config_manager, provider)


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
) -> tuple[LoadResult, ConfigManager, dict, Path | None]:
    """One-call CLI bootstrap (P4.4).

    Composes :func:`load_cli_session` + :func:`load_pricing_with_hint` into the
    standard command preamble shared by trans/paper (and future commands).
    Provider/model resolution is NOT included: callers follow up with
    :func:`resolve_and_prepare` once the command-specific effective temperature
    (CLI flag or command config section) is known, then :func:`gate_api_key_or_exit`.

    Returns:
        ``(load_result, config_manager, pricing_map, pricing_source)``.
    """
    load_result, config_manager = load_cli_session(config_path)
    pricing_map, pricing_source = load_pricing_with_hint(pricing_path)
    return load_result, config_manager, pricing_map, pricing_source
