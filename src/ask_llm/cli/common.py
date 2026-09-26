"""Shared CLI helpers (config init).

Config-init is the only remaining shared CLI helper; path-resolution lives in
``ask_llm.utils.path_resolver`` and translation paths flow through the
service layer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer

from ask_llm.config.cli_session import (
    LoadResult,
    gate_api_key_or_exit,
    load_cli_session,
    resolve_and_prepare,
)
from ask_llm.config.manager import ConfigManager
from ask_llm.utils.api_key_gate import (
    UnresolvedAPIKeyError,  # noqa: F401  (re-exported for callers)
)
from ask_llm.utils.console import console


def _config_init(output_path: str | None = None, *, yes: bool = False) -> None:
    """Generate default_config.yml and providers.yml templates.

    ``yes`` (L12/2.25) overwrites existing files without prompting, so
    scripts and non-interactive environments don't die on typer's Abort.
    """
    pkg_dir = Path(__file__).resolve().parent.parent / "config"
    pkg_config = pkg_dir / "default_config.yml"
    pkg_providers = pkg_dir / "providers.yml"
    if not pkg_config.exists():
        console.print_error("Package default config not found")
        raise typer.Exit(1)

    if output_path:
        dest = Path(output_path)
    else:
        dest = Path.home() / ".config" / "ask_llm" / "default_config.yml"

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        console.print_warning(f"File exists: {dest}")
        if not yes and not typer.confirm("Overwrite?"):
            raise typer.Exit(0)

    try:
        content = pkg_config.read_text(encoding="utf-8")
        dest.write_text(content, encoding="utf-8")
        console.print_success(f"Configuration template written to: {dest}")
        console.print("Edit the file to set your API keys (use ${VAR} for environment variables).")
    except Exception as e:
        console.print_error(f"Failed to write config: {e}")
        raise typer.Exit(1) from e

    if pkg_providers.exists():
        providers_dest = dest.parent / "providers.yml"
        if providers_dest.exists():
            console.print_warning(f"File exists: {providers_dest}")
            if not yes and not typer.confirm("Overwrite providers.yml?"):
                return
        try:
            providers_dest.write_text(pkg_providers.read_text(encoding="utf-8"), encoding="utf-8")
            console.print_success(f"Provider catalog written to: {providers_dest}")
        except Exception as e:
            console.print_error(f"Failed to write providers.yml: {e}")
            raise typer.Exit(1) from e


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


def paid_command_prelude(
    config_path: str | Path | None,
    *,
    provider: str | None,
    model: str | None,
    temperature: float | Callable[[LoadResult], float],
    pricing_path: str | Path | None = None,
    skip_api_key_check: bool = False,
) -> tuple[LoadResult, ConfigManager, str, str, dict, Path | None]:
    """One preamble for every command that can spend API tokens.

    Composes the standard paid-command sequence: load config + pricing →
    resolve provider/model → gate the API key. Commands only add their
    command-specific parsing after this returns; none of the load/resolve/
    gate logic may be duplicated per command.

    Args:
        config_path: Optional explicit default_config.yml path.
        provider: CLI --provider override (None = config default).
        model: CLI --model override (None = config default).
        temperature: Effective temperature, either a plain float (CLI flag)
            or a callable evaluated on the loaded config for the command's
            own section default (e.g. ``lambda lr: lr.unified_config.paper.temperature``).
        pricing_path: Optional explicit providers.yml pricing path.
        skip_api_key_check: Forwarded to the gate (``--dry-run`` passes True).

    Returns:
        ``(load_result, config_manager, provider, model, pricing_map, pricing_source)``.
    """
    load_result, config_manager, pricing_map, pricing_source = bootstrap_command(
        config_path,
        pricing_path=pricing_path,
    )
    effective_temperature = temperature(load_result) if callable(temperature) else temperature
    final_provider, final_model = resolve_and_prepare(
        config_manager,
        cli_provider=provider,
        cli_model=model,
        temperature=effective_temperature,
    )
    gate_api_key_or_exit(config_manager, final_provider, skip_api_key_check=skip_api_key_check)
    return load_result, config_manager, final_provider, final_model, pricing_map, pricing_source
