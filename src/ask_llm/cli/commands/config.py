"""Typer command `config` (split from former cli.py)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from ask_llm.cli.common import _config_init
from ask_llm.cli.errors import cli_errors
from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.utils.api_key_gate import (
    PROVIDERS_WITHOUT_API_KEYS,
    api_key_is_missing_or_unresolved,
)
from ask_llm.utils.console import console
from ask_llm.utils.engine_facade import create_engine_adapter
from ask_llm.utils.interactive_config import set_config_value

# Config key paths routed to the providers.yml user file (credentials /
# base_url live there per the runtime/catalog split; everything else belongs
# to default_config.yml).
_PROVIDERS_PREFIX = "providers."


def _mask_if_secret(key_path: str, value: object) -> str:
    """Render a config value for display; secrets are never printed."""
    import re as _re

    leaf = key_path.rsplit(".", 1)[-1]
    if _re.search(r"(api_key|(^|_)(key|token|secret|password)s?$)", leaf, _re.IGNORECASE):
        rendered = str(value)
        if not rendered or rendered.startswith("${"):
            return "✗ Not configured"
        return "✓ Configured (hidden)"
    return str(value)


def _config_get_set(
    action: str,
    key_path: str | None,
    value: str | None,
    config_path: str | None = None,
) -> None:
    """Implement ``config get`` / ``config set`` (plan 5.3)."""
    if not key_path or (action == "set" and value is None):
        console.print_error(
            f"config {action} requires a dotted key path"
            + (" and a value" if action == "set" else "")
        )
        raise typer.Exit(1)

    if action == "get":
        load_result = ConfigLoader.load(config_path)
        set_config(load_result)
        # Walk both the provider config and the unified config.
        # M17/2.25: a sentinel distinguishes "no such attribute/key" from a
        # legitimately-None value, so typos error while nullable keys print.
        _missing_sentinel = object()
        # Unified sections (translation.*, format_body.*, …) live on
        # unified_config, providers on app_config — start the walk at the
        # right root so non-provider keys are reachable too.
        first_segment = key_path.split(".", 1)[0]
        root = (
            load_result.unified_config
            if hasattr(load_result.unified_config, first_segment)
            else load_result.app_config
        )
        target: object = root
        for part in key_path.split("."):
            if isinstance(target, dict):
                target = target.get(part, _missing_sentinel)
            else:
                target = getattr(target, part, _missing_sentinel)
            if target is _missing_sentinel:
                console.print_error(f"Key not found: {key_path}")
                raise typer.Exit(1)
        if isinstance(target, list):
            console.print(str(target))
        else:
            console.print(_mask_if_secret(key_path, target))
        return

    # set: route to the right user file — provider keys/base_url to
    # providers.yml (0600), everything else to default_config.yml.
    if key_path.startswith(_PROVIDERS_PREFIX):
        target_file = Path.home() / ".config" / "ask_llm" / "providers.yml"
        mode: int | None = 0o600
    else:
        target_file = Path.home() / ".config" / "ask_llm" / "default_config.yml"
        mode = None

    assert value is not None  # guarded above
    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if not target_file.exists():
            target_file.touch()
        preserved = set_config_value(target_file, key_path, value, mode=mode)
    except Exception as e:
        console.print_error(f"Failed to set {key_path}: {e}")
        raise typer.Exit(1) from e

    console.print_success(f"{key_path} set in {target_file}")
    if not preserved:
        console.print_warning("New key created; the file's comments were normalized.")


def config(
    action: Annotated[
        str,
        typer.Argument(help="Action: show, test, init, get, set"),
    ] = "show",
    config_path: Annotated[
        str | None, typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Provider to test (with test action)")
    ] = None,
    output_path: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output path for init (default: ~/.config/ask_llm/default_config.yml)",
        ),
    ] = None,
    debug_config: Annotated[
        bool,
        typer.Option(
            "--debug-config",
            help="Show configuration provenance: loaded file path and active env-var overrides",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="With init: overwrite existing files without prompting (script-friendly)",
        ),
    ] = False,
    key_path: Annotated[
        str | None,
        typer.Argument(help="Dotted config key for get/set, e.g. providers.deepseek.api_key"),
    ] = None,
    value: Annotated[
        str | None,
        typer.Argument(help="Value to set (with set action)"),
    ] = None,
) -> None:
    """
    Manage configuration.

    Examples:
        ask-llm config show
        ask-llm config test
        ask-llm config test -p deepseek
        ask-llm config init
        ask-llm config init -o ./my_config.yml
        ask-llm config show --debug-config
        ask-llm config get providers.deepseek.api_key
        ask-llm config set providers.deepseek.api_key sk-...
        ask-llm config set translation.max_chunk_tokens 3000
    """
    with cli_errors("config"):
        if action == "init":
            _config_init(output_path, yes=yes)
            return

        if action in ("get", "set"):
            _config_get_set(action, key_path, value, config_path)
            return

        # Load existing config
        load_result = ConfigLoader.load(config_path)
        set_config(load_result)
        config = load_result.app_config

        if debug_config:
            console.print("")
            console.print("[bold]Configuration Provenance:[/bold]")
            console.print(f"  Loaded config file: {load_result.config_path}")
            active_env = [k for k in os.environ if k.startswith("ASK_LLM_") and os.environ[k]]
            if active_env:
                console.print("  Active ASK_LLM_* env overrides:")
                for key in sorted(active_env):
                    masked = "***" if "KEY" in key or "SECRET" in key else os.environ[key]
                    console.print(f"    {key} = {masked}")
            else:
                console.print("  Active ASK_LLM_* env overrides: (none)")

            # Per-key provenance: which layer supplied each final value.
            if load_result.provenance:
                by_source: dict[str, list[str]] = {}
                for key_path, source in load_result.provenance.items():
                    by_source.setdefault(source, []).append(key_path)
                console.print("  Value sources (per key, raw config-file naming):")
                for source in sorted(by_source):
                    keys = sorted(by_source[source])
                    console.print(f"    {source} ({len(keys)} keys):")
                    for key_path in keys:
                        console.print(f"      {key_path}")
            console.print("")

        if action == "show":
            console.print("")
            console.print("[bold]Configuration:[/bold]")
            console.print(f"  Default Provider: {config.default_provider}")
            console.print()

            for name, pc in config.providers.items():
                default_marker = (
                    " [green]✓ default[/green]" if name == config.default_provider else ""
                )
                console.print(f"[cyan]{name}[/cyan]{default_marker}")
                console.print(f"  API Base: {pc.api_base}")
                # Show default model: use first model from provider's models (which should be the default)
                default_model = pc.models[0] if pc.models else "N/A"
                console.print(f"  Default Model: {default_model}")
                if pc.models:
                    console.print(f"  Available Models: {', '.join(pc.models)}")
                # H3: judge by the same rule as `config test` — an unresolved
                # ${VAR} placeholder is NOT a configured key.
                if api_key_is_missing_or_unresolved(pc.api_key):
                    console.print("  API Key: ✗ Not configured")
                else:
                    console.print("  API Key: ✓ Configured")
                console.print()

        elif action == "test":
            providers_to_test = [provider] if provider else list(config.providers.keys())
            # H3: a failed connection check must yield a non-zero exit code so
            # `config test` works as a health check in scripts/CI. A missing
            # key is only fatal when the provider was requested explicitly —
            # the default catalog sweep always includes unconfigured providers.
            explicit_target = provider is not None
            any_failed = False

            for name in providers_to_test:
                if name not in config.providers:
                    console.print_error(f"Provider '{name}' not found")
                    any_failed = True
                    continue

                pc = config.providers[name]

                if name not in PROVIDERS_WITHOUT_API_KEYS and api_key_is_missing_or_unresolved(
                    pc.api_key
                ):
                    console.print_warning(f"[{name}] API key not configured")
                    if explicit_target:
                        any_failed = True
                    continue

                console.print(f"\nTesting [cyan]{name}[/cyan]...", end=" ")

                try:
                    # Same priority as ConfigManager.get_default_model: the
                    # provider's own models[0] first, global default_model only
                    # as fallback (H3: the reversed order tested e.g. an
                    # OpenAI-named model against DeepSeek and reported bogus
                    # failures).
                    test_default_model = pc.models[0] if pc.models else config.default_model
                    if not test_default_model:
                        console.print("[red]✗[/red]")
                        console.print("  Error: No default model available")
                        any_failed = True
                        continue

                    llm_provider = create_engine_adapter(pc, default_model=test_default_model)
                    success, message, latency = llm_provider.test_connection()

                    if success:
                        console.print(f"[green]✓ ({latency:.2f}s)[/green]")
                        console.print(f"  {message}")
                    else:
                        console.print("[red]✗[/red]")
                        console.print(f"  Error: {message}")
                        any_failed = True

                except Exception as e:
                    console.print("[red]✗[/red]")
                    console.print_error(f"  {e}")
                    any_failed = True

            console.print()
            if any_failed:
                raise typer.Exit(1)

        else:
            console.print_error(f"Unknown action: {action}")
            console.print("Available actions: show, test, init, get, set")
            raise typer.Exit(1)
