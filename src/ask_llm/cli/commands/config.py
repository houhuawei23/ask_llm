"""Typer command `config` (split from former cli.py)."""

from __future__ import annotations

import typer
from typing_extensions import Annotated

from ask_llm.cli.errors import cli_errors
from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.utils.console import console

try:
    from llm_engine import create_provider_adapter
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise

from ask_llm.cli.common import _config_init


def config(
    action: Annotated[
        str,
        typer.Argument(help="Action: show, test, init"),
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
) -> None:
    """
    Manage configuration.

    Examples:
        ask-llm config show
        ask-llm config test
        ask-llm config test -p deepseek
        ask-llm config init
        ask-llm config init -o ./my_config.yml
    """
    with cli_errors("config"):
        if action == "init":
            _config_init(output_path)
            return

        # Load existing config
        load_result = ConfigLoader.load(config_path)
        set_config(load_result)
        config = load_result.app_config

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
                console.print(f"  API Key: {'✓ Configured' if pc.api_key else '✗ Not configured'}")
                console.print()

        elif action == "test":
            providers_to_test = [provider] if provider else list(config.providers.keys())

            for name in providers_to_test:
                if name not in config.providers:
                    console.print_error(f"Provider '{name}' not found")
                    continue

                pc = config.providers[name]

                if not pc.api_key or pc.api_key == "your-api-key-here":
                    console.print_warning(f"[{name}] API key not configured")
                    continue

                console.print(f"\nTesting [cyan]{name}[/cyan]...", end=" ")

                try:
                    # Get default model for this provider
                    test_default_model = config.default_model or (
                        pc.models[0] if pc.models else None
                    )
                    if not test_default_model:
                        console.print("[red]✗[/red]")
                        console.print("  Error: No default model available")
                        continue

                    llm_provider = create_provider_adapter(pc, default_model=test_default_model)
                    success, message, latency = llm_provider.test_connection()

                    if success:
                        console.print(f"[green]✓ ({latency:.2f}s)[/green]")
                        console.print(f"  {message}")
                    else:
                        console.print("[red]✗[/red]")
                        console.print(f"  Error: {message}")

                except Exception as e:
                    console.print("[red]✗[/red]")
                    console.print_error(f"  {e}")

            console.print()

        else:
            console.print_error(f"Unknown action: {action}")
            console.print("Available actions: show, test, init")
            raise typer.Exit(1)
