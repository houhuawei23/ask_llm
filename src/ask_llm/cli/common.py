"""Shared CLI helpers (config init, translation path resolution).

The path-resolution helpers live in ``ask_llm.utils.path_resolver`` (P4.3)
and are re-exported here for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ask_llm.utils.console import console


def _config_init(output_path: str | None = None) -> None:
    """Generate default_config.yml and providers.yml templates."""
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
        if not typer.confirm("Overwrite?"):
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
            if not typer.confirm("Overwrite providers.yml?"):
                return
        try:
            providers_dest.write_text(pkg_providers.read_text(encoding="utf-8"), encoding="utf-8")
            console.print_success(f"Provider catalog written to: {providers_dest}")
        except Exception as e:
            console.print_error(f"Failed to write providers.yml: {e}")
            raise typer.Exit(1) from e
