"""Typer app assembly and global callback."""

from __future__ import annotations

import typer
from typing_extensions import Annotated

from ask_llm import __version__
from ask_llm.cli.commands.ask import ask
from ask_llm.cli.commands.batch import batch
from ask_llm.cli.commands.chat import chat
from ask_llm.cli.commands.config import config
from ask_llm.cli.commands.format_cmd import format_cmd
from ask_llm.cli.commands.paper import paper
from ask_llm.cli.commands.trans import trans
from ask_llm.utils.console import console

app = typer.Typer(
    name="ask-llm",
    help="Ask LLM - A flexible command-line tool for calling multiple LLM APIs",
    rich_markup_mode="rich",
    add_completion=False,
)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"Ask LLM version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Enable debug logging")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-error output")] = False,
) -> None:
    """Ask LLM - A flexible command-line tool for calling multiple LLM APIs."""
    console.setup(quiet=quiet, debug=debug)


app.command()(ask)
app.command()(chat)
app.command()(config)
app.command()(batch)
app.command()(trans)
app.command("format")(format_cmd)
app.command("paper")(paper)


def run_cli() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    run_cli()
