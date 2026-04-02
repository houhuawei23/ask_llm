"""Unified CLI error handling: user-visible messages and loguru diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import typer
from loguru import logger

from ask_llm.utils.console import console


def raise_unexpected_cli_error(command_name: str, exc: Exception) -> None:
    """Print a short user message, log full traceback, and exit with code 1."""
    console.print_error(f"Unexpected error: {exc}")
    logger.exception("%s command failed", command_name)
    raise typer.Exit(1) from exc


@contextmanager
def cli_errors(command_name: str) -> Iterator[None]:
    """Outer catch-all for Typer commands: maps common exceptions to exit 1 and logging."""
    try:
        yield
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        raise
    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        raise_unexpected_cli_error(command_name, e)
