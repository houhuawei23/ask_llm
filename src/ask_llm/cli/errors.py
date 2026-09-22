"""Unified CLI error handling: user-visible messages and loguru diagnostics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import click
import typer
from loguru import logger

from ask_llm.utils.console import console


def raise_unexpected_cli_error(command_name: str, exc: Exception) -> None:
    """Print a short user message, log full traceback, and exit with code 1."""
    console.print_error(f"Unexpected error: {exc}")
    logger.exception("%s command failed", command_name)
    raise typer.Exit(1) from exc


_API_ERROR_MARKERS = (
    "api",
    "authentication",
    "rate limit",
    "provider",
    "http",
    "timeout",
    "request",
    "model",
)


def _render_runtime_error(e: RuntimeError) -> str:
    """Prefix API-origin RuntimeErrors; render service RuntimeErrors plainly."""
    message = str(e)
    lowered = message.lower()
    if any(marker in lowered for marker in _API_ERROR_MARKERS):
        return f"API error: {message}"
    return message


@contextmanager
def cli_errors(command_name: str) -> Iterator[None]:
    """Outer catch-all for Typer commands: maps common exceptions to exit 1 and logging."""
    try:
        yield
    except typer.Exit:
        raise
    except click.exceptions.UsageError:
        # H4: argument validation errors (typer.BadParameter et al.) must keep
        # click's standard usage message + exit code 2, not be swallowed into
        # a generic "Unexpected error" with exit 1.
        raise
    except KeyboardInterrupt:
        console.print("\nInterrupted by user")
        raise typer.Exit(1) from None
    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except OSError as e:
        # H4: permission errors, unwritable outputs, unreadable configs — the
        # most common failure class — deserve a readable message, not a
        # traceback-shaped "Unexpected error".
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        # L6/2.25: not every RuntimeError is an API error — services raise
        # them for resume refusal, write failures, etc. Only API-looking
        # messages keep the "API error:" prefix; the rest render plainly.
        console.print_error(_render_runtime_error(e))
        raise typer.Exit(1) from e
    except Exception as e:
        raise_unexpected_cli_error(command_name, e)
