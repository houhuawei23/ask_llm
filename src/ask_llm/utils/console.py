"""Terminal output: Loguru for leveled logs; Rich for layout, tables, progress, streaming."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger
from rich.console import Console as RichConsole
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Align with arxiv2md-beta / paper_pipeline_beta: timestamp | source | level | message
_LOGURU_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<cyan>{extra[component]}</cyan> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

logger.configure(extra={"component": "ask-llm"})

# Default sink before Typer callback (e.g. import-time errors). CLI.setup() replaces levels.
logger.add(
    sys.stderr,
    level="INFO",
    format=_LOGURU_CONSOLE_FORMAT,
    colorize=True,
)


class Console:
    """
    User-facing output.

    - Loguru: print_success/info/warning/error map to logger levels.
    - Rich: print (styled/Panel), print_markdown, print_table, progress, print_stream, etc.
    """

    _instance: Console | None = None

    def __new__(cls) -> Console:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize console."""
        if not hasattr(self, "_initialized"):
            self._console = RichConsole()
            self._quiet = False
            self._debug = False
            self._initialized = True

    def setup(self, quiet: bool = False, debug: bool = False) -> None:
        """
        Setup console configuration.

        Args:
            quiet: Suppress non-error output
            debug: Enable debug output
        """
        self._quiet = quiet
        self._debug = debug

        logger.remove()
        logger.configure(extra={"component": "ask-llm"})

        if quiet:
            level = "ERROR"
        elif debug:
            level = "DEBUG"
        else:
            level = "INFO"

        fmt = _LOGURU_CONSOLE_FORMAT
        if debug:
            fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<cyan>{extra[component]}</cyan> | "
                "<level>{level: <8}</level> | "
                "<dim>{name}:{function}:{line}</dim> | "
                "<level>{message}</level>"
            )

        logger.add(
            sys.stderr,
            level=level,
            format=fmt,
            colorize=True,
        )

    def print(
        self,
        message: Any = "",
        style: str | None = None,
        panel: bool = False,
        panel_title: str | None = None,
        end: str = "\n",
    ) -> None:
        """
        Rich to stdout: styled text, panels, non-log presentation.

        Args:
            message: Message to print
            style: Rich style string
            panel: Whether to wrap in panel
            panel_title: Panel title
            end: Ending character (default: newline)
        """
        if self._quiet:
            return

        # If end is not newline, we need special handling since Rich's print() may not support end
        if end != "\n":
            # Render with Rich but write to stdout directly
            if panel:
                rendered = Panel(message, title=panel_title, border_style=style)
            else:
                rendered = Text(str(message), style=style) if style else str(message)

            # Use Rich's export_text to get plain text, then write with custom end
            from rich.console import Console as RichConsole

            temp_console = RichConsole(file=sys.stdout, width=self._console.width)
            with temp_console.capture() as capture:
                temp_console.print(rendered)
            output = capture.get()
            sys.stdout.write(output.rstrip("\n") + end)
            sys.stdout.flush()
        else:
            # Normal print with newline
            if panel:
                self._console.print(Panel(message, title=panel_title, border_style=style))
            else:
                self._console.print(message, style=style)

    def print_success(self, message: str) -> None:
        """Map to loguru SUCCESS level."""
        if self._quiet:
            return
        logger.success(message)

    def print_error(self, message: str) -> None:
        """Map to loguru ERROR (not suppressed by quiet)."""
        logger.error(message)

    def print_warning(self, message: str) -> None:
        """Map to loguru WARNING."""
        if self._quiet:
            return
        logger.warning(message)

    def print_info(self, message: str) -> None:
        """Map to loguru INFO."""
        if self._quiet:
            return
        logger.info(message)

    def print_markdown(self, text: str) -> None:
        """Print markdown formatted text."""
        if self._quiet:
            return
        md = Markdown(text)
        self._console.print(md)

    def print_code(self, code: str, language: str = "python", theme: str = "monokai") -> None:
        """Print syntax-highlighted code."""
        if self._quiet:
            return
        syntax = Syntax(code, language, theme=theme, line_numbers=True)
        self._console.print(syntax)

    def print_table(
        self, headers: list[str], rows: list[list[Any]], title: str | None = None
    ) -> None:
        """
        Print data as table.

        Args:
            headers: Column headers
            rows: Table rows
            title: Table title
        """
        if self._quiet:
            return

        table = Table(title=title, show_header=True, header_style="bold magenta")

        for header in headers:
            table.add_column(header)

        for row in rows:
            table.add_row(*[str(cell) for cell in row])

        self._console.print(table)

    def print_stream(self, text: str, end: str = "") -> None:
        """
        Print text without newline (for streaming).

        Args:
            text: Text to print
            end: Ending character
        """
        if self._quiet:
            return
        # For streaming, write directly to stdout to avoid Rich's formatting overhead
        # and ensure compatibility with all Rich versions
        sys.stdout.write(str(text) + end)
        sys.stdout.flush()

    def clear_line(self) -> None:
        """Clear current line."""
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def clear_screen(self) -> None:
        """Clear screen."""
        self._console.clear()

    def progress(self, description: str = "Working...", transient: bool = False) -> Progress:
        """
        Create a progress display.

        Args:
            description: Progress description
            transient: Whether to clear after completion

        Returns:
            Rich Progress object
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self._console,
            transient=transient,
        )

    def input(self, prompt: str = "") -> str:
        """
        Get user input.

        Args:
            prompt: Input prompt

        Returns:
            User input
        """
        return self._console.input(f"[bold]{prompt}[/bold]")

    def confirm(self, message: str, default: bool = False) -> bool:
        """
        Ask for confirmation.

        Args:
            message: Confirmation message
            default: Default value

        Returns:
            True if confirmed
        """
        if self._quiet:
            return default

        prompt = f"{message} [{'Y/n' if default else 'y/N'}]: "
        response = self._console.input(f"[bold yellow]{prompt}[/bold yellow]").strip().lower()

        if not response:
            return default

        return response in ("y", "yes")

    @property
    def width(self) -> int:
        """Get console width."""
        return self._console.width

    @property
    def is_terminal(self) -> bool:
        """Check if output is a terminal."""
        return self._console.is_terminal


# Global console instance
console = Console()
