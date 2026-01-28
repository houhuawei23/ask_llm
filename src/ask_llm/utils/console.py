"""Console output utilities using Rich."""

import sys
from typing import Any, Optional

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from loguru import logger


class Console:
    """Console output handler with Rich formatting."""
    
    _instance: Optional["Console"] = None
    
    def __new__(cls) -> "Console":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize console."""
        if not hasattr(self, "_initialized"):
            self._console = RichConsole()
            self._error_console = RichConsole(stderr=True)
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
        
        # Configure loguru
        logger.remove()
        
        if quiet:
            level = "ERROR"
        elif debug:
            level = "DEBUG"
        else:
            level = "INFO"
        
        logger.add(
            sys.stderr,
            level=level,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )
    
    def print(
        self,
        message: Any = "",
        style: Optional[str] = None,
        panel: bool = False,
        panel_title: Optional[str] = None,
        end: str = "\n"
    ) -> None:
        """
        Print message to console.
        
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
                self._console.print(
                    Panel(message, title=panel_title, border_style=style)
                )
            else:
                self._console.print(message, style=style)
    
    def print_success(self, message: str) -> None:
        """Print success message."""
        self.print(f"✓ {message}", style="green")
    
    def print_error(self, message: str) -> None:
        """Print error message."""
        self._error_console.print(f"✗ {message}", style="bold red")
    
    def print_warning(self, message: str) -> None:
        """Print warning message."""
        self.print(f"⚠ {message}", style="yellow")
    
    def print_info(self, message: str) -> None:
        """Print info message."""
        self.print(f"ℹ {message}", style="blue")
    
    def print_markdown(self, text: str) -> None:
        """Print markdown formatted text."""
        if self._quiet:
            return
        md = Markdown(text)
        self._console.print(md)
    
    def print_code(
        self,
        code: str,
        language: str = "python",
        theme: str = "monokai"
    ) -> None:
        """Print syntax-highlighted code."""
        if self._quiet:
            return
        syntax = Syntax(code, language, theme=theme, line_numbers=True)
        self._console.print(syntax)
    
    def print_table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        title: Optional[str] = None
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
    
    def progress(
        self,
        description: str = "Working...",
        transient: bool = False
    ) -> Progress:
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
