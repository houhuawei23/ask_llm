"""Custom completer for chat mode meta commands."""

from typing import Iterable
from pathlib import Path

try:
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.document import Document
except ImportError:
    # Fallback if prompt_toolkit is not available
    Completer = object
    Completion = None
    Document = None


class ChatCompleter(Completer):
    """Custom completer for chat mode commands."""

    # Meta commands that start with /
    META_COMMANDS = [
        "help",
        "info",
        "config",
        "check-config",
        "providers",
        "models",
        "model",
        "switch-model",
        "history",
        "hist",
        "save-history",
        "save",
        "clear-history",
        "clear",
        "system-prompt",
        "system",
        "prompt",
        "clear-system-prompt",
        "clear-system",
    ]

    # Shell command prefixes
    SHELL_PREFIXES = ["!", "!!"]

    def __init__(self):
        """Initialize the completer."""
        super().__init__()

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """
        Get completions for the current input.

        Args:
            document: The current document/input
            complete_event: The completion event

        Yields:
            Completion objects
        """
        if Completion is None:
            return

        text = document.text_before_cursor

        # Complete meta commands (starting with /)
        if text.startswith("/"):
            # Get the command part (after /)
            parts = text[1:].split(" ", 1)
            cmd_prefix = parts[0].lower()

            for cmd in self.META_COMMANDS:
                if cmd.startswith(cmd_prefix):
                    # Check if command already has arguments
                    if len(parts) > 1:
                        # Command already has arguments, don't complete
                        continue
                    yield Completion(
                        f"/{cmd}",
                        start_position=-len(text),
                        display=cmd,
                        display_meta=f"Meta command: {cmd}",
                    )

        # Complete shell commands (starting with !)
        elif text.startswith("!"):
            # For shell commands, we can complete file paths if it looks like a path
            # But for now, just complete common shell commands
            if text == "!":
                # Show common shell commands
                common_commands = ["ls", "pwd", "cd", "cat", "grep", "find", "ps"]
                for cmd in common_commands:
                    yield Completion(
                        f"!{cmd}",
                        start_position=-len(text),
                        display=cmd,
                        display_meta=f"Shell command: {cmd}",
                    )
            elif text == "!!":
                # Already complete
                pass
            else:
                # Try to complete file paths if it looks like a path
                cmd_part = text[1:].strip()
                if "/" in cmd_part or cmd_part.startswith("~"):
                    # Looks like a path, try file completion
                    yield from self._complete_file_path(cmd_part, text)

    def _complete_file_path(self, path_part: str, full_text: str) -> Iterable[Completion]:
        """
        Complete file paths.

        Args:
            path_part: The path part to complete
            full_text: The full text before cursor

        Yields:
            Completion objects for file paths
        """
        if Completion is None:
            return

        try:
            # Expand ~ to home directory
            if path_part.startswith("~"):
                path_part = str(Path.home()) + path_part[1:]

            path = Path(path_part)

            # If it's a directory, list its contents
            if path.is_dir():
                parent = path
                prefix = ""
            elif path.parent.exists() and path.parent.is_dir():
                parent = path.parent
                prefix = path.name
            else:
                return

            # List directory contents
            try:
                for item in parent.iterdir():
                    if item.name.startswith(prefix):
                        if item.is_dir():
                            display = f"{item.name}/"
                            meta = "Directory"
                        else:
                            display = item.name
                            meta = "File"

                        # Calculate the completion text
                        if path_part.endswith("/"):
                            completion_text = f"!{str(item)}"
                        else:
                            # Replace the last part
                            completion_text = f"!{str(parent / item.name)}"

                        yield Completion(
                            completion_text,
                            start_position=-len(full_text),
                            display=display,
                            display_meta=meta,
                        )
            except PermissionError:
                pass
        except Exception:
            pass
