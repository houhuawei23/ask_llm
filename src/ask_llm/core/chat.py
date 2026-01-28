"""Interactive chat session implementation."""

import json
import shlex
import subprocess
import time
from datetime import datetime
from typing import ClassVar, List, Optional

from loguru import logger

from ask_llm.core.models import ChatHistory, ChatMessage, MessageRole
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.utils.console import console
from ask_llm.utils.token_counter import TokenCounter


class ChatSession:
    """Interactive chat session with meta commands."""

    # Meta commands
    META_COMMANDS: ClassVar[dict[str, str]] = {
        "/help": "Show this help message",
        "/info": "Show current session information",
        "/config": "Show current configuration",
        "/providers": "List all available providers",
        "/models": "List available models",
        "/model": "Show or switch model (usage: /model <name>)",
        "/history": "Show conversation history summary",
        "/save": "Save conversation history to file (usage: /save <file>)",
        "/clear": "Clear conversation history",
        "/system": "Show or set system prompt (usage: /system <text>)",
        "/clear-system": "Clear system prompt",
    }

    def __init__(
        self,
        provider: LLMProviderProtocol,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        history: Optional[ChatHistory] = None,
        config_manager=None,
    ):
        """
        Initialize chat session.

        Args:
            provider: LLM provider instance
            temperature: Sampling temperature
            model: Model name
            history: Existing chat history
            config_manager: Configuration manager for switching providers
        """
        self.provider = provider
        self.temperature = temperature
        self.model = model or provider.default_model
        self.history = history or ChatHistory(provider=provider.name, model=self.model)
        self.config_manager = config_manager

        # Shell command history
        self._last_shell_cmd: Optional[str] = None
        self._shell_history: List[str] = []

    def start(self) -> None:
        """Start interactive chat session."""
        console.print()
        console.print("═" * 50, style="cyan")
        console.print("  Interactive Chat Mode", style="bold cyan")
        console.print("═" * 50, style="cyan")
        console.print()
        console.print("Commands:", style="dim")
        console.print("  Type [bold]/help[/bold] for available commands")
        console.print("  Type [bold]!command[/bold] to execute shell commands")
        console.print("  Type [bold]exit[/bold] or [bold]quit[/bold] to exit")
        console.print()

        logger.info(f"Started chat session with {self.provider.name}")

        try:
            self._chat_loop()
        except KeyboardInterrupt:
            console.print("\n\nGoodbye!", style="green")

    def _chat_loop(self) -> None:
        """Main chat loop."""
        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ").strip()

                if not user_input:
                    continue

                # Handle exit
                if user_input.lower() in ("exit", "quit", "q"):
                    console.print("Goodbye!", style="green")
                    break

                # Handle shell commands
                if user_input.startswith("!") and self._handle_shell_command(user_input[1:]):
                    continue

                # Handle meta commands
                if user_input.startswith("/") and self._handle_meta_command(user_input):
                    continue

                # Regular chat message
                self._send_message(user_input)

            except KeyboardInterrupt:
                console.print("\nUse 'exit' or Ctrl+D to quit", style="dim")
            except EOFError:
                console.print("\nGoodbye!", style="green")
                break
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                console.print_error(f"Error: {e}")

    def _send_message(self, content: str) -> None:
        """
        Send message to LLM and display response.

        Args:
            content: User message content
        """
        # Add user message to history
        self.history.add_message(MessageRole.USER, content)

        # Get messages for API
        messages = self.history.get_messages()

        console.print("[bold blue]Assistant:[/bold blue] ", end="")

        try:
            start_time = time.time()

            # Stream response
            response_parts = []
            stream = self.provider.call(
                messages=messages, temperature=self.temperature, model=self.model, stream=True
            )

            for chunk in stream:
                response_parts.append(chunk)
                console.print_stream(chunk, end="")

            response = "".join(response_parts)
            latency = time.time() - start_time

            console.print()  # Newline after response
            console.print(f"  (Latency: {latency:.2f}s)", style="dim")
            console.print()

            # Add to history
            self.history.add_message(MessageRole.ASSISTANT, response)

            logger.debug(f"Response received: {len(response)} chars in {latency:.2f}s")

        except Exception as e:
            console.print()
            console.print_error(f"Failed to get response: {e}")
            # Remove user message from history if it was just added
            if self.history.messages and self.history.messages[-1].role == MessageRole.USER:
                self.history.messages.pop()

    def _handle_meta_command(self, command: str) -> bool:
        """
        Handle meta command.

        Args:
            command: Command string including /

        Returns:
            True if command was handled
        """
        parts = command[1:].split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handler = getattr(self, f"_cmd_{cmd.replace('-', '_')}", None)
        if handler:
            handler(args)
            return True

        console.print_warning(f"Unknown command: {command}")
        return True

    def _cmd_help(self, args: str) -> None:
        """Show help message."""
        console.print()
        console.print("[bold]Available Commands:[/bold]")

        table_data = []
        for cmd, desc in self.META_COMMANDS.items():
            table_data.append((f"[cyan]{cmd}[/cyan]", desc))

        for cmd, desc in table_data:
            console.print(f"  {cmd:<20} {desc}")

        console.print()
        console.print("[bold]Shell Commands:[/bold]")
        console.print("  [cyan]!command[/cyan]           Execute shell command")
        console.print("  [cyan]!![/cyan]                 Repeat last command")
        console.print()

    def _cmd_info(self, args: str) -> None:
        """Show session info."""
        user_msgs = sum(1 for m in self.history.messages if m.role == MessageRole.USER)
        assistant_msgs = sum(1 for m in self.history.messages if m.role == MessageRole.ASSISTANT)

        # Estimate tokens
        total_tokens = sum(
            TokenCounter.count_tokens(m.content, self.model) for m in self.history.messages
        )

        console.print()
        console.print("[bold]Session Information:[/bold]")
        console.print(f"  Provider: {self.provider.name}")
        console.print(f"  Model: {self.model}")
        temp_display = self.temperature if self.temperature is not None else "default"
        console.print(f"  Temperature: {temp_display}")
        console.print(f"  Messages: {user_msgs} user, {assistant_msgs} assistant")
        console.print(f"  Estimated Tokens: {total_tokens}")
        console.print()

    def _cmd_config(self, args: str) -> None:
        """Show configuration."""
        config = self.provider.config

        console.print()
        console.print("[bold]Current Configuration:[/bold]")
        console.print(f"  Provider: {self.provider.name}")
        console.print(f"  Model: {self.model}")
        console.print(f"  Temperature: {self.temperature}")
        console.print(f"  API Base: {config.api_base}")
        console.print(f"  API Key: {'✓ Configured' if config.api_key else '✗ Not configured'}")
        console.print()

    def _cmd_providers(self, args: str) -> None:
        """List available providers."""
        if not self.config_manager:
            console.print_warning("Provider switching not available")
            return

        providers = self.config_manager.get_available_providers()
        current = self.provider.name

        console.print()
        console.print("[bold]Available Providers:[/bold]")
        for name in providers:
            marker = " [green]✓[/green]" if name == current else ""
            console.print(f"  {name}{marker}")
        console.print()

    def _cmd_models(self, args: str) -> None:
        """List available models."""
        models = self.provider.available_models
        current = self.model

        console.print()
        console.print(f"[bold]Available Models for {self.provider.name}:[/bold]")
        for model_name in models:
            marker = " [green]✓[/green]" if model_name == current else ""
            console.print(f"  {model_name}{marker}")
        console.print()

    def _cmd_model(self, args: str) -> None:
        """Show or switch model."""
        if not args:
            console.print(f"\nCurrent model: [bold]{self.model}[/bold]\n")
            return

        new_model = args.strip()
        available = self.provider.available_models

        if available and new_model not in available:
            console.print_warning(f"Model '{new_model}' not in available list")

        self.model = new_model
        self.history.model = new_model
        console.print_success(f"Switched to model: {new_model}")

    def _cmd_history(self, args: str) -> None:
        """Show conversation history."""
        if not self.history.messages:
            console.print("\nNo messages yet.\n")
            return

        console.print()
        console.print("[bold]Conversation History:[/bold]")

        for i, msg in enumerate(self.history.messages[-10:], 1):
            role_color = {
                MessageRole.SYSTEM: "yellow",
                MessageRole.USER: "green",
                MessageRole.ASSISTANT: "blue",
            }.get(msg.role, "white")

            preview = msg.content[:60].replace("\n", " ")
            if len(msg.content) > 60:
                preview += "..."

            console.print(f"  {i}. [{role_color}]{msg.role.value.upper()}[/{role_color}] {preview}")

        console.print()

    def _cmd_save(self, args: str) -> None:
        """Save history to file."""
        if not args:
            console.print_warning("Usage: /save <filename>")
            return

        filename = args.strip()
        try:
            data = self.history.to_dict()
            data["saved_at"] = datetime.now().isoformat()

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            console.print_success(f"History saved to: {filename}")
        except Exception as e:
            console.print_error(f"Failed to save: {e}")

    def _cmd_clear(self, args: str) -> None:
        """Clear conversation history."""
        self.history.clear(keep_system=True)
        console.print_success("Conversation history cleared")

    def _cmd_system(self, args: str) -> None:
        """Show or set system prompt."""
        if not args:
            # Show current system prompt
            system_msgs = [m for m in self.history.messages if m.role == MessageRole.SYSTEM]
            if system_msgs:
                console.print(f"\n[bold]System Prompt:[/bold]\n{system_msgs[-1].content}\n")
            else:
                console.print("\nNo system prompt set.\n")
            return

        # Set system prompt
        # Remove existing system messages
        self.history.messages = [m for m in self.history.messages if m.role != MessageRole.SYSTEM]
        # Add new system prompt
        self.history.messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=args))
        console.print_success("System prompt set")

    def _cmd_clear_system(self, args: str) -> None:
        """Clear system prompt."""
        self.history.messages = [m for m in self.history.messages if m.role != MessageRole.SYSTEM]
        console.print_success("System prompt cleared")

    def _handle_shell_command(self, cmd: str) -> bool:
        """
        Execute shell command.

        Args:
            cmd: Command string (without !)

        Returns:
            True if command was handled
        """
        cmd = cmd.strip()

        if not cmd:
            return False

        # Handle !! (repeat last command)
        if cmd == "!":
            if self._last_shell_cmd:
                cmd = self._last_shell_cmd
            else:
                console.print_warning("No previous command")
                return True

        # Store in history
        if cmd != self._last_shell_cmd:
            self._shell_history.append(cmd)
        self._last_shell_cmd = cmd

        console.print(f"[dim]$ {cmd}[/dim]")

        try:
            # Parse command safely - use shell=False for simple commands
            # For complex shell features (pipes, redirects), we need shell=True
            # This is intentional for interactive shell command execution
            try:
                # Try to parse as a simple command without shell
                cmd_parts = shlex.split(cmd)
                result = subprocess.run(
                    cmd_parts,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except ValueError:
                # If parsing fails (e.g., contains shell operators), use shell=True
                # This is intentional for interactive shell command execution
                result = subprocess.run(  # nosec B602
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            if result.stdout:
                console.print(result.stdout.rstrip())
            if result.stderr:
                console.print(result.stderr.rstrip(), style="red")

            if result.returncode != 0:
                console.print(f"[red]Exit code: {result.returncode}[/red]")

        except subprocess.TimeoutExpired:
            console.print_error("Command timed out after 30 seconds")
        except Exception as e:
            console.print_error(f"Command failed: {e}")

        console.print()
        return True
