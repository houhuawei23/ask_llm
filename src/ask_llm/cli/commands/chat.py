"""Typer command `chat` (split from former cli.py)."""

from __future__ import annotations

from pathlib import Path

import typer
from typing_extensions import Annotated

from ask_llm.cli.errors import cli_errors
from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.chat import ChatSession
from ask_llm.utils.api_key_gate import (
    ensure_api_key_for_provider,
    require_resolved_api_key,
)
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler

try:
    from llm_engine import create_provider_adapter
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise


def chat(
    input_file: Annotated[
        str | None,
        typer.Option(
            "--input",
            "-i",
            help="Input file for initial context",
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Prompt template for initial context",
        ),
    ] = None,
    system: Annotated[
        str | None,
        typer.Option(
            "--system",
            "-s",
            help="System prompt",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-a",
            help="API provider to use",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model name to use",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            "-t",
            help="Sampling temperature (0.0-2.0)",
            min=0.0,
            max=2.0,
        ),
    ] = None,
    config_path: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path",
        ),
    ] = None,
    skip_api_key_check: Annotated[
        bool,
        typer.Option(
            "--skip-api-key-check",
            help="Skip API key presence check (not recommended)",
        ),
    ] = False,
) -> None:
    """
    Start interactive chat session.

    Examples:
        ask-llm chat
        ask-llm chat -i context.txt
        ask-llm chat -s "You are a helpful assistant"
    """
    try:
        with cli_errors("chat"):
            # Load configuration
            load_result = ConfigLoader.load(config_path)
            set_config(load_result)
            config_manager = ConfigManager(load_result.app_config)

            # Set provider
            if provider:
                config_manager.set_provider(provider)

            config_manager.apply_overrides(
                model=model,
                temperature=temperature,
            )

            strict_gate = ensure_api_key_for_provider(
                config_manager,
                config_manager.current_provider_name,
                skip_api_key_check=skip_api_key_check,
            )
            if strict_gate:
                require_resolved_api_key(config_manager, config_manager.current_provider_name)

            provider_config = config_manager.get_provider_config()

            # Get default model (use override if set, otherwise use default from config)
            default_model = (
                config_manager.get_model_override() or config_manager.get_default_model()
            )

            # Initialize provider using llm_engine factory
            llm_provider = create_provider_adapter(provider_config, default_model=default_model)

            # Load initial context
            initial_context = None
            if input_file:
                initial_context = FileHandler.read(input_file)
                console.print_info(f"Loaded context: {len(initial_context)} characters")

            # Load prompt template
            prompt_template = None
            if prompt:
                prompt_path = Path(prompt)
                if prompt_path.exists() and prompt_path.is_file():
                    prompt_template = FileHandler.read(prompt)
                else:
                    prompt_template = prompt

            # Create chat session
            from ask_llm.core.models import ChatHistory

            history = ChatHistory(
                provider=llm_provider.name, model=model or llm_provider.default_model
            )

            # Add system prompt
            if system:
                from ask_llm.core.models import ChatMessage, MessageRole

                history.messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=system))

            # Add initial context
            if initial_context:
                if prompt_template and "{content}" in prompt_template:
                    content = prompt_template.format(content=initial_context)
                else:
                    content = initial_context

                from ask_llm.core.models import MessageRole

                history.add_message(MessageRole.USER, content)

                # Get initial response
                console.print("[dim]Getting initial response...[/dim]")
                messages = history.get_messages()

                console.print("[bold blue]Assistant:[/bold blue] ", end="")
                response_parts = []

                for chunk in llm_provider.call(
                    messages=messages, temperature=temperature, model=model, stream=True
                ):
                    response_parts.append(chunk)
                    console.print_stream(chunk, end="")

                console.print("\n")

                from ask_llm.core.models import MessageRole

                history.add_message(MessageRole.ASSISTANT, "".join(response_parts))

            # Start interactive session
            session = ChatSession(
                provider=llm_provider,
                temperature=temperature,
                model=model or llm_provider.default_model,
                history=history,
                config_manager=config_manager,
            )
            session.start()

    except KeyboardInterrupt:
        console.print("\nGoodbye!", style="green")
        raise typer.Exit(0) from None
