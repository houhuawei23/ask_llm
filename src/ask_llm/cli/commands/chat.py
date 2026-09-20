"""Typer command `chat` (split from former cli.py)."""

from __future__ import annotations

from typing import Annotated

import typer

from ask_llm.cli.errors import cli_errors
from ask_llm.config.cli_session import (
    gate_api_key_or_exit,
    load_cli_session,
    resolve_and_prepare,
)
from ask_llm.core.chat import ChatSession
from ask_llm.utils.console import console
from ask_llm.utils.engine_facade import create_engine_adapter
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.prompt_resolver import resolve_prompt_or_template


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
    # M9: the outer ``except KeyboardInterrupt`` here was unreachable —
    # cli_errors already converts KI to a message + exit 1, and the REPL
    # handles its own Ctrl-C (core/chat.py). Removed the dead wrapper.
    with cli_errors("chat"):
        # Load configuration
        _load_result, config_manager = load_cli_session(config_path)

        final_provider, final_model = resolve_and_prepare(
            config_manager,
            cli_provider=provider,
            cli_model=model,
            temperature=temperature,
        )

        gate_api_key_or_exit(
            config_manager,
            final_provider,
            skip_api_key_check=skip_api_key_check,
        )

        provider_config = config_manager.get_provider_config()

        # Initialize provider using llm_engine factory
        llm_provider = create_engine_adapter(provider_config, default_model=final_model)

        # Load initial context
        initial_context = None
        if input_file:
            initial_context = FileHandler.read(input_file)
            console.print_info(f"Loaded context: {len(initial_context)} characters")

        # Load prompt template (file path or literal template string).
        # M4: @/~ paths resolve explicitly and error when missing instead
        # of silently becoming literal prompt text.
        prompt_template = resolve_prompt_or_template(prompt)

        session = ChatSession.from_initial_context(
            llm_provider,
            model=final_model,
            temperature=temperature,
            system_prompt=system,
            initial_context=initial_context,
            prompt_template=prompt_template,
            config_manager=config_manager,
        )
        session.start()
