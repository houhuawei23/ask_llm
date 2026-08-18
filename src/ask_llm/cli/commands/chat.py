"""Typer command `chat` (split from former cli.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ask_llm.cli.errors import cli_errors
from ask_llm.config.cli_session import load_cli_session, resolve_provider_and_model_or_exit
from ask_llm.core.chat import ChatSession
from ask_llm.utils.api_key_gate import (
    ensure_api_key_for_provider,
    require_resolved_api_key,
)
from ask_llm.utils.console import console
from ask_llm.utils.engine_facade import create_engine_adapter
from ask_llm.utils.file_handler import FileHandler


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
            _load_result, config_manager = load_cli_session(config_path)

            final_provider, final_model = resolve_provider_and_model_or_exit(
                config_manager,
                cli_provider=provider,
                cli_model=model,
            )

            config_manager.apply_overrides(
                model=final_model,
                temperature=temperature,
            )

            strict_gate = ensure_api_key_for_provider(
                config_manager,
                final_provider,
                skip_api_key_check=skip_api_key_check,
            )
            if strict_gate:
                require_resolved_api_key(config_manager, final_provider)

            provider_config = config_manager.get_provider_config()

            # Initialize provider using llm_engine factory
            llm_provider = create_engine_adapter(provider_config, default_model=final_model)

            # Load initial context
            initial_context = None
            if input_file:
                initial_context = FileHandler.read(input_file)
                console.print_info(f"Loaded context: {len(initial_context)} characters")

            # Load prompt template (file path or literal template string)
            prompt_template = None
            if prompt:
                prompt_path = Path(prompt)
                prompt_template = FileHandler.read(prompt) if prompt_path.is_file() else prompt

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

    except KeyboardInterrupt:
        console.print("\nGoodbye!", style="green")
        raise typer.Exit(0) from None
