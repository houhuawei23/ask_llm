"""Typer command `ask` (split from former cli.py)."""

from __future__ import annotations

from pathlib import Path

import typer
from typing_extensions import Annotated

from ask_llm.cli.errors import cli_errors
from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.processor import RequestProcessor
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


def ask(
    input_source: Annotated[
        str | None,
        typer.Argument(
            help="Input file path or direct text",
            show_default=False,
        ),
    ] = None,
    input_file: Annotated[
        str | None,
        typer.Option(
            "--input",
            "-i",
            help="Input file path (alternative to argument)",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (default: auto-generated)",
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Prompt template file or text (use {content} placeholder)",
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
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing output file",
        ),
    ] = False,
    metadata: Annotated[
        bool,
        typer.Option(
            "--metadata",
            help="Include metadata in output",
        ),
    ] = False,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Stream response to console",
        ),
    ] = True,
    skip_api_key_check: Annotated[
        bool,
        typer.Option(
            "--skip-api-key-check",
            help="Skip API key presence check (not recommended)",
        ),
    ] = False,
    system: Annotated[
        str | None,
        typer.Option(
            "--system",
            "-s",
            help="System prompt for the LLM",
        ),
    ] = None,
    include_reasoning: Annotated[
        bool,
        typer.Option(
            "--include-reasoning",
            help="Include reasoning content from reasoner models (e.g., DeepSeek)",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Preview prompt and token estimate without making API call",
        ),
    ] = False,
) -> None:
    """
    Send a request to LLM API with input content.

    Examples:
        ask-llm input.txt
        ask-llm "Translate to Chinese" -p "Translate: {content}"
        ask-llm input.md -o output.md -m gpt-4
        ask-llm input.txt --no-stream -o result.txt
    """
    # Resolve input source
    source = input_source or input_file
    if not source:
        console.print_error("No input provided. Use positional argument or -i/--input")
        raise typer.Exit(1)

    with cli_errors("ask"):
        # Load configuration
        load_result = ConfigLoader.load(config_path)
        set_config(load_result)
        config_manager = ConfigManager(load_result.app_config)

        # Set provider and apply overrides
        if provider:
            config_manager.set_provider(provider)

        config_manager.apply_overrides(
            model=model,
            temperature=temperature,
        )

        # Get default model (use override if set, otherwise use default from config)
        default_model = config_manager.get_model_override() or config_manager.get_default_model()

        # Get input content (needed for dry-run before API key check)
        input_path = Path(source)
        if input_path.exists() and input_path.is_file():
            content = FileHandler.read(source, show_progress=not stream)
            input_is_file = True
        else:
            content = source
            input_is_file = False

        if not content.strip():
            console.print_error("Input is empty")
            raise typer.Exit(1)

        # Load prompt template
        prompt_template = None
        if prompt:
            prompt_path = Path(prompt)
            if prompt_path.exists() and prompt_path.is_file():
                prompt_template = FileHandler.read(prompt)
            else:
                prompt_template = prompt

            # Ensure {content} placeholder exists
            if "{content}" not in prompt_template:
                prompt_template = prompt_template + "\n\n{content}"

        # Dry-run: preview prompt and token estimate (no API key required)
        if dry_run:
            from ask_llm.utils.token_counter import TokenCounter

            template = prompt_template or load_result.unified_config.general.default_prompt_template
            if "{content}" in template:
                final_prompt = template.replace("{content}", content)
            else:
                final_prompt = f"{template}\n\n{content}"
            stats = TokenCounter.estimate_tokens(final_prompt, default_model)
            console.print("[bold]--- Dry Run Preview ---[/bold]")
            console.print(f"Model: {default_model}")
            console.print(f"Estimated input: {stats['token_count']} tokens ({stats['word_count']} words)")
            if system:
                system_stats = TokenCounter.estimate_tokens(system, default_model)
                console.print(f"System prompt: {system_stats['token_count']} tokens")
            console.print(f"\nPrompt (first 500 chars):\n{final_prompt[:500]}{'...' if len(final_prompt) > 500 else ''}")
            raise typer.Exit(0)

        # API key check and provider initialization (after dry-run)
        strict_gate = ensure_api_key_for_provider(
            config_manager,
            config_manager.current_provider_name,
            skip_api_key_check=skip_api_key_check,
        )
        if strict_gate:
            require_resolved_api_key(config_manager, config_manager.current_provider_name)

        provider_config = config_manager.get_provider_config()

        # Initialize provider using llm_engine factory
        llm_provider = create_provider_adapter(provider_config, default_model=default_model)
        processor = RequestProcessor(llm_provider)

        # Determine output mode
        output_to_file = input_is_file or output

        if output_to_file:
            # Process with metadata
            result = processor.process_with_metadata(
                content=content,
                prompt_template=prompt_template,
                temperature=temperature,
                model=model,
                system_prompt=system,
                return_reasoning=include_reasoning,
            )

            # Generate output path
            if input_is_file:
                output_path = FileHandler.generate_output_path(source, output)
            else:
                default_output = load_result.unified_config.general.default_output_filename
                output_path = output or default_output

            # Prepare output content
            output_content = result.content
            if metadata and result.metadata:
                output_content = result.metadata.format() + output_content

            # Write output
            FileHandler.write(output_path, output_content, force=force)

            result.output_path = output_path
            console.print_success(f"Output saved to: {output_path}")

            if metadata and result.metadata:
                console.print(result.metadata.format())
        else:
            # Output to console
            if stream:
                if include_reasoning:
                    # Use raw stream to capture reasoning
                    from ask_llm.core.protocols import ReasoningChunk

                    final_prompt = processor._format_prompt(content, prompt_template)
                    console.print("[bold blue]Response:[/bold blue] ", end="")
                    reasoning_parts = []
                    for chunk in processor.iter_process_raw_stream(
                        final_prompt,
                        temperature=temperature,
                        model=model,
                        return_reasoning=True,
                        system_prompt=system,
                    ):
                        if isinstance(chunk, ReasoningChunk):
                            reasoning_parts.append(chunk.reasoning)
                        else:
                            console.print_stream(chunk, end="")
                    console.print()
                    if reasoning_parts:
                        console.print("[bold yellow]Reasoning:[/bold yellow]")
                        console.print("".join(reasoning_parts), style="dim")
                        console.print()
                else:
                    console.print("[bold blue]Response:[/bold blue] ", end="")
                    for chunk in processor.process(
                        content=content,
                        prompt_template=prompt_template,
                        temperature=temperature,
                        model=model,
                        stream=True,
                        system_prompt=system,
                    ):
                        console.print_stream(chunk, end="")
                    console.print()
            else:
                result = processor.process_with_metadata(
                    content=content,
                    prompt_template=prompt_template,
                    temperature=temperature,
                    model=model,
                    system_prompt=system,
                    return_reasoning=include_reasoning,
                )
                if result.reasoning:
                    console.print("[bold yellow]Reasoning:[/bold yellow]")
                    console.print(result.reasoning, style="dim")
                    console.print()
                console.print(result.content)

                if metadata and result.metadata:
                    console.print(result.metadata.format())
