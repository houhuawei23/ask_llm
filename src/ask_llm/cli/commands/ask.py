"""Typer command `ask` (split from former cli.py)."""

from __future__ import annotations

from typing import Annotated

import typer

from ask_llm.cli.errors import cli_errors
from ask_llm.config.cli_session import load_cli_session, resolve_provider_and_model_or_exit
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.protocols import ReasoningChunk
from ask_llm.services.ask_service import AskService
from ask_llm.utils.api_key_gate import (
    ensure_api_key_for_provider,
    require_resolved_api_key,
)
from ask_llm.utils.console import console
from ask_llm.utils.engine_facade import create_engine_adapter


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
    source = input_source or input_file
    if not source:
        console.print_error("No input provided. Use positional argument or -i/--input")
        raise typer.Exit(1)

    with cli_errors("ask"):
        load_result, config_manager = load_cli_session(config_path)

        _final_provider, final_model = resolve_provider_and_model_or_exit(
            config_manager,
            cli_provider=provider,
            cli_model=model,
        )

        config_manager.apply_overrides(
            model=final_model,
            temperature=temperature,
        )

        service = AskService(
            config_manager=config_manager,
            unified_config=load_result.unified_config,
            model=final_model,
        )

        content, input_is_file = service.load_content(source, show_progress=not stream)
        if not content.strip():
            console.print_error("Input is empty")
            raise typer.Exit(1)

        prompt_template = service.load_prompt_template(prompt)

        if dry_run:
            info = service.dry_run(content, prompt_template, system)
            console.print("[bold]--- Dry Run Preview ---[/bold]")
            console.print(f"Model: {info.model}")
            console.print(
                f"Estimated input: {info.estimated_input_tokens} tokens ({info.estimated_words} words)"
            )
            if info.system_prompt_tokens is not None:
                console.print(f"System prompt: {info.system_prompt_tokens} tokens")
            console.print(
                f"\nPrompt (first 500 chars):\n{info.final_prompt[:500]}"
                f"{'...' if len(info.final_prompt) > 500 else ''}"
            )
            raise typer.Exit(0)

        strict_gate = ensure_api_key_for_provider(
            config_manager,
            config_manager.current_provider_name,
            skip_api_key_check=skip_api_key_check,
        )
        if strict_gate:
            require_resolved_api_key(config_manager, config_manager.current_provider_name)

        provider_config = config_manager.get_provider_config()
        llm_provider = create_engine_adapter(provider_config, default_model=final_model)
        processor = RequestProcessor(llm_provider)
        service.set_processor(processor)

        output_to_file = input_is_file or output

        if output_to_file:
            result = service.process_to_file(
                content,
                prompt_template=prompt_template,
                system_prompt=system,
                include_metadata=metadata,
                return_reasoning=include_reasoning,
            )
            output_path = service.determine_output_path(source, input_is_file, output)
            service.write_output(output_path, result.output_content, force=force)
            console.print_success(f"Output saved to: {output_path}")
            if metadata and result.metadata:
                console.print(result.metadata.format())
        else:
            if stream:
                _stream_to_console(
                    processor,
                    content,
                    prompt_template=prompt_template,
                    temperature=temperature,
                    model=model,
                    system_prompt=system,
                    include_reasoning=include_reasoning,
                )
            else:
                processing_result = service.process(
                    content,
                    prompt_template=prompt_template,
                    system_prompt=system,
                    return_reasoning=include_reasoning,
                )
                if processing_result.reasoning:
                    console.print("[bold yellow]Reasoning:[/bold yellow]")
                    console.print(processing_result.reasoning, style="dim")
                    console.print()
                console.print(processing_result.content)
                if metadata and processing_result.metadata:
                    console.print(processing_result.metadata.format())


def _stream_to_console(
    processor: RequestProcessor,
    content: str,
    *,
    prompt_template: str | None,
    temperature: float | None,
    model: str | None,
    system_prompt: str | None,
    include_reasoning: bool,
) -> None:
    """Stream response to console, handling reasoning content if requested."""
    if include_reasoning:
        final_prompt = processor._format_prompt(content, prompt_template)
        console.print("[bold blue]Response:[/bold blue] ", end="")
        reasoning_parts: list[str] = []
        for chunk in processor.iter_process_raw_stream(
            final_prompt,
            temperature=temperature,
            model=model,
            return_reasoning=True,
            system_prompt=system_prompt,
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
            system_prompt=system_prompt,
        ):
            stream_chunk = chunk.content if isinstance(chunk, ReasoningChunk) else chunk
            console.print_stream(stream_chunk, end="")
        console.print()
