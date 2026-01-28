"""
Ask LLM - Main CLI entry point using Typer.

A flexible command-line tool for calling multiple LLM APIs.
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from typing_extensions import Annotated

from ask_llm import __version__
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.providers.openai_compatible import OpenAICompatibleProvider
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.chat import ChatSession
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.console import console
from ask_llm.utils.token_counter import TokenCounter

# Create Typer app
app = typer.Typer(
    name="ask-llm",
    help="Ask LLM - A flexible command-line tool for calling multiple LLM APIs",
    rich_markup_mode="rich",
    add_completion=False,
)

# Version callback
def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"Ask LLM version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        )
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Enable debug logging")
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-error output")
    ] = False,
) -> None:
    """Ask LLM - A flexible command-line tool for calling multiple LLM APIs."""
    console.setup(quiet=quiet, debug=debug)


@app.command()
def ask(
    input_source: Annotated[
        Optional[str],
        typer.Argument(
            help="Input file path or direct text",
            show_default=False,
        )
    ] = None,
    input_file: Annotated[
        Optional[str],
        typer.Option(
            "--input", "-i",
            help="Input file path (alternative to argument)",
        )
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output", "-o",
            help="Output file path (default: auto-generated)",
        )
    ] = None,
    prompt: Annotated[
        Optional[str],
        typer.Option(
            "--prompt", "-p",
            help="Prompt template file or text (use {content} placeholder)",
        )
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider", "-a",
            help="API provider to use",
        )
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model", "-m",
            help="Model name to use",
        )
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option(
            "--temperature", "-t",
            help="Sampling temperature (0.0-2.0)",
            min=0.0,
            max=2.0,
        )
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config", "-c",
            help="Configuration file path",
        )
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f",
            help="Overwrite existing output file",
        )
    ] = False,
    metadata: Annotated[
        bool,
        typer.Option(
            "--metadata",
            help="Include metadata in output",
        )
    ] = False,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Stream response to console",
        )
    ] = True,
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
    
    try:
        # Load configuration
        config = ConfigLoader.load(config_path)
        config_manager = ConfigManager(config)
        
        # Set provider and apply overrides
        if provider:
            config_manager.set_provider(provider)
        
        config_manager.apply_overrides(
            model=model,
            temperature=temperature,
        )
        
        provider_config = config_manager.get_provider_config()
        
        # Get default model (use override if set, otherwise use default from config)
        default_model = config_manager.get_model_override() or config_manager.get_default_model()
        
        # Initialize provider
        llm_provider = OpenAICompatibleProvider(provider_config, default_model=default_model)
        processor = RequestProcessor(llm_provider)
        
        # Get input content
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
        
        # Determine output mode
        output_to_file = input_is_file or output
        
        if output_to_file:
            # Process with metadata
            result = processor.process_with_metadata(
                content=content,
                prompt_template=prompt_template,
                temperature=temperature,
                model=model,
            )
            
            # Generate output path
            if input_is_file:
                output_path = FileHandler.generate_output_path(source, output)
            else:
                output_path = output or "output.txt"
            
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
                console.print("[bold blue]Response:[/bold blue] ", end="")
                for chunk in processor.process(
                    content=content,
                    prompt_template=prompt_template,
                    temperature=temperature,
                    model=model,
                    stream=True
                ):
                    console.print_stream(chunk, end="")
                console.print()
            else:
                result = processor.process_with_metadata(
                    content=content,
                    prompt_template=prompt_template,
                    temperature=temperature,
                    model=model,
                )
                console.print(result.content)
                
                if metadata and result.metadata:
                    console.print(result.metadata.format())
    
    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1)
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print_error(f"Unexpected error: {e}")
        raise typer.Exit(1)


@app.command()
def chat(
    input_file: Annotated[
        Optional[str],
        typer.Option(
            "--input", "-i",
            help="Input file for initial context",
        )
    ] = None,
    prompt: Annotated[
        Optional[str],
        typer.Option(
            "--prompt", "-p",
            help="Prompt template for initial context",
        )
    ] = None,
    system: Annotated[
        Optional[str],
        typer.Option(
            "--system", "-s",
            help="System prompt",
        )
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider", "-a",
            help="API provider to use",
        )
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model", "-m",
            help="Model name to use",
        )
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option(
            "--temperature", "-t",
            help="Sampling temperature (0.0-2.0)",
            min=0.0,
            max=2.0,
        )
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config", "-c",
            help="Configuration file path",
        )
    ] = None,
) -> None:
    """
    Start interactive chat session.
    
    Examples:
        ask-llm chat
        ask-llm chat -i context.txt
        ask-llm chat -s "You are a helpful assistant"
    """
    try:
        # Load configuration
        config = ConfigLoader.load(config_path)
        config_manager = ConfigManager(config)
        
        # Set provider
        if provider:
            config_manager.set_provider(provider)
        
        config_manager.apply_overrides(
            model=model,
            temperature=temperature,
        )
        
        provider_config = config_manager.get_provider_config()
        
        # Get default model (use override if set, otherwise use default from config)
        default_model = config_manager.get_model_override() or config_manager.get_default_model()
        
        llm_provider = OpenAICompatibleProvider(provider_config, default_model=default_model)
        
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
            provider=llm_provider.name,
            model=model or llm_provider.default_model
        )
        
        # Add system prompt
        if system:
            from ask_llm.core.models import MessageRole, ChatMessage
            history.messages.insert(
                0,
                ChatMessage(role=MessageRole.SYSTEM, content=system)
            )
        
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
                messages=messages,
                temperature=temperature,
                model=model,
                stream=True
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
    
    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1)
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\nGoodbye!", style="green")
        raise typer.Exit(0)


@app.command()
def config(
    action: Annotated[
        str,
        typer.Argument(help="Action: show, test, init")
    ] = "show",
    config_path: Annotated[
        Optional[str],
        typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="Provider to test (with test action)")
    ] = None,
) -> None:
    """
    Manage configuration.
    
    Examples:
        ask-llm config show
        ask-llm config test
        ask-llm config test -p deepseek
        ask-llm config init
    """
    try:
        if action == "init":
            path = config_path or "config.json"
            ConfigLoader.create_example_config(path)
            console.print_success(f"Example config created: {path}")
            return
        
        # Load existing config
        config = ConfigLoader.load(config_path)
        
        if action == "show":
            console.print("")
            console.print("[bold]Configuration:[/bold]")
            console.print(f"  Default Provider: {config.default_provider}")
            console.print()
            
            for name, pc in config.providers.items():
                default_marker = " [green]✓ default[/green]" if name == config.default_provider else ""
                console.print(f"[cyan]{name}[/cyan]{default_marker}")
                console.print(f"  API Base: {pc.api_base}")
                # Show default model: use config.default_model or first model from provider's models
                default_model = config.default_model or (pc.models[0] if pc.models else "N/A")
                console.print(f"  Default Model: {default_model}")
                if pc.models:
                    console.print(f"  Available Models: {', '.join(pc.models)}")
                console.print(f"  API Key: {'✓ Configured' if pc.api_key else '✗ Not configured'}")
                console.print()
        
        elif action == "test":
            providers_to_test = [provider] if provider else list(config.providers.keys())
            
            for name in providers_to_test:
                if name not in config.providers:
                    console.print_error(f"Provider '{name}' not found")
                    continue
                
                pc = config.providers[name]
                
                if not pc.api_key or pc.api_key == "your-api-key-here":
                    console.print_warning(f"[{name}] API key not configured")
                    continue
                
                console.print(f"\nTesting [cyan]{name}[/cyan]...", end=" ")
                
                try:
                    # Get default model for this provider
                    test_default_model = config.default_model or (pc.models[0] if pc.models else None)
                    if not test_default_model:
                        console.print(f"[red]✗[/red]")
                        console.print(f"  Error: No default model available")
                        continue
                    
                    llm_provider = OpenAICompatibleProvider(pc, default_model=test_default_model)
                    success, message, latency = llm_provider.test_connection()
                    
                    if success:
                        console.print(f"[green]✓ ({latency:.2f}s)[/green]")
                        console.print(f"  {message}")
                    else:
                        console.print(f"[red]✗[/red]")
                        console.print(f"  Error: {message}")
                
                except Exception as e:
                    console.print(f"[red]✗[/red]")
                    console.print_error(f"  {e}")
            
            console.print()
        
        else:
            console.print_error(f"Unknown action: {action}")
            console.print("Available actions: show, test, init")
            raise typer.Exit(1)
    
    except FileNotFoundError as e:
        console.print_error(str(e))
        console.print_info("Run 'ask-llm config init' to create an example config")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main_entry()
