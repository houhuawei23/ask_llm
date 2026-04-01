"""
Ask LLM - Main CLI entry point using Typer.

A flexible command-line tool for calling multiple LLM APIs.
"""

import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import typer
from loguru import logger
from typing_extensions import Annotated

from ask_llm import __version__
from ask_llm.config.cli_session import (
    apply_cli_overrides_and_gate_api_key,
    load_cli_session,
    resolve_default_model_or_exit,
)
from ask_llm.config.context import get_config, set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import (
    BatchTask,
    ModelConfig,
    TaskStatus,
)
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.tasks.builders import build_paper_explain_task
from ask_llm.core.chat import ChatSession
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.md_heading_formatter import (
    HeadingApplier,
    HeadingExtractor,
    HeadingFormatter,
)
from ask_llm.core.paper_explain import (
    build_bundle_from_directory,
    build_bundle_from_file,
    build_explain_preamble_text,
    explain_output_filename,
    format_prompt,
    load_prompt_template,
    resolve_prompt_key,
    section_display_name,
)
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.text_splitter import TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.utils.api_key_gate import (
    api_key_is_missing_or_unresolved,
    ensure_api_key_for_provider,
    require_resolved_api_key,
)
from ask_llm.utils.chunk_balance import plain_text_chunks_by_tokens, rebalance_translation_chunks
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.notebook_translator import NotebookTranslator
from ask_llm.utils.pricing import format_cost_estimate, load_providers_pricing
from ask_llm.utils.provider_specs import (
    load_providers_model_limits,
    resolve_paper_max_tokens,
)
from ask_llm.utils.translation_exporter import TranslationExporter

# Import from llm_engine
try:
    from llm_engine import create_provider_adapter
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise


def _config_init(output_path: Optional[str] = None) -> None:
    """Generate default_config.yml template."""
    pkg_config = Path(__file__).resolve().parent / "config" / "default_config.yml"
    if not pkg_config.exists():
        console.print_error("Package default config not found")
        raise typer.Exit(1)

    if output_path:
        dest = Path(output_path)
    else:
        dest = Path.home() / ".config" / "ask_llm" / "default_config.yml"

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        console.print_warning(f"File exists: {dest}")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(0)

    try:
        content = pkg_config.read_text(encoding="utf-8")
        dest.write_text(content, encoding="utf-8")
        console.print_success(f"Configuration template written to: {dest}")
        console.print("Edit the file to set your API keys (use ${VAR} for environment variables).")
    except Exception as e:
        console.print_error(f"Failed to write config: {e}")
        raise typer.Exit(1) from e


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
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-d", help="Enable debug logging")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-error output")] = False,
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
        ),
    ] = None,
    input_file: Annotated[
        Optional[str],
        typer.Option(
            "--input",
            "-i",
            help="Input file path (alternative to argument)",
        ),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output file path (default: auto-generated)",
        ),
    ] = None,
    prompt: Annotated[
        Optional[str],
        typer.Option(
            "--prompt",
            "-p",
            help="Prompt template file or text (use {content} placeholder)",
        ),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-a",
            help="API provider to use",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name to use",
        ),
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option(
            "--temperature",
            "-t",
            help="Sampling temperature (0.0-2.0)",
            min=0.0,
            max=2.0,
        ),
    ] = None,
    config_path: Annotated[
        Optional[str],
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

        strict_gate = ensure_api_key_for_provider(
            config_manager,
            config_manager.current_provider_name,
            skip_api_key_check=skip_api_key_check,
        )
        if strict_gate:
            require_resolved_api_key(config_manager, config_manager.current_provider_name)

        provider_config = config_manager.get_provider_config()

        # Get default model (use override if set, otherwise use default from config)
        default_model = config_manager.get_model_override() or config_manager.get_default_model()

        # Initialize provider using llm_engine factory
        llm_provider = create_provider_adapter(provider_config, default_model=default_model)
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
                console.print("[bold blue]Response:[/bold blue] ", end="")
                for chunk in processor.process(
                    content=content,
                    prompt_template=prompt_template,
                    temperature=temperature,
                    model=model,
                    stream=True,
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
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print_error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@app.command()
def chat(
    input_file: Annotated[
        Optional[str],
        typer.Option(
            "--input",
            "-i",
            help="Input file for initial context",
        ),
    ] = None,
    prompt: Annotated[
        Optional[str],
        typer.Option(
            "--prompt",
            "-p",
            help="Prompt template for initial context",
        ),
    ] = None,
    system: Annotated[
        Optional[str],
        typer.Option(
            "--system",
            "-s",
            help="System prompt",
        ),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-a",
            help="API provider to use",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name to use",
        ),
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option(
            "--temperature",
            "-t",
            help="Sampling temperature (0.0-2.0)",
            min=0.0,
            max=2.0,
        ),
    ] = None,
    config_path: Annotated[
        Optional[str],
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
        default_model = config_manager.get_model_override() or config_manager.get_default_model()

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

        history = ChatHistory(provider=llm_provider.name, model=model or llm_provider.default_model)

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

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\nGoodbye!", style="green")
        raise typer.Exit(0) from None


@app.command()
def config(
    action: Annotated[
        str,
        typer.Argument(help="Action: show, test, init"),
    ] = "show",
    config_path: Annotated[
        Optional[str], typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
    provider: Annotated[
        Optional[str], typer.Option("--provider", "-p", help="Provider to test (with test action)")
    ] = None,
    output_path: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output path for init (default: ~/.config/ask_llm/default_config.yml)",
        ),
    ] = None,
) -> None:
    """
    Manage configuration.

    Examples:
        ask-llm config show
        ask-llm config test
        ask-llm config test -p deepseek
        ask-llm config init
        ask-llm config init -o ./my_config.yml
    """
    try:
        if action == "init":
            _config_init(output_path)
            return

        # Load existing config
        load_result = ConfigLoader.load(config_path)
        set_config(load_result)
        config = load_result.app_config

        if action == "show":
            console.print("")
            console.print("[bold]Configuration:[/bold]")
            console.print(f"  Default Provider: {config.default_provider}")
            console.print()

            for name, pc in config.providers.items():
                default_marker = (
                    " [green]✓ default[/green]" if name == config.default_provider else ""
                )
                console.print(f"[cyan]{name}[/cyan]{default_marker}")
                console.print(f"  API Base: {pc.api_base}")
                # Show default model: use first model from provider's models (which should be the default)
                default_model = pc.models[0] if pc.models else "N/A"
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
                    test_default_model = config.default_model or (
                        pc.models[0] if pc.models else None
                    )
                    if not test_default_model:
                        console.print("[red]✗[/red]")
                        console.print("  Error: No default model available")
                        continue

                    llm_provider = create_provider_adapter(pc, default_model=test_default_model)
                    success, message, latency = llm_provider.test_connection()

                    if success:
                        console.print(f"[green]✓ ({latency:.2f}s)[/green]")
                        console.print(f"  {message}")
                    else:
                        console.print("[red]✗[/red]")
                        console.print(f"  Error: {message}")

                except Exception as e:
                    console.print("[red]✗[/red]")
                    console.print_error(f"  {e}")

            console.print()

        else:
            console.print_error(f"Unknown action: {action}")
            console.print("Available actions: show, test, init")
            raise typer.Exit(1)

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e


@app.command()
def batch(
    config_file: Annotated[
        str,
        typer.Argument(help="Batch configuration file path (YAML format)"),
    ],
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory path (default: auto-generated)",
        ),
    ] = None,
    output_format: Annotated[
        Optional[str],
        typer.Option(
            "--format",
            "-f",
            help="Output format: json, yaml, csv, or markdown (auto-detected from file extension if not specified)",
        ),
    ] = None,
    threads: Annotated[
        Optional[int],
        typer.Option(
            "--threads",
            "-t",
            help="Number of concurrent threads (from default_config.yml if not set)",
            min=1,
            max=50,
        ),
    ] = None,
    retries: Annotated[
        Optional[int],
        typer.Option(
            "--retries",
            "-r",
            help="Maximum number of retries (from default_config.yml if not set)",
            min=0,
            max=10,
        ),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path",
        ),
    ] = None,
    separate_files: Annotated[
        bool,
        typer.Option(
            "--separate-files",
            help="Save results in separate files per model",
        ),
    ] = False,
    split: Annotated[
        bool,
        typer.Option(
            "--split",
            help="Split results into separate files (one file per task, content only)",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable verbose output with detailed API call information",
        ),
    ] = False,
    skip_api_key_check: Annotated[
        bool,
        typer.Option(
            "--skip-api-key-check",
            help="Skip API key presence check (not recommended)",
        ),
    ] = False,
) -> None:
    """
    Process batch tasks from YAML configuration file.

    Supports two configuration formats:
    1. prompt-contents.yml: Same prompt with multiple contents
    2. prompt-content-pairs.yml: Multiple (prompt, content) pairs

    Examples:
        ask-llm batch batch-examples/prompt-contents.yml
        ask-llm batch batch-examples/prompt-content-pairs.yml -o results.json -f json
        ask-llm batch config.yml --threads 10 --retries 5
    """
    try:
        from ask_llm.core.batch import (
            BatchStatistics,
            BatchTask,
            ModelConfig,
        )
        from ask_llm.utils.batch_exporter import BatchResultExporter
        from ask_llm.utils.batch_loader import BatchConfigLoader
        from ask_llm.utils.interactive_config import InteractiveConfigHelper
        from ask_llm.utils.pricing import format_cost_estimate, load_providers_pricing

        # Setup console with verbose mode
        if verbose:
            console.setup(quiet=False, debug=True)
        else:
            console.setup(quiet=False, debug=False)

        # Load configuration first (required for batch defaults)
        load_result, config_manager = load_cli_session(config_path)
        batch_cfg = load_result.unified_config.batch
        pricing_map, pricing_source = load_providers_pricing(None)
        effective_threads = threads if threads is not None else batch_cfg.threads
        effective_retries = retries if retries is not None else batch_cfg.retries

        # Load batch configuration
        console.print_info(f"Loading batch configuration from: {config_file}")
        batch_config = BatchConfigLoader.load(config_file)

        tasks = batch_config["tasks"]
        provider_models = batch_config.get("provider_models", [])

        # Auto-detect output format from file extension if not specified
        if output_format is None and output:
            output_path_obj = Path(output)
            suffix = output_path_obj.suffix.lower()
            # Map file extensions to formats
            extension_to_format = {
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".csv": "csv",
                ".md": "markdown",
                ".markdown": "markdown",
            }
            output_format = extension_to_format.get(suffix, "json")
            logger.debug(
                f"Auto-detected output format '{output_format}' from file extension '{suffix}'"
            )
        elif output_format is None:
            output_format = batch_cfg.default_output_format

        console.print_success(f"Loaded {len(tasks)} tasks from configuration")

        # If no models specified, use interactive selection
        if not provider_models:
            console.print_info("No models specified in configuration. Using interactive selection.")
            helper = InteractiveConfigHelper(config_manager)
            provider_models = helper.select_provider_and_models(allow_multiple=True)

        batch_mode = batch_config.get("mode", batch_cfg.mode)
        app_config = load_result.app_config

        unique_providers = sorted({m.provider for m in provider_models})
        for pname in unique_providers:
            strict_gate = ensure_api_key_for_provider(
                config_manager,
                pname,
                skip_api_key_check=skip_api_key_check,
            )
            if strict_gate:
                require_resolved_api_key(config_manager, pname)

        # Step 1: Validate all models and test connections
        console.print()
        console.print("[bold]Validating models and testing connections...[/bold]")
        validated_models: list[ModelConfig] = []
        skipped_providers: list[str] = []

        for model_config in provider_models:
            model_key = f"{model_config.provider}/{model_config.model}"
            console.print(f"  Checking {model_key}...", end=" ")

            try:
                # Check if provider exists
                if model_config.provider not in app_config.providers:
                    console.print("[red]✗[/red] Provider not found")
                    skipped_providers.append(model_key)
                    continue

                # Get provider config
                provider_config = app_config.providers[model_config.provider]

                # Check API key
                if api_key_is_missing_or_unresolved(provider_config.api_key):
                    console.print("[red]✗[/red] API key not configured")
                    skipped_providers.append(model_key)
                    continue

                # Check if model is available for this provider
                if provider_config.models and model_config.model not in provider_config.models:
                    console.print(
                        f"[red]✗[/red] Model not available. Available: {', '.join(provider_config.models)}"
                    )
                    skipped_providers.append(model_key)
                    continue

                # Set provider in config manager
                config_manager.set_provider(model_config.provider)

                # Apply model-specific overrides
                config_manager.apply_overrides(
                    model=model_config.model,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    top_p=model_config.top_p,
                )

                provider_config_with_overrides = config_manager.get_provider_config()
                default_model = (
                    config_manager.get_model_override() or config_manager.get_default_model()
                )

                # Test connection
                try:
                    test_provider = create_provider_adapter(
                        provider_config_with_overrides, default_model=default_model
                    )
                    success, message, latency = test_provider.test_connection()

                    if not success:
                        console.print(f"[red]✗[/red] Connection test failed: {message}")
                        skipped_providers.append(model_key)
                        continue

                    console.print(f"[green]✓ ({latency:.2f}s)[/green]")
                    validated_models.append(model_config)

                except Exception as e:
                    console.print(f"[red]✗[/red] Connection test error: {e}")
                    skipped_providers.append(model_key)
                    continue

            except Exception as e:
                console.print(f"[red]✗[/red] Error: {e}")
                skipped_providers.append(model_key)
                continue

        # Display summary of skipped providers
        if skipped_providers:
            console.print()
            console.print_warning(f"Skipped {len(skipped_providers)} provider(s):")
            for skipped in skipped_providers:
                console.print(f"  - {skipped}")

        # Check if we have any validated models
        if not validated_models:
            console.print_error("No providers were successfully validated. Cannot process tasks.")
            raise typer.Exit(1)

        # Step 2: Create global task list (each task with model_config)
        console.print()
        console.print(
            f"[bold]Processing {len(validated_models)} model(s) with {len(tasks)} task(s) each...[/bold]"
        )
        global_tasks: list[BatchTask] = []
        task_id_counter = 0

        for model_config in validated_models:
            for original_task in tasks:
                # Create a new task with task_model_config attached
                # Preserve original task_id for split mode filename generation
                # Use a composite ID: (model_index * num_tasks + original_task_id) for unique identification
                # But for split mode, we'll use original_task_id directly
                global_task = BatchTask(
                    task_id=task_id_counter,
                    prompt=original_task.prompt,
                    content=original_task.content,
                    output_filename=original_task.output_filename,  # Preserve output_filename
                    task_model_config=model_config,
                )
                global_tasks.append(global_task)
                task_id_counter += 1

        all_results_list, global_processor = run_global_batch_tasks(
            global_tasks,
            config_manager,
            max_workers=effective_threads,
            max_retries=effective_retries,
            retry_delay=batch_cfg.retry_delay,
            retry_delay_max=batch_cfg.retry_delay_max,
            verbose=verbose,
            show_progress=True,
            clamp_workers_to_task_count=False,
        )

        # Step 4: Group results by model and calculate statistics
        all_results: dict[str, list] = {}
        all_statistics: dict[str, BatchStatistics] = {}

        # Group results by model
        for result in all_results_list:
            model_key = f"{result.model_settings.provider}/{result.model_settings.model}"
            if model_key not in all_results:
                all_results[model_key] = []
            all_results[model_key].append(result)

        # Calculate statistics for each model
        model_statistics = global_processor.calculate_statistics(all_results_list)
        all_statistics = model_statistics

        # Display statistics for each model
        console.print()
        for model_key, statistics in all_statistics.items():
            console.print(f"[bold]Statistics for {model_key}:[/bold]")
            console.print(f"  Total Tasks: {statistics.total_tasks}")
            console.print(f"  Successful: {statistics.successful_tasks}")
            console.print(f"  Failed: {statistics.failed_tasks}")
            if statistics.successful_tasks > 0:
                success_rate = statistics.successful_tasks / statistics.total_tasks * 100
                console.print(f"  Success Rate: {success_rate:.1f}%")
                console.print(f"  Average Latency: {statistics.average_latency:.2f}s")
                console.print(
                    f"  Total Tokens: {statistics.total_input_tokens + statistics.total_output_tokens:,}"
                )
                parts = model_key.split("/", 1)
                prov, mod = parts[0], parts[1] if len(parts) > 1 else ""
                console.print(
                    format_cost_estimate(
                        prov,
                        mod,
                        statistics.total_input_tokens,
                        statistics.total_output_tokens,
                        pricing_map,
                        pricing_source=pricing_source,
                    )
                )

        # Display summary of skipped providers
        if skipped_providers:
            console.print()
            console.print_warning(f"Skipped {len(skipped_providers)} provider(s):")
            for skipped in skipped_providers:
                console.print(f"  - {skipped}")

        # Check if we have any results to export
        if not all_results:
            console.print_error(
                "No providers were successfully processed. Cannot generate results."
            )
            raise typer.Exit(1)

        # Handle split mode: export each task to a separate file
        if split:
            # Combine all results from all models
            combined_results = []
            for results in all_results.values():
                combined_results.extend(results)

            # In split mode with multiple models, group results by original task
            # and use the first result for each task
            # Group by (prompt, content, output_filename) to identify original tasks
            from collections import defaultdict

            task_groups: dict[tuple[str, str, str | None], list] = defaultdict(list)
            for result in combined_results:
                task_key = (result.prompt, result.content, result.output_filename)
                task_groups[task_key].append(result)

            # Use the first result for each original task
            # Sort by minimum task_id to maintain original order
            # Since task_id is assigned sequentially (0, 1, 2, ...) for each model,
            # we can recover original task index by: task_id % num_original_tasks
            num_original_tasks = len(tasks)
            combined_results = []
            for task_key in sorted(
                task_groups.keys(),
                key=lambda k: min(r.task_id % num_original_tasks for r in task_groups[k]),
            ):
                # Use the first result (lowest task_id) for each original task
                task_results = sorted(task_groups[task_key], key=lambda r: r.task_id)
                combined_results.append(task_results[0])

            # Determine output directory
            if output:
                output_dir = output
                # Validate that output is a directory (not a file)
                output_path_obj = Path(output_dir)
                if output_path_obj.exists() and output_path_obj.is_file():
                    console.print_error(
                        f"Output path '{output_dir}' is a file. "
                        "When using --split, output must be a directory."
                    )
                    raise typer.Exit(1)
            else:
                # Default output directory
                config_file_path = Path(config_file)
                output_dir = str(config_file_path.parent / batch_cfg.batch_output_dir)

            # Export split files
            exported_files = BatchResultExporter.export_split_files(
                combined_results, output_dir, batch_mode
            )
            console.print()
            console.print_success(
                f"Results exported to {len(exported_files)} files in: {output_dir}"
            )
            for file_path in exported_files:
                console.print(f"  - {file_path}")
            return

        # Export results
        if separate_files and len(validated_models) > 1:
            # Export to separate files per model
            output_dir = output or batch_cfg.batch_results_dir
            exported_files = BatchResultExporter.export_multiple_models(
                all_results, all_statistics, output_dir, output_format, batch_mode
            )
            console.print()
            console.print_success(f"Results exported to {len(exported_files)} files:")
            for file_path in exported_files:
                console.print(f"  - {file_path}")
        else:
            # Export all results to a single file
            # Combine all results
            combined_results = []
            for results in all_results.values():
                combined_results.extend(results)

            # Calculate combined statistics
            combined_stats = BatchStatistics(total_tasks=len(combined_results))
            combined_stats.successful_tasks = sum(
                stats.successful_tasks for stats in all_statistics.values()
            )
            combined_stats.failed_tasks = sum(
                stats.failed_tasks for stats in all_statistics.values()
            )
            combined_stats.total_latency = sum(
                stats.total_latency for stats in all_statistics.values()
            )
            if combined_stats.successful_tasks > 0:
                combined_stats.average_latency = (
                    combined_stats.total_latency / combined_stats.successful_tasks
                )
                combined_stats.total_input_tokens = sum(
                    stats.total_input_tokens for stats in all_statistics.values()
                )
                combined_stats.total_output_tokens = sum(
                    stats.total_output_tokens for stats in all_statistics.values()
                )

            # Generate output path
            if output:
                output_path = output
            else:
                config_file_path = Path(config_file)
                output_path = str(
                    config_file_path.parent
                    / f"{config_file_path.stem}{batch_cfg.output_suffix}.{output_format}"
                )

            exporter = BatchResultExporter(combined_results, combined_stats, batch_mode)
            exported_file = exporter.export(output_path, output_format)
            console.print()
            console.print_success(f"Results exported to: {exported_file}")

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\nBatch processing interrupted by user")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print_error(f"Unexpected error: {e}")
        logger.exception("Unexpected error in batch command")
        raise typer.Exit(1) from e


def _resolve_trans_input_paths(
    files: List[str],
    translatable_extensions: List[str],
    recursive_dir: bool,
) -> List[str]:
    """
    Resolve input paths to a list of translatable files.

    Supports: directory (expands to matching files), file path, glob pattern.
    """
    resolved: List[str] = []
    for pattern in files:
        p = Path(pattern)
        if p.is_dir():
            for ext in translatable_extensions:
                ext_clean = ext if ext.startswith(".") else f".{ext}"
                if recursive_dir:
                    resolved.extend(str(f) for f in p.rglob(f"*{ext_clean}"))
                else:
                    resolved.extend(str(f) for f in p.glob(f"*{ext_clean}"))
        elif p.exists() and p.is_file():
            resolved.append(str(p.resolve()))
        else:
            matched = glob.glob(pattern)
            if matched:
                for m in matched:
                    mp = Path(m)
                    if mp.is_file():
                        resolved.append(str(mp.resolve()))
            elif p.exists():
                resolved.append(str(p.resolve()))
            else:
                console.print_warning(f"File not found: {pattern}")
    return sorted(set(resolved))


def _process_notebook_translation(
    file_path: str,
    output: Optional[str],
    trans_config: Any,
    config_manager: Any,
    final_provider: str,
    final_model: str,
    force: bool,
    stream: bool,
    pricing_map: Any,
    pricing_source: Any,
) -> Optional[Tuple[int, int]]:
    """Process .ipynb notebook translation (markdown cells only). Returns (input_tokens, output_tokens) if any."""
    # Determine output path
    if output:
        output_path = output
        if Path(output).is_dir():
            input_file = Path(file_path)
            output_name = f"{input_file.stem}{get_config().unified_config.file.translated_suffix}{input_file.suffix}"
            output_path = str(Path(output) / output_name)
    else:
        output_path = FileHandler.generate_output_path(
            file_path, suffix=get_config().unified_config.file.translated_suffix
        )

    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )

    translator = Translator(
        target_language=trans_config.target_language,
        source_language=trans_config.source_language,
        style=trans_config.style,
        custom_prompt_template=trans_config.prompt_template,
        prompt_file=trans_config.prompt_file,
    )

    model_config = ModelConfig(
        provider=final_provider,
        model=final_model,
        temperature=trans_config.temperature,
        max_tokens=trans_config.max_output_tokens,
    )

    notebook_translator = NotebookTranslator(
        translator=translator,
        model_config=model_config,
    )

    try:
        successful, failed, total_in, total_out = notebook_translator.translate_notebook(
            input_path=file_path,
            output_path=output_path,
            config_manager=config_manager,
            max_workers=trans_config.threads,
            max_retries=trans_config.retries,
            show_progress=not stream,
            balance_chunks=trans_config.balance_translation_chunks,
            max_chunk_tokens=trans_config.max_chunk_tokens,
            min_chunk_merge_tokens=trans_config.min_chunk_merge_tokens,
        )
    except RuntimeError as e:
        if "API authentication failed" in str(e):
            console.print_error(str(e))
            raise typer.Exit(1) from e
        raise

    total = successful + failed
    console.print_success(f"Translation saved to: {output_path}")
    console.print(f"  Successful: {successful}/{total}")
    if failed > 0:
        console.print_warning(f"  Failed: {failed}/{total}")
    console.print(
        format_cost_estimate(
            final_provider,
            final_model,
            total_in,
            total_out,
            pricing_map,
            pricing_source=pricing_source,
        )
    )
    return (total_in, total_out)


@app.command()
def trans(
    files: Annotated[
        List[str],
        typer.Argument(help="Input file(s) to translate (supports glob patterns)"),
    ],
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory path (default: auto-generated)",
        ),
    ] = None,
    config: Annotated[
        Optional[str],
        typer.Option(
            "--config",
            "-c",
            help="Path to default_config.yml",
        ),
    ] = None,
    target_lang: Annotated[
        Optional[str],
        typer.Option(
            "--target-lang",
            "-t",
            help="Target language code (from default_config.yml if not set)",
        ),
    ] = None,
    source_lang: Annotated[
        Optional[str],
        typer.Option(
            "--source-lang",
            "-s",
            help="Source language code (default: auto-detect)",
        ),
    ] = None,
    threads: Annotated[
        Optional[int],
        typer.Option(
            "--threads",
            "-T",
            help="Per-file concurrent API calls (from default_config.yml if not set)",
            min=1,
            max=100,
        ),
    ] = None,
    max_parallel_files: Annotated[
        Optional[int],
        typer.Option(
            "--max-parallel-files",
            help="Max files to process in parallel when translating a directory (default: 3)",
            min=1,
            max=50,
        ),
    ] = None,
    retries: Annotated[
        Optional[int],
        typer.Option(
            "--retries",
            "-r",
            help="Maximum number of retries (from default_config.yml if not set)",
            min=0,
            max=10,
        ),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-a",
            help="API provider to use",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name to use",
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
    preserve_format: Annotated[
        bool,
        typer.Option(
            "--preserve-format/--no-preserve-format",
            help="Preserve original formatting (default: True)",
        ),
    ] = True,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Stream translation progress to console",
        ),
    ] = False,
    prompt_file: Annotated[
        Optional[str],
        typer.Option(
            "--prompt",
            "-p",
            help="Path to prompt template file (supports @ prefix for project-relative paths, e.g., @prompts/tech-paper-trans.md)",
        ),
    ] = None,
    providers_pricing: Annotated[
        Optional[str],
        typer.Option(
            "--providers-pricing",
            help="Path to providers.yml (pricing_per_million_tokens). "
            "Default search: ASK_LLM_PROVIDERS_YML, ./providers.yml, package root, ~/.config/ask_llm/providers.yml",
        ),
    ] = None,
    no_balance_chunks: Annotated[
        bool,
        typer.Option(
            "--no-balance-chunks",
            help="Disable token-based chunk rebalancing (structure-only splitting)",
        ),
    ] = False,
    max_chunk_tokens: Annotated[
        Optional[int],
        typer.Option(
            "--max-chunk-tokens",
            help="Max estimated body tokens per chunk after rebalance (default: config)",
            min=256,
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
    Translate text files using LLM API.

    Supports plain text (.txt), Markdown (.md), and Jupyter notebooks (.ipynb).
    For .ipynb files: only markdown cells are translated, code cells are preserved.
    Uses intelligent text splitting to handle long documents.

    Examples:
        ask-llm trans document.txt
        ask-llm trans /path/to/dir/ -o translated/
        ask-llm trans *.md -o translated/
        ask-llm trans notebook.ipynb -o translated/
        ask-llm trans file.txt -t en -s zh --threads 10
        ask-llm trans doc.md -m gpt-4 --preserve-format
        ask-llm trans paper.md -p @prompts/tech-paper-trans.md
        ask-llm trans ./posts/ --max-parallel-files 5
    """
    try:
        load_result, config_manager = load_cli_session(config)
        app_config = load_result.app_config
        trans_cfg = load_result.unified_config.translation

        pricing_map, pricing_source = load_providers_pricing(providers_pricing)
        if pricing_source:
            console.print_info(f"API pricing loaded from: {pricing_source}")
        else:
            console.print_info(
                "No providers.yml with pricing found; token counts will still be shown, "
                "cost estimate unavailable (add pricing_per_million_tokens or use --providers-pricing)"
            )

        # Build trans config from default_config.yml, override with CLI
        effective_threads = threads if threads is not None else trans_cfg.max_concurrent_api_calls
        effective_max_parallel = (
            max_parallel_files if max_parallel_files is not None else trans_cfg.max_parallel_files
        )
        trans_config = SimpleNamespace(
            target_language=target_lang or trans_cfg.target_language,
            source_language=trans_cfg.source_language if source_lang is None else source_lang,
            style=trans_cfg.style,
            threads=effective_threads,
            max_parallel_files=effective_max_parallel,
            retries=retries if retries is not None else trans_cfg.retries,
            balance_translation_chunks=trans_cfg.balance_translation_chunks
            and not no_balance_chunks,
            max_chunk_tokens=max_chunk_tokens
            if max_chunk_tokens is not None
            else trans_cfg.max_chunk_tokens,
            min_chunk_merge_tokens=trans_cfg.min_chunk_merge_tokens,
            max_output_tokens=trans_cfg.max_output_tokens,
            preserve_format=preserve_format,
            include_original=trans_cfg.include_original,
            provider=provider,
            model=model,
            prompt_file=prompt_file,
            prompt_template=None,
            temperature=trans_cfg.temperature,
            translatable_extensions=trans_cfg.translatable_extensions,
            recursive_dir=trans_cfg.recursive_dir,
        )

        # Determine provider and model
        final_provider = trans_config.provider or app_config.default_provider
        final_model = trans_config.model or config_manager.get_default_model()

        if not final_provider:
            console.print_error(
                "No provider specified. Use --provider or configure default provider."
            )
            raise typer.Exit(1)

        if not final_model:
            console.print_error("No model specified. Use --model or configure default model.")
            raise typer.Exit(1)

        apply_cli_overrides_and_gate_api_key(
            config_manager,
            provider=final_provider,
            model=final_model,
            temperature=trans_config.temperature,
            skip_api_key_check=skip_api_key_check,
        )

        # Resolve file patterns (supports directory, glob, and file paths)
        resolved_files = _resolve_trans_input_paths(
            files,
            translatable_extensions=trans_config.translatable_extensions,
            recursive_dir=trans_config.recursive_dir,
        )

        if not resolved_files:
            console.print_error("No files found to translate")
            raise typer.Exit(1)

        console.print_info(f"Found {len(resolved_files)} file(s) to translate")
        if trans_config.max_parallel_files > 1:
            console.print_info(
                f"Processing with max {trans_config.max_parallel_files} file(s) in parallel"
            )

        def _process_one_file(file_path: str) -> Optional[Tuple[int, int]]:
            """Process a single file (md, txt, or ipynb). Returns (input_tokens, output_tokens) for session total."""
            console.print()
            console.print(f"[bold]Processing: {file_path}[/bold]")

            # Check file type (auto-detect by extension)
            file_type = TextSplitter.detect_file_type(file_path)
            if file_type not in ("markdown", "text", "notebook"):
                console.print_warning(
                    f"Unsupported file type: {Path(file_path).suffix}. "
                    f"Only .txt, .md, and .ipynb files are supported. Skipping."
                )
                return None

            # Handle .ipynb notebook translation (markdown cells only, code cells preserved)
            if file_type == "notebook":
                try:
                    return _process_notebook_translation(
                        file_path=file_path,
                        output=output,
                        trans_config=trans_config,
                        config_manager=config_manager,
                        final_provider=final_provider,
                        final_model=final_model,
                        force=force,
                        stream=stream,
                        pricing_map=pricing_map,
                        pricing_source=pricing_source,
                    )
                except FileExistsError:
                    console.print_error("Output file already exists. Use --force to overwrite.")
                except Exception as e:
                    console.print_error(f"Failed to translate notebook {file_path}: {e}")
                    logger.exception("Notebook translation error")
                return None

            # Read file content (for .txt and .md)
            try:
                content = FileHandler.read(file_path, show_progress=not stream)
            except Exception as e:
                console.print_error(f"Failed to read file {file_path}: {e}")
                return None

            if not content.strip():
                console.print_warning(f"File {file_path} is empty. Skipping.")
                return None

            # Split by token budget (structure-aware for Markdown)
            if TextSplitter.detect_file_type(file_path) == "markdown":
                chunks = MarkdownTokenSplitter(final_model, trans_config.max_chunk_tokens).split(
                    content
                )
            else:
                chunks = plain_text_chunks_by_tokens(
                    content, final_model, trans_config.max_chunk_tokens
                )

            if not chunks:
                console.print_warning(f"No chunks created from {file_path}. Skipping.")
                return None

            n_before = len(chunks)
            chunks = rebalance_translation_chunks(
                chunks,
                final_model,
                max_chunk_tokens=trans_config.max_chunk_tokens,
                min_merge_tokens=trans_config.min_chunk_merge_tokens,
                enabled=trans_config.balance_translation_chunks,
            )
            if len(chunks) != n_before:
                console.print_info(
                    f"Token rebalance: {n_before} -> {len(chunks)} chunk(s) "
                    f"(cap ≈ {trans_config.max_chunk_tokens} tokens)"
                )
            else:
                console.print_info(f"Split into {len(chunks)} chunk(s)")

            # Create translator
            translator = Translator(
                target_language=trans_config.target_language,
                source_language=trans_config.source_language,
                style=trans_config.style,
                custom_prompt_template=trans_config.prompt_template,
                prompt_file=trans_config.prompt_file,
            )

            # Create model config
            model_config = ModelConfig(
                provider=final_provider,
                model=final_model,
                temperature=trans_config.temperature,
                max_tokens=trans_config.max_output_tokens,
            )

            # Create translation tasks
            tasks = translator.create_translation_tasks(chunks, model_config)

            console.print_info(
                f"Translating {len(tasks)} chunk(s) with {trans_config.threads} thread(s)..."
            )
            results, processor = run_global_batch_tasks(
                tasks,
                config_manager,
                max_workers=trans_config.threads,
                max_retries=trans_config.retries,
                show_progress=not stream,
                clamp_workers_to_task_count=False,
            )

            # Check for failures
            failed_count = sum(1 for r in results if r.status.value == "failed")
            successful_chunks = sum(1 for r in results if r.status.value == "success")
            if failed_count > 0:
                console.print_warning(f"{failed_count} chunk(s) failed to translate")
            if (
                successful_chunks == 0
                and failed_count > 0
                and getattr(processor, "_auth_error_logged", False)
            ):
                console.print_error("翻译失败: API 认证错误, 未产生有效译文。")
                raise typer.Exit(1)

            # Determine output path
            if output:
                output_path = output
                # If output is a directory, create file-specific name
                if Path(output).is_dir():
                    input_file = Path(file_path)
                    translated_suffix = get_config().unified_config.file.translated_suffix
                    output_name = f"{input_file.stem}{translated_suffix}{input_file.suffix}"
                    output_path = str(Path(output) / output_name)
            else:
                # Auto-generate output path
                output_path = FileHandler.generate_output_path(
                    file_path, suffix=get_config().unified_config.file.translated_suffix
                )

            # Export results
            exporter = TranslationExporter(
                chunks=chunks,
                results=results,
                preserve_format=preserve_format,
                include_original=trans_config.include_original,
            )

            # Detect output format from extension
            output_format = None
            output_ext = Path(output_path).suffix.lower()
            if output_ext == ".json":
                output_format = "json"
            elif output_ext in (".md", ".markdown"):
                output_format = "markdown"

            try:
                # Check if output file exists
                output_file = Path(output_path)
                if output_file.exists() and not force:
                    raise FileExistsError(
                        f"Output file already exists: {output_path}. Use --force to overwrite."
                    )

                exported_path = exporter.export(output_path, format_type=output_format)
                console.print_success(f"Translation saved to: {exported_path}")

                # Display statistics
                console.print(f"  Successful: {successful_chunks}/{len(results)}")
                if failed_count > 0:
                    console.print_warning(f"  Failed: {failed_count}/{len(results)}")

                by_model = processor.calculate_statistics(results)
                model_key = f"{final_provider}/{final_model}"
                st = by_model.get(model_key)
                if st is None and by_model:
                    st = next(iter(by_model.values()))
                total_in = st.total_input_tokens if st else 0
                total_out = st.total_output_tokens if st else 0
                console.print(
                    format_cost_estimate(
                        final_provider,
                        final_model,
                        total_in,
                        total_out,
                        pricing_map,
                        pricing_source=pricing_source,
                    )
                )
                return (total_in, total_out)

            except FileExistsError:
                console.print_error(
                    f"Output file already exists: {output_path}. Use --force to overwrite."
                )
            except Exception as e:
                console.print_error(f"Failed to export translation: {e}")
                logger.exception("Export error")
            return None

        prompt_preview = Translator(
            target_language=trans_config.target_language,
            source_language=trans_config.source_language,
            style=trans_config.style,
            custom_prompt_template=trans_config.prompt_template,
            prompt_file=trans_config.prompt_file,
        )
        prompt_template_tokens = prompt_preview.count_prompt_template_tokens(final_model)
        pf = trans_config.prompt_file
        if pf:
            prompt_label = pf if pf.startswith("@") else str(Path(pf).expanduser().name)
        else:
            prompt_label = f"内置样式 ({trans_config.style})"
        console.print_info(
            f"提示词模板「{prompt_label}」指令部分(已替换语言占位、不含待译正文)≈ "
            f"{prompt_template_tokens} tokens (tiktoken · {final_model})"
        )

        # Run file processing (sequential or parallel)
        session_in = 0
        session_out = 0
        if trans_config.max_parallel_files <= 1:
            for file_path in resolved_files:
                usage = _process_one_file(file_path)
                if usage:
                    session_in += usage[0]
                    session_out += usage[1]
        else:
            with ThreadPoolExecutor(max_workers=trans_config.max_parallel_files) as executor:
                futures = {executor.submit(_process_one_file, fp): fp for fp in resolved_files}
                for future in as_completed(futures):
                    try:
                        usage = future.result()
                        if usage:
                            session_in += usage[0]
                            session_out += usage[1]
                    except Exception as e:
                        file_path = futures[future]
                        console.print_error(f"Failed to translate {file_path}: {e}")
                        logger.exception("Translation error")

        if len(resolved_files) > 1:
            console.print()
            console.print("[bold]Session total (all files)[/bold]")
            console.print(
                format_cost_estimate(
                    final_provider,
                    final_model,
                    session_in,
                    session_out,
                    pricing_map,
                    pricing_source=pricing_source,
                )
            )

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\nTranslation interrupted by user")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print_error(f"Unexpected error: {e}")
        logger.exception("Unexpected error in trans command")
        raise typer.Exit(1) from e


@app.command()
def format(
    files: Annotated[
        List[str],
        typer.Argument(help="Input markdown file(s) to format (supports glob patterns)"),
    ],
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory path (default: auto-generated)",
        ),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path",
        ),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-a",
            help="API provider to use",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name to use",
        ),
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option(
            "--temperature",
            "-t",
            help="Sampling temperature (0.0-2.0)",
            min=0.0,
            max=2.0,
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
    inplace: Annotated[
        bool,
        typer.Option(
            "--inplace",
            "-i",
            help="Overwrite original file(s) in place instead of creating new file(s)",
        ),
    ] = False,
    heading_batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--heading-batch-size",
            help="Max headings per LLM API call (default 80). Reduce if output is truncated.",
        ),
    ] = None,
    heading_concurrency: Annotated[
        Optional[int],
        typer.Option(
            "--heading-concurrency",
            help="Max concurrent API calls for heading batches (default 4). Set 1 to disable.",
        ),
    ] = None,
    prompt_file: Annotated[
        Optional[str],
        typer.Option(
            "--prompt",
            "-p",
            help="Path to prompt template file (supports @ prefix for project-relative paths, e.g., @prompts/md-heading-format.md)",
        ),
    ] = None,
) -> None:
    """
    Format markdown heading hierarchy using LLM API.

    Extracts all headings from markdown files, uses LLM to infer proper heading
    levels based on numbering (1, 1.1, 1.1.1) or context, and applies the
    formatted headings back to the original text.

    Currently supports markdown (.md, .markdown) files only.

    Examples:
        ask-llm format document.md
        ask-llm format *.md -o formatted/
        ask-llm format doc.md --inplace
        ask-llm format doc.md -m gpt-4 -o formatted.md
        ask-llm format paper.md -p @prompts/md-heading-format.md
    """
    try:
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

        provider_config = config_manager.get_provider_config()

        # Get default model
        default_model = config_manager.get_model_override() or config_manager.get_default_model()

        if not default_model:
            console.print_error("No model specified. Use --model or configure default model.")
            raise typer.Exit(1)

        # Initialize provider
        llm_provider = create_provider_adapter(provider_config, default_model=default_model)
        processor = RequestProcessor(llm_provider)

        # Resolve file patterns
        resolved_files = []
        for file_pattern in files:
            matched_files = glob.glob(file_pattern)
            if not matched_files:
                # If no match, treat as literal file path
                if Path(file_pattern).exists():
                    resolved_files.append(file_pattern)
                else:
                    console.print_warning(f"File not found: {file_pattern}")
            else:
                resolved_files.extend(matched_files)

        if not resolved_files:
            console.print_error("No files found to format")
            raise typer.Exit(1)

        # Remove duplicates and sort
        resolved_files = sorted(set(resolved_files))

        console.print_info(f"Found {len(resolved_files)} file(s) to format")

        # Process each file
        successful_count = 0
        failed_count = 0

        for file_path in resolved_files:
            console.print()
            console.print(f"[bold]Processing: {file_path}[/bold]")

            # Check file type
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in (".md", ".markdown"):
                console.print_warning(
                    f"Unsupported file type: {file_ext}. "
                    f"Only .md and .markdown files are supported. Skipping."
                )
                failed_count += 1
                continue

            # Read file content
            try:
                content = FileHandler.read(file_path, show_progress=False)
            except Exception as e:
                console.print_error(f"Failed to read file {file_path}: {e}")
                failed_count += 1
                continue

            if not content.strip():
                console.print_warning(f"File {file_path} is empty. Skipping.")
                failed_count += 1
                continue

            # Extract headings
            headings = HeadingExtractor.extract(content)

            if not headings:
                console.print_warning(f"No headings found in {file_path}. Skipping.")
                failed_count += 1
                continue

            console.print_info(f"Found {len(headings)} heading(s)")

            # Format headings using LLM
            try:
                # Use custom prompt file or default from config
                fh_config = load_result.unified_config.format_heading
                default_prompt = prompt_file or fh_config.default_prompt_file
                formatter = HeadingFormatter(
                    processor=processor,
                    prompt_file=default_prompt,
                    batch_size=heading_batch_size,
                    concurrency=heading_concurrency,
                )

                formatted_headings = formatter.format_headings(headings)
            except Exception as e:
                console.print_error(f"Failed to format headings: {e}")
                logger.exception("Heading formatting error")
                failed_count += 1
                continue

            # Apply formatted headings
            try:
                applier = HeadingApplier()
                formatted_content = applier.apply(content, headings, formatted_headings)
            except Exception as e:
                console.print_error(f"Failed to apply formatted headings: {e}")
                logger.exception("Heading application error")
                failed_count += 1
                continue

            # Determine output path
            if inplace:
                output_path = file_path
            elif output:
                output_path = output
                # If output is a directory, create file-specific name
                if Path(output).is_dir():
                    input_file = Path(file_path)
                    formatted_suffix = get_config().unified_config.file.formatted_suffix
                    output_name = f"{input_file.stem}{formatted_suffix}{input_file.suffix}"
                    output_path = str(Path(output) / output_name)
            else:
                # Auto-generate output path
                output_path = FileHandler.generate_output_path(
                    file_path, suffix=get_config().unified_config.file.formatted_suffix
                )

            # Write output
            try:
                output_file = Path(output_path)
                # When inplace, always overwrite; otherwise check force
                if output_file.exists() and not force and not inplace:
                    raise FileExistsError(
                        f"Output file already exists: {output_path}. Use --force to overwrite."
                    )

                FileHandler.write(output_path, formatted_content, force=force or inplace)
                if inplace:
                    console.print_success(f"Formatted in place: {output_path}")
                else:
                    console.print_success(f"Formatted markdown saved to: {output_path}")
                console.print(f"  Formatted {len(headings)} heading(s)")
                successful_count += 1

            except FileExistsError:
                console.print_error(
                    f"Output file already exists: {output_path}. Use --force to overwrite."
                )
                failed_count += 1
            except Exception as e:
                console.print_error(f"Failed to write output file: {e}")
                logger.exception("File write error")
                failed_count += 1

        # Summary
        console.print()
        if successful_count > 0:
            console.print_success(f"Successfully formatted {successful_count} file(s)")
        if failed_count > 0:
            console.print_warning(f"Failed to format {failed_count} file(s)")

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\nFormatting interrupted by user")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print_error(f"Unexpected error: {e}")
        logger.exception("Unexpected error in format command")
        raise typer.Exit(1) from e


@app.command("paper")
def paper(
    input_path: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="Path to a paper .md file or an arxiv2md-beta output directory",
        ),
    ],
    run_mode: Annotated[
        str,
        typer.Option(
            "--run",
            "-r",
            help="sections (per-section + meta), full (whole paper), or all",
        ),
    ] = "all",
    sections: Annotated[
        Optional[str],
        typer.Option(
            "--sections",
            "-s",
            help="Comma-separated keys: meta,abstract,introduction,...,full (default: all available)",
        ),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-a", help="API provider"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model name"),
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option("--temperature", "-t", help="Sampling temperature", min=0.0, max=2.0),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option("--config", "-c", help="Configuration file path"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing explain/*.md files"),
    ] = False,
    metadata: Annotated[
        bool,
        typer.Option("--metadata", help="Include token/latency metadata in each output file"),
    ] = False,
    skip_api_key_check: Annotated[
        bool,
        typer.Option("--skip-api-key-check", help="Skip API key check"),
    ] = False,
    concurrency: Annotated[
        Optional[int],
        typer.Option(
            "--concurrency",
            "-j",
            help="Parallel LLM calls (default: paper.concurrency in config; thread pool for I/O)",
        ),
    ] = None,
) -> None:
    """
    Explain a paper: split Markdown by headings (or load arxiv2md-beta dir), call LLM per section,
    write results to ./explain/ next to the input file or directory.

    Multiple sections run in parallel via GlobalBatchProcessor (same pipeline as ``trans``) when concurrency > 1.

    Examples:
        ask-llm paper -i paper.md --run all
        ask-llm paper -i output2/20170612-Arxiv-Attention-Is-All-You-Need --run sections
        ask-llm paper -i paper.md -r full
        ask-llm paper -i paper.md -j 8
    """
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        console.print_error(f"Path not found: {path}")
        raise typer.Exit(1)

    run_norm = run_mode.strip().lower()
    if run_norm not in ("sections", "full", "all"):
        console.print_error("--run must be one of: sections, full, all")
        raise typer.Exit(1)

    section_filter: Optional[set[str]] = None
    if sections:
        section_filter = {x.strip().lower() for x in sections.split(",") if x.strip()}

    try:
        load_result, config_manager = load_cli_session(config_path)
        apply_cli_overrides_and_gate_api_key(
            config_manager,
            provider=provider,
            model=model,
            temperature=temperature,
            skip_api_key_check=skip_api_key_check,
        )
        default_model = resolve_default_model_or_exit(config_manager)
        paper_cfg = load_result.unified_config.paper
        prompt_dir = paper_cfg.prompt_dir
        out_sub = paper_cfg.output_subdir.strip() or "explain"
        workers = concurrency if concurrency is not None else paper_cfg.concurrency
        if workers < 1 or workers > 64:
            console.print_error("--concurrency / paper.concurrency must be between 1 and 64")
            raise typer.Exit(1)

        if path.is_file():
            if path.suffix.lower() not in (".md", ".markdown"):
                console.print_error("Paper file must be .md or .markdown")
                raise typer.Exit(1)
            bundle = build_bundle_from_file(path)
            explain_root = path.parent / out_sub
        elif path.is_dir():
            bundle = build_bundle_from_directory(path)
            explain_root = path / out_sub
        else:
            console.print_error(f"Not a file or directory: {path}")
            raise typer.Exit(1)

        explain_root.mkdir(parents=True, exist_ok=True)
        console.print_info(f"Output directory: {explain_root}")
        console.print_info(f"Paper title: {bundle.paper_title}")

        jobs: list[tuple[str, str]] = []

        def want(key: str) -> bool:
            if section_filter is None:
                return True
            return key in section_filter

        if run_norm in ("sections", "all"):
            if want("meta"):
                jobs.append(("meta", bundle.meta_text))
            for key in bundle.section_order:
                if not want(key):
                    continue
                body = bundle.sections.get(key)
                if not body or not str(body).strip():
                    logger.warning(f"Skipping empty section: {key}")
                    continue
                jobs.append((key, body))

        if run_norm in ("full", "all") and want("full"):
            ft = bundle.full_text.strip()
            if not ft:
                console.print_error("Full text is empty; cannot run full analysis")
                raise typer.Exit(1)
            jobs.append(("full", ft))

        if not jobs:
            console.print_error("No jobs to run (check --run and --sections, or empty sections)")
            raise typer.Exit(1)

        for idx, (key, _) in enumerate(jobs):
            out_name = explain_output_filename(idx, key)
            out_file = explain_root / out_name
            if out_file.exists() and not force:
                console.print_error(
                    f"Output exists: {out_file}. Use --force to overwrite."
                )
                raise typer.Exit(1)

        paper_max_tokens = paper_cfg.max_output_tokens
        full_model_name = (paper_cfg.full_model or "").strip() or "deepseek-reasoner"
        model_limits_map, _providers_spec_path = load_providers_model_limits()
        # Section/meta jobs use CLI --model if set; otherwise the provider default (must not be
        # None or resolve_paper_max_tokens skips per-model caps and sends paper.max_output_tokens raw).
        section_job_model = (model or default_model).strip()

        idx_to_meta: dict[int, tuple[str, str]] = {}
        paper_tasks: list[BatchTask] = []
        current_provider = config_manager.current_provider_name

        for idx, (key, body) in enumerate(jobs):
            template = load_prompt_template(prompt_dir, key)
            label = section_display_name(bundle, key)
            heading = bundle.section_headings.get(key)
            full_prompt = format_prompt(
                template,
                paper_title=bundle.paper_title,
                section_name=label,
                content=body,
                section_heading=heading,
            )
            job_model = full_model_name if key == "full" else section_job_model
            eff_max = resolve_paper_max_tokens(
                job_model, paper_max_tokens, model_limits_map
            )
            logger.debug(
                f"paper job: key={key!r} model={job_model!r} max_tokens={eff_max} "
                f"(paper.max_output_tokens={paper_max_tokens})"
            )
            idx_to_meta[idx] = (key, template)
            paper_tasks.append(
                build_paper_explain_task(
                    idx,
                    full_prompt,
                    task_model_config=ModelConfig(
                        provider=current_provider,
                        model=job_model,
                        temperature=temperature,
                        max_tokens=eff_max,
                    ),
                    output_filename=f"paper:{key}",
                    return_reasoning=(key == "full"),
                )
            )

        max_workers = max(1, min(workers, len(paper_tasks)))
        console.print_info(
            f"Paper explain: {len(paper_tasks)} job(s), "
            f"up to {max_workers} concurrent worker(s) (GlobalBatchProcessor, same as trans)"
        )
        results, _global_processor = run_global_batch_tasks(
            paper_tasks,
            config_manager,
            max_workers=workers,
            max_retries=3,
            show_progress=True,
            clamp_workers_to_task_count=True,
        )

        failed = [r for r in results if r.status != TaskStatus.SUCCESS]
        if failed:
            for r in failed:
                console.print_error(
                    f"Paper job {r.task_id} failed: {r.error or 'unknown error'}"
                )
            raise typer.Exit(1)

        for result in results:
            idx = result.task_id
            key, template = idx_to_meta[idx]
            pk = resolve_prompt_key(key)
            out_name = explain_output_filename(idx, key)
            out_file = explain_root / out_name
            main_name = bundle.main_path.name if bundle.main_path else "主 Markdown"
            if key == "full":
                src_full = (
                    f"论文全文拼接（主文件：{main_name}；若存在侧车则含参考文献与附录）。"
                    f"全文解读使用模型 `{full_model_name}`，并写入 API 返回的推理内容（若存在）。"
                )
                preamble = build_explain_preamble_text(
                    bundle, key, pk, template, source_override=src_full
                )
            else:
                preamble = build_explain_preamble_text(bundle, key, pk, template)
            body_out = (result.response or "").strip()
            if result.reasoning:
                body_out = (
                    "## 推理过程（思维链）\n\n"
                    + result.reasoning.strip()
                    + "\n\n---\n\n## 正文解析\n\n"
                    + body_out
                )
            if metadata and result.metadata:
                body_out = result.metadata.format() + body_out
            text_out = preamble + body_out
            FileHandler.write(str(out_file), text_out, force=force)
            console.print_success(f"Wrote {out_file}")

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print_error(f"Unexpected error: {e}")
        logger.exception("paper command failed")
        raise typer.Exit(1) from e


def run_cli() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    run_cli()
