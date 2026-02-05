"""
Ask LLM - Main CLI entry point using Typer.

A flexible command-line tool for calling multiple LLM APIs.
"""

import glob
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from typing_extensions import Annotated

from ask_llm import __version__
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import GlobalBatchProcessor, ModelConfig
from ask_llm.core.chat import ChatSession
from ask_llm.core.processor import RequestProcessor
from ask_llm.core.text_splitter import TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.trans_config_loader import TransConfigLoader
from ask_llm.utils.translation_exporter import TranslationExporter

# Import from llm_engine
try:
    from llm_engine import create_provider_adapter
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise

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
    action: Annotated[str, typer.Argument(help="Action: show, test")] = "show",
    config_path: Annotated[
        Optional[str], typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
    provider: Annotated[
        Optional[str], typer.Option("--provider", "-p", help="Provider to test (with test action)")
    ] = None,
) -> None:
    """
    Manage configuration.

    Examples:
        ask-llm config show
        ask-llm config test
        ask-llm config test -p deepseek
    """
    try:
        # Load existing config
        config = ConfigLoader.load(config_path)

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
            console.print("Available actions: show, test")
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
        int,
        typer.Option(
            "--threads",
            "-t",
            help="Number of concurrent threads (default: 5)",
            min=1,
            max=50,
        ),
    ] = 5,
    retries: Annotated[
        int,
        typer.Option(
            "--retries",
            "-r",
            help="Maximum number of retries for failed tasks (default: 3)",
            min=0,
            max=10,
        ),
    ] = 3,
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
            GlobalBatchProcessor,
            ModelConfig,
        )
        from ask_llm.utils.batch_exporter import BatchResultExporter
        from ask_llm.utils.batch_loader import BatchConfigLoader
        from ask_llm.utils.interactive_config import InteractiveConfigHelper

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
            output_format = "json"  # Default format

        console.print_success(f"Loaded {len(tasks)} tasks from configuration")

        # If no models specified, use interactive selection
        if not provider_models:
            console.print_info("No models specified in configuration. Using interactive selection.")
            config = ConfigLoader.load(config_path)
            config_manager = ConfigManager(config)
            helper = InteractiveConfigHelper(config_manager)
            provider_models = helper.select_provider_and_models(allow_multiple=True)

        batch_mode = batch_config.get("mode", "prompt-content-pairs")  # Store batch mode

        # Load configuration once
        config = ConfigLoader.load(config_path)
        config_manager = ConfigManager(config)

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
                if model_config.provider not in config.providers:
                    console.print("[red]✗[/red] Provider not found")
                    skipped_providers.append(model_key)
                    continue

                # Get provider config
                provider_config = config.providers[model_config.provider]

                # Check API key
                if not provider_config.api_key or provider_config.api_key.strip() in (
                    "",
                    "your-api-key-here",
                    "placeholder",
                ):
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
                global_task = BatchTask(
                    task_id=task_id_counter,
                    prompt=original_task.prompt,
                    content=original_task.content,
                    task_model_config=model_config,
                )
                global_tasks.append(global_task)
                task_id_counter += 1

        # Step 3: Process all tasks concurrently using GlobalBatchProcessor
        global_processor = GlobalBatchProcessor(
            max_workers=threads,
            max_retries=retries,
        )

        all_results_list = global_processor.process_global_tasks(
            global_tasks, config_manager, show_progress=True
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

        # Export results
        if separate_files and len(validated_models) > 1:
            # Export to separate files per model
            output_dir = output or "batch_results"
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
                    config_file_path.parent / f"{config_file_path.stem}_results.{output_format}"
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
            help="Translation configuration file path",
        ),
    ] = None,
    target_lang: Annotated[
        str,
        typer.Option(
            "--target-lang",
            "-t",
            help="Target language code (default: zh)",
        ),
    ] = "zh",
    source_lang: Annotated[
        Optional[str],
        typer.Option(
            "--source-lang",
            "-s",
            help="Source language code (default: auto-detect)",
        ),
    ] = None,
    threads: Annotated[
        int,
        typer.Option(
            "--threads",
            "-T",
            help="Number of concurrent threads (default: 5)",
            min=1,
            max=50,
        ),
    ] = 5,
    retries: Annotated[
        int,
        typer.Option(
            "--retries",
            "-r",
            help="Maximum number of retries for failed tasks (default: 3)",
            min=0,
            max=10,
        ),
    ] = 3,
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
    trans_config_path: Annotated[
        Optional[str],
        typer.Option(
            "--trans-config",
            help="Translation configuration file path (alternative to --config)",
        ),
    ] = None,
    prompt_file: Annotated[
        Optional[str],
        typer.Option(
            "--prompt",
            "-p",
            help="Path to prompt template file (supports @ prefix for project-relative paths, e.g., @prompts/tech-paper-trans.md)",
        ),
    ] = None,
) -> None:
    """
    Translate text files using LLM API.

    Supports both plain text (.txt) and Markdown (.md) files.
    Uses intelligent text splitting to handle long documents.

    Examples:
        ask-llm trans document.txt
        ask-llm trans *.md -o translated/
        ask-llm trans file.txt -t en -s zh --threads 10
        ask-llm trans doc.md -m gpt-4 --preserve-format
        ask-llm trans paper.md -p @prompts/tech-paper-trans.md
    """
    try:
        # Load translation configuration
        trans_config_file = trans_config_path or config
        trans_config = TransConfigLoader.load(trans_config_file)
        if trans_config is None:
            trans_config = TransConfigLoader.get_default_config()

        # Override with command-line arguments
        if target_lang != "zh":
            trans_config.target_language = target_lang
        if source_lang:
            trans_config.source_language = source_lang
        if threads != 5:
            trans_config.threads = threads
        if retries != 3:
            trans_config.retries = retries
        if provider:
            trans_config.provider = provider
        if model:
            trans_config.model = model
        if prompt_file:
            trans_config.prompt_file = prompt_file

        # Load main configuration
        config_path = config if config and not trans_config_file else None
        app_config = ConfigLoader.load(config_path)
        config_manager = ConfigManager(app_config)

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

        # Set provider in config manager
        config_manager.set_provider(final_provider)
        config_manager.apply_overrides(
            model=final_model,
            temperature=trans_config.temperature,
        )

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
            console.print_error("No files found to translate")
            raise typer.Exit(1)

        # Remove duplicates and sort
        resolved_files = sorted(set(resolved_files))

        console.print_info(f"Found {len(resolved_files)} file(s) to translate")

        # Process each file
        for file_path in resolved_files:
            console.print()
            console.print(f"[bold]Processing: {file_path}[/bold]")

            # Check file type
            file_type = TextSplitter.detect_file_type(file_path)
            if file_type not in ("markdown", "text"):
                console.print_warning(
                    f"Unsupported file type: {Path(file_path).suffix}. "
                    f"Only .txt and .md files are supported. Skipping."
                )
                continue

            # Read file content
            try:
                content = FileHandler.read(file_path, show_progress=not stream)
            except Exception as e:
                console.print_error(f"Failed to read file {file_path}: {e}")
                continue

            if not content.strip():
                console.print_warning(f"File {file_path} is empty. Skipping.")
                continue

            # Split text into chunks
            splitter = TextSplitter.create_splitter(
                file_path, max_chunk_size=trans_config.max_chunk_size
            )
            chunks = splitter.split(content)

            if not chunks:
                console.print_warning(f"No chunks created from {file_path}. Skipping.")
                continue

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
            )

            # Create translation tasks
            tasks = translator.create_translation_tasks(chunks, model_config)

            # Process tasks using GlobalBatchProcessor
            processor = GlobalBatchProcessor(
                max_workers=trans_config.threads,
                max_retries=trans_config.retries,
            )

            console.print_info(
                f"Translating {len(tasks)} chunk(s) with {trans_config.threads} thread(s)..."
            )
            results = processor.process_global_tasks(
                tasks, config_manager, show_progress=not stream
            )

            # Check for failures
            failed_count = sum(1 for r in results if r.status.value == "failed")
            if failed_count > 0:
                console.print_warning(f"{failed_count} chunk(s) failed to translate")

            # Determine output path
            if output:
                output_path = output
                # If output is a directory, create file-specific name
                if Path(output).is_dir():
                    input_file = Path(file_path)
                    output_name = f"{input_file.stem}_translated{input_file.suffix}"
                    output_path = str(Path(output) / output_name)
            else:
                # Auto-generate output path
                input_file = Path(file_path)
                output_path = FileHandler.generate_output_path(file_path, suffix="_translated")

            # Export results
            exporter = TranslationExporter(
                chunks=chunks,
                results=results,
                preserve_format=preserve_format,
                include_original=False,
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
                successful = sum(1 for r in results if r.status.value == "success")
                console.print(f"  Successful: {successful}/{len(results)}")
                if failed_count > 0:
                    console.print_warning(f"  Failed: {failed_count}/{len(results)}")

            except FileExistsError:
                console.print_error(
                    f"Output file already exists: {output_path}. Use --force to overwrite."
                )
            except Exception as e:
                console.print_error(f"Failed to export translation: {e}")
                logger.exception("Export error")

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


def run_cli() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    run_cli()
