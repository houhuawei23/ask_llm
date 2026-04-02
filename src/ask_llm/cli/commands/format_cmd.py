"""Typer command `format_cmd` (split from former cli.py)."""

from __future__ import annotations

import glob
from pathlib import Path

import typer
from loguru import logger
from typing_extensions import Annotated

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.config.context import get_config, set_config
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager
from ask_llm.core.md_heading_formatter import (
    HeadingApplier,
    HeadingExtractor,
    HeadingFormatter,
)
from ask_llm.core.processor import RequestProcessor
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler

try:
    from llm_engine import create_provider_adapter
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise


def format_cmd(
    files: Annotated[
        list[str],
        typer.Argument(help="Input markdown file(s) to format (supports glob patterns)"),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory path (default: auto-generated)",
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
        int | None,
        typer.Option(
            "--heading-batch-size",
            help="Max headings per LLM API call (default 80). Reduce if output is truncated.",
        ),
    ] = None,
    heading_concurrency: Annotated[
        int | None,
        typer.Option(
            "--heading-concurrency",
            help="Max concurrent API calls for heading batches (default 4). Set 1 to disable.",
        ),
    ] = None,
    prompt_file: Annotated[
        str | None,
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
        raise_unexpected_cli_error("format", e)
