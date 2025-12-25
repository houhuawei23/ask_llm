#!/usr/bin/env python3
"""Main CLI script for asking LLM APIs."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from providers import OpenAICompatibleProvider
from utils.logger import logger
from utils.file_handler import read_input_file, write_output_file, generate_output_path
from utils.config_loader import load_config, get_provider_config, merge_cli_overrides
from utils.token_counter import count_words, count_tokens
from utils.config_checker import check_and_print_config
from chat_mode import chat_mode


__version__ = "1.0.0"

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = "Please process the following text:\n\n{content}"


def get_provider_class(provider_name: str):
    """
    Get provider class by name.

    Args:
        provider_name: Name of the provider

    Returns:
        Provider class
    """
    # All OpenAI-compatible providers use the same class
    # The differentiation is done through configuration
    supported_providers = ["deepseek", "qwen"]

    if provider_name.lower() not in supported_providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported providers: {', '.join(supported_providers)}"
        )

    return OpenAICompatibleProvider


def format_metadata(
    provider_name: str,
    model_name: str,
    temperature: float,
    input_text: str,
    output_text: str,
    latency: float,
) -> str:
    """
    Format metadata for output file.

    Args:
        provider_name: Provider name
        model_name: Model name
        temperature: Temperature value
        input_text: Input text
        output_text: Output text
        latency: API call latency in seconds

    Returns:
        Formatted metadata string
    """
    input_words = count_words(input_text)
    input_tokens = count_tokens(input_text, model_name)
    output_words = count_words(output_text)
    output_tokens = count_tokens(output_text, model_name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metadata = f"""=== API Call Metadata ===
Provider: {provider_name}
Model: {model_name}
Temperature: {temperature}
Input Words: {input_words}
Input Tokens: {input_tokens}
Output Words: {output_words}
Output Tokens: {output_tokens}
Latency: {latency:.2f}s
Timestamp: {timestamp}
========================

"""
    return metadata


def generate_output_path_for_text(custom_path: Optional[str] = None) -> str:
    """
    Generate output file path when input is text (not a file).

    Args:
        custom_path: Custom output path (if provided)

    Returns:
        Output file path
    """
    if custom_path:
        return custom_path

    # Default to output.txt if no custom path and input is text
    return "output.txt"


def get_input_content(input_source: Optional[str]) -> str:
    """
    Get input content from file or direct text.

    Args:
        input_source: File path or direct text

    Returns:
        Input content as string
    """
    if not input_source:
        return ""

    # Check if it's a file path that exists
    input_path = Path(input_source)
    if input_path.exists() and input_path.is_file():
        # It's a file path, read it
        logger.debug(f"Reading input from file: {input_source}")
        return read_input_file(input_source)
    else:
        # Treat as direct text
        logger.debug("Treating input as direct text")
        return input_source


def load_prompt_template(prompt_input: Optional[str] = None) -> str:
    """
    Load prompt template from file or use as direct text.

    Args:
        prompt_input: Path to prompt file or prompt text (optional)

    Returns:
        Prompt template string with {content} placeholder
    """
    if prompt_input:
        # Check if it's a file path that exists
        prompt_path = Path(prompt_input)
        if prompt_path.exists() and prompt_path.is_file():
            # It's a file path, read it
            try:
                content = read_input_file(prompt_input)
                # Check if template has {content} placeholder
                if "{content}" not in content:
                    logger.warning(
                        "Prompt template doesn't contain {content} placeholder. "
                        "It will be appended to the input."
                    )
                    return content + "\n\n{content}"
                return content
            except Exception as e:
                logger.error(f"Failed to load prompt file: {e}")
                logger.info("Using default prompt template")
                return DEFAULT_PROMPT_TEMPLATE
        else:
            # Treat as direct prompt text
            logger.debug("Treating prompt argument as direct text")
            # Check if it contains {content} placeholder
            if "{content}" not in prompt_input:
                return prompt_input + "\n\n{content}"
            return prompt_input

    return DEFAULT_PROMPT_TEMPLATE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ask LLM APIs with flexible configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.txt
  %(prog)s -i input.txt
  %(prog)s "hello" -p "translate into chinese"
  %(prog)s -i "hello" -p "translate into chinese" -f
  %(prog)s input.txt -o output.txt -p prompt.txt
  %(prog)s input.txt -p "translate into chinese"
  %(prog)s input.txt -m deepseek-chat -t 0.5 -a deepseek
  %(prog)s input.md -c custom_config.json -f --include-metadata
  %(prog)s --chat
  %(prog)s input.txt --chat
        """,
    )

    # Positional argument for input file or text
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Input file path or direct text - can also use -i/--input",
    )

    # Optional input file/text argument (for backward compatibility)
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Input file path or direct text - alternative to positional argument",
    )

    # Optional arguments
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: input_name_output.txt/md for files, stdout for direct text input)",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Prompt template file path or direct prompt text (default: use built-in prompt). If file exists, reads file; otherwise treats as text.",
    )

    parser.add_argument("-m", "--model", default=None, help="Model name (overrides config default)")

    parser.add_argument(
        "-a", "--api_provider", default=None, help="API provider name (overrides config default)"
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=None,
        help="Temperature parameter (overrides config default)",
    )

    CONFIG_PATH = Path(__file__).parent / "config.json"

    parser.add_argument(
        "-c",
        "--config",
        default=CONFIG_PATH,
        help="Configuration file path (default: config.json)",
    )

    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")

    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode (only show errors)")

    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite output file if it exists"
    )

    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include detailed metadata (tokens, latency, etc.) in output file",
    )

    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")

    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check configuration file and list all providers and models",
    )

    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connection (use with --check-config or --api_provider)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logger
    logger.setup(debug=args.debug, quiet=args.quiet)

    logger.info(f"Starting Ask LLM with args: {args}")

    try:
        # Handle config check mode
        if args.check_config:
            check_and_print_config(
                config_path=args.config,
                test_api_flag=args.test_api,
                provider_name=args.api_provider,
            )
            return

        # Determine input source (positional argument takes precedence, fallback to -i)
        input_source = args.input_file or args.input

        # Load configuration
        logger.debug(f"Loading config from: {args.config}")
        config = load_config(args.config)

        # Get provider configuration
        provider_name = args.api_provider or config.get("default_provider")
        logger.debug(f"Using provider: {provider_name}")
        provider_config = get_provider_config(config, provider_name)

        # Merge CLI overrides
        cli_overrides = {
            "model": args.model,
            "temperature": args.temperature,
        }
        provider_config = merge_cli_overrides(provider_config, cli_overrides)

        # Initialize provider
        provider_class = get_provider_class(provider_name)
        provider = provider_class(provider_config)
        logger.info(f"Initialized provider: {provider_name}")

        # Get model name for metadata
        model_name = args.model or provider_config.get("api_model", "unknown")
        temperature = (
            args.temperature
            if args.temperature is not None
            else provider_config.get("api_temperature", 0.7)
        )

        # Handle chat mode
        if args.chat:
            # Load initial context if provided
            initial_context = None
            if input_source:
                initial_context = get_input_content(input_source)
                if initial_context:
                    logger.info(f"Using initial context ({len(initial_context)} characters)")

            # Load prompt template if provided
            prompt_template = None
            if args.prompt:
                prompt_template = load_prompt_template(args.prompt)

            # Enter chat mode
            chat_mode(
                provider=provider,
                provider_name=provider_name,
                model=args.model,
                temperature=temperature,
                initial_context=initial_context,
                prompt_template=prompt_template,
                config_dict=config,
                provider_config=provider_config,
            )
            return

        # Non-chat mode: require input
        if not input_source:
            raise ValueError(
                "Input is required (file path or text). Use positional argument or -i/--input option, or use --chat for interactive mode."
            )

        # Get input content (file or text)
        input_content = get_input_content(input_source)
        if not input_content:
            raise ValueError("Input content is empty")

        logger.info(f"Input content length: {len(input_content)} characters")

        # Check if input was a file (for output path generation)
        input_is_file = Path(input_source).exists() and Path(input_source).is_file()

        # Load prompt template
        prompt_template = load_prompt_template(args.prompt)
        prompt = prompt_template.format(content=input_content)
        logger.debug(f"Prompt length: {len(prompt)} characters")

        # Call API
        logger.info("Calling LLM API...")
        start_time = time.time()
        response = provider.call(prompt, temperature=temperature, model=args.model)
        latency = time.time() - start_time
        logger.info(f"Received response ({len(response)} characters) in {latency:.2f}s")

        # Determine output destination
        # If input is direct text (not a file) and no output file specified, output to console
        if not input_is_file and not args.output:
            # Output to console
            if args.include_metadata:
                metadata = format_metadata(
                    provider_name=provider_name,
                    model_name=model_name,
                    temperature=temperature,
                    input_text=input_content,
                    output_text=response,
                    latency=latency,
                )
                if not args.quiet:
                    print(metadata)
            print(response)
        else:
            # Output to file
            if input_is_file:
                output_path = generate_output_path(input_source, args.output)
            else:
                output_path = generate_output_path_for_text(args.output)
            logger.debug(f"Output path: {output_path}")

            # Prepare output content
            output_content = response
            if args.include_metadata:
                metadata = format_metadata(
                    provider_name=provider_name,
                    model_name=model_name,
                    temperature=temperature,
                    input_text=input_content,
                    output_text=response,
                    latency=latency,
                )
                output_content = metadata + response

            # Write output file
            write_output_file(output_path, output_content, force=args.force)
            logger.info(f"Output written to: {output_path}")

            if not args.quiet:
                print(f"✓ Success! Output saved to: {output_path}")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.debug:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
