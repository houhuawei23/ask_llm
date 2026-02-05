"""Interactive configuration helper for batch processing."""

import os
from pathlib import Path
from typing import List

from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import ModelConfig
from ask_llm.utils.console import console

try:
    from llm_engine import create_provider_adapter
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise


class InteractiveConfigHelper:
    """Helper for interactive configuration of models and API keys."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize interactive config helper.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

    def select_provider_and_models(self, allow_multiple: bool = True) -> List[ModelConfig]:
        """
        Interactively select provider and models.

        Args:
            allow_multiple: Whether to allow selecting multiple models

        Returns:
            List of selected ModelConfig objects
        """
        available_providers = self.config_manager.get_available_providers()

        if not available_providers:
            raise ValueError("No providers available in configuration")

        console.print()
        console.print("[bold]Available Providers:[/bold]")
        for idx, provider_name in enumerate(available_providers, 1):
            console.print(f"  {idx}. {provider_name}")

        console.print()
        provider_choice = console.input(f"Select provider (1-{len(available_providers)}): ").strip()

        try:
            provider_idx = int(provider_choice) - 1
            if provider_idx < 0 or provider_idx >= len(available_providers):
                raise ValueError("Invalid provider selection")
        except ValueError as err:
            raise ValueError(f"Invalid provider selection: {provider_choice}") from err

        selected_provider = available_providers[provider_idx]
        self.config_manager.set_provider(selected_provider)

        # Get available models for this provider
        available_models = self.config_manager.get_available_models(selected_provider)

        if not available_models:
            raise ValueError(f"No models available for provider '{selected_provider}'")

        console.print()
        console.print(f"[bold]Available Models for {selected_provider}:[/bold]")
        for idx, model_name in enumerate(available_models, 1):
            console.print(f"  {idx}. {model_name}")

        console.print()
        if allow_multiple:
            model_choice = console.input(
                "Select models (comma-separated, e.g., 1,2 or 'all'): "
            ).strip()
        else:
            model_choice = console.input(f"Select model (1-{len(available_models)}): ").strip()

        # Parse model selection
        selected_models: List[str] = []
        if model_choice.lower() == "all":
            selected_models = available_models
        else:
            try:
                if allow_multiple:
                    indices = [int(x.strip()) - 1 for x in model_choice.split(",")]
                else:
                    indices = [int(model_choice.strip()) - 1]

                for idx in indices:
                    if idx < 0 or idx >= len(available_models):
                        raise ValueError(f"Invalid model index: {idx + 1}")
                    selected_models.append(available_models[idx])
            except ValueError as e:
                raise ValueError(f"Invalid model selection: {e}") from e

        if not selected_models:
            raise ValueError("No models selected")

        # Check API keys for selected provider
        self._ensure_api_key(selected_provider)

        # Create ModelConfig objects
        model_configs = [
            ModelConfig(provider=selected_provider, model=model) for model in selected_models
        ]

        return model_configs

    def _ensure_api_key(self, provider_name: str) -> None:
        """
        Ensure API key is configured and valid for a provider.

        Args:
            provider_name: Provider name

        Raises:
            ValueError: If API key cannot be configured or is invalid
        """
        provider_config = self.config_manager.get_provider_config(provider_name)

        # Check if API key is configured
        api_key = provider_config.api_key

        if not api_key or api_key.strip() in ("", "your-api-key-here", "placeholder"):
            console.print()
            console.print_warning(f"API key not configured for provider '{provider_name}'")
            console.print_info("You can set it via environment variable or enter it now.")

            # Try to get from environment variable first
            env_var_name = f"{provider_name.upper().replace('-', '_')}_API_KEY"
            env_key = os.getenv(env_var_name)

            if env_key:
                console.print_info(f"Found API key in environment variable {env_var_name}")
                # Update config manager with the API key
                self.config_manager.apply_overrides(api_key=env_key)
                provider_config = self.config_manager.get_provider_config(provider_name)
            else:
                # Prompt user for API key
                api_key = console.input(f"Enter API key for {provider_name}: ").strip()

                if not api_key:
                    raise ValueError(f"API key is required for provider '{provider_name}'")

                # Update config manager
                self.config_manager.apply_overrides(api_key=api_key)
                provider_config = self.config_manager.get_provider_config(provider_name)

                # Ask if user wants to save to config file
                save_to_file = console.confirm(
                    "Save API key to configuration file? (not recommended for security)",
                    default=False,
                )

                if save_to_file:
                    self._save_api_key_to_config(provider_name, api_key)

        # Test API key validity
        console.print()
        console.print(f"Testing connection to {provider_name}...", end=" ")

        try:
            default_model = self.config_manager.get_default_model(provider_name)
            llm_provider = create_provider_adapter(provider_config, default_model=default_model)
            success, message, latency = llm_provider.test_connection()

            if success:
                console.print(f"[green]✓ ({latency:.2f}s)[/green]")
                logger.info(f"API key validated for {provider_name}")
            else:
                console.print("[red]✗[/red]")
                console.print_error(f"API key validation failed: {message}")
                raise ValueError(f"Invalid API key for provider '{provider_name}': {message}")

        except Exception as e:
            console.print("[red]✗[/red]")
            logger.error(f"Failed to test API key for {provider_name}: {e}")
            raise ValueError(f"Failed to validate API key for '{provider_name}': {e}") from e

    def _save_api_key_to_config(self, provider_name: str, api_key: str) -> None:
        """
        Save API key to configuration file.

        Args:
            provider_name: Provider name
            api_key: API key to save
        """
        # Try to find providers.yml file
        config_paths = [
            Path("providers.yml"),
            Path.home() / ".config" / "ask_llm" / "providers.yml",
        ]

        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break

        if not config_path:
            console.print_warning("Could not find providers.yml file to update")
            console.print_info(
                f"You can manually set the API key in your providers.yml file "
                f"or use environment variable: {provider_name.upper().replace('-', '_')}_API_KEY"
            )
            return

        try:
            import yaml

            # Read existing config
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

            # Update API key
            if "providers" not in config_data:
                config_data["providers"] = {}

            if provider_name not in config_data["providers"]:
                config_data["providers"][provider_name] = {}

            config_data["providers"][provider_name]["api_key"] = api_key

            # Write back
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            console.print_success(f"API key saved to {config_path}")
            logger.info(f"API key saved to {config_path}")

        except Exception as e:
            console.print_warning(f"Failed to save API key to config file: {e}")
            console.print_info(
                f"You can manually set the API key in your providers.yml file "
                f"or use environment variable: {provider_name.upper().replace('-', '_')}_API_KEY"
            )
