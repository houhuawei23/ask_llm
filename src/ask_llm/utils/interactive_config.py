"""Interactive configuration helper for batch processing."""

import os
import shutil
from pathlib import Path

import yaml
from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.config.providers_catalog import load_first_providers_yml
from ask_llm.core.batch_models import ModelConfig
from ask_llm.core.checkpoint import atomic_write_text
from ask_llm.utils.api_key_gate import (
    PROVIDERS_WITHOUT_API_KEYS,
    api_key_is_missing_or_unresolved,
    provider_env_var_name,
)
from ask_llm.utils.console import console
from ask_llm.utils.engine_facade import create_engine_adapter


def apply_interactive_key(config_manager: ConfigManager, provider_name: str, key: str) -> None:
    """Apply an interactively obtained API key (M12: single shared path).

    Does three things, all required for the key to actually take effect:
    records the ConfigManager override for *this* provider, syncs the
    conventional env var so llm-engine's providers.yml ``${VAR}`` resolution
    matches, and invalidates cached provider adapters built from the old/empty
    key (cli_session's gate did this; interactive_config previously didn't, so
    batch flows could keep calling with a stale empty-key adapter).
    """
    config_manager.apply_overrides(api_key=key)
    os.environ[provider_env_var_name(provider_name)] = key
    from ask_llm.utils.provider_cache import ProviderAdapterCache

    ProviderAdapterCache.clear()


class InteractiveConfigHelper:
    """Helper for interactive configuration of models and API keys."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize interactive config helper.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

    def select_provider_and_models(self, allow_multiple: bool = True) -> list[ModelConfig]:
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
        selected_models: list[str] = []
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
        # Keyless providers (local services) need no API key
        if provider_name in PROVIDERS_WITHOUT_API_KEYS:
            return

        provider_config = self.config_manager.get_provider_config(provider_name)

        # Check if API key is configured
        api_key = provider_config.api_key

        if api_key_is_missing_or_unresolved(api_key):
            console.print()
            console.print_warning(f"API key not configured for provider '{provider_name}'")
            console.print_info("You can set it via environment variable or enter it now.")

            # Try to get from environment variable first
            env_var_name = provider_env_var_name(provider_name)
            env_key = os.getenv(env_var_name)

            if env_key:
                console.print_info(f"Found API key in environment variable {env_var_name}")
                # Shared injection path (M12): override + env sync + adapter
                # cache invalidation.
                apply_interactive_key(self.config_manager, provider_name, env_key)
                provider_config = self.config_manager.get_provider_config(provider_name)
            else:
                # Prompt user for API key (kept in a str variable: the outer
                # ``api_key`` holds the ProviderConfig SecretStr field)
                entered_key = console.input(f"Enter API key for {provider_name}: ").strip()

                if not entered_key:
                    raise ValueError(f"API key is required for provider '{provider_name}'")

                apply_interactive_key(self.config_manager, provider_name, entered_key)
                provider_config = self.config_manager.get_provider_config(provider_name)

                # Ask if user wants to save to config file
                save_to_file = console.confirm(
                    "Save API key to configuration file? (not recommended for security)",
                    default=False,
                )

                if save_to_file:
                    self._save_api_key_to_config(provider_name, entered_key)

        # Test API key validity
        console.print()
        console.print(f"Testing connection to {provider_name}...", end=" ")

        try:
            default_model = self.config_manager.get_default_model(provider_name)
            llm_provider = create_engine_adapter(provider_config, default_model=default_model)
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
        Save API key to the user's providers.yml (never cwd, never packaged).

        The write target is always ``~/.config/ask_llm/providers.yml``: a
        ``providers.yml`` in the working directory is project-local reference
        data, and the packaged copy is a shared/possibly-symlinked file —
        neither may receive secrets. When the user file does not exist yet it
        is seeded from the resolved catalog copy so no provider data is lost.
        The write is atomic and the file is restricted to 0600.

        Args:
            provider_name: Provider name
            api_key: API key to save
        """
        config_path = Path.home() / ".config" / "ask_llm" / "providers.yml"

        try:
            if not config_path.exists():
                seeded = self._seed_user_providers_yml(config_path)
                if not seeded:
                    console.print_warning(
                        f"Could not find a providers.yml catalog to seed {config_path}"
                    )
                    console.print_info(self._manual_key_hint(provider_name))
                    return

            # Read existing config
            config_data: dict = {}
            with open(config_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_data = loaded

            # Update API key
            providers = config_data.setdefault("providers", {})
            if not isinstance(providers, dict):
                console.print_warning(f"'providers' in {config_path} is not a mapping")
                return
            provider_cfg = providers.setdefault(provider_name, {})
            if isinstance(provider_cfg, dict):
                provider_cfg["api_key"] = api_key

            # Atomic write, restricted permissions, replacing any prior file.
            payload = yaml.dump(config_data, default_flow_style=False, allow_unicode=True)
            atomic_write_text(config_path, payload, mode=0o600)

            console.print_success(f"API key saved to {config_path} (permissions 0600)")
            console.print_warning(
                "Note: YAML comments/formatting in the file may have been normalized."
            )
            logger.info(f"API key saved to {config_path}")

        except Exception as e:
            console.print_warning(f"Failed to save API key to config file: {e}")
            console.print_info(self._manual_key_hint(provider_name))

    @staticmethod
    def _seed_user_providers_yml(config_path: Path) -> bool:
        """Seed the user providers.yml from the resolved catalog copy, if any.

        Copies the file *bytes* so comments and structure survive; the caller
        then applies the key update on top.
        """
        catalog_data, catalog_path = load_first_providers_yml()
        if catalog_data is None or catalog_path is None:
            return False
        config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(catalog_path, config_path)
        return True

    @staticmethod
    def _manual_key_hint(provider_name: str) -> str:
        return (
            f"You can manually set the API key in your providers.yml file "
            f"or use environment variable: {provider_env_var_name(provider_name)}"
        )
