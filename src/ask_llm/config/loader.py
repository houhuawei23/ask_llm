"""Configuration loading orchestration.

Parameter priority: CLI args > environment variables > user config > package default.

Provider configuration priority:
  1. default_config.yml (user layer) — providers defined here override everything
  2. providers.yml (provider specs) — auto-loaded as fallback for provider base_url/api_key/models
  3. Package built-in default_config.yml — general defaults only, no built-in providers

Supporting modules:
  - ``ask_llm.config.env`` — ``${VAR}`` expansion and ``ASK_LLM_*`` overrides
  - ``ask_llm.config.merge`` — layered deep-merge
  - ``ask_llm.config.providers_catalog`` — ``providers.yml`` runtime extraction
"""

from pathlib import Path
from typing import Any, ClassVar

import yaml
from loguru import logger

from ask_llm.config.env import _apply_env_overrides, resolve_env_vars
from ask_llm.config.merge import _deep_merge
from ask_llm.config.providers_catalog import _load_providers_yml
from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.models import AppConfig


class LoadResult:
    """Result of loading default_config.yml, containing both provider and unified config."""

    def __init__(self, app_config: AppConfig, unified_config: UnifiedConfig, config_path: Path):
        self.app_config = app_config
        self.unified_config = unified_config
        self.config_path = config_path


class ConfigLoader:
    """Load and parse default_config.yml."""

    DEFAULT_CONFIG_FILENAME = "default_config.yml"
    DEFAULT_CONFIG_PATHS: ClassVar[list[Path]] = [
        Path(DEFAULT_CONFIG_FILENAME),
        Path.home() / ".config" / "ask_llm" / DEFAULT_CONFIG_FILENAME,
        Path("/etc/ask_llm") / DEFAULT_CONFIG_FILENAME,
    ]

    @classmethod
    def _get_package_config_path(cls) -> Path:
        """Get path to built-in default config in package."""
        return Path(__file__).parent / cls.DEFAULT_CONFIG_FILENAME

    @classmethod
    def _load_yaml(cls, path: Path) -> dict[str, Any]:
        """Load and parse a YAML config file. Resolves ${VAR} references."""
        if path.suffix not in (".yml", ".yaml"):
            raise ValueError(
                f"Unsupported config file format: {path.suffix}. "
                f"Only YAML (.yml, .yaml) files are supported."
            )
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not data:
                    data = {}
                resolved = resolve_env_vars(data)
                if not isinstance(resolved, dict):
                    resolved = {}
                return resolved
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in config file: {e}\nPlease check the syntax of {path}"
            ) from e
        except Exception as e:
            raise OSError(f"Failed to read config file: {e}") from e

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> LoadResult:
        """
        Load configuration from default_config.yml.

        Search order: --config > ./default_config.yml > ~/.config/ask_llm/ >
        /etc/ask_llm/ > package built-in.

        Args:
            config_path: Path to configuration file. If None, searches default paths.

        Returns:
            LoadResult with app_config and unified_config

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid or missing required fields
        """
        pkg_path = cls._get_package_config_path()
        user_path = cls._resolve_config_path(config_path)

        if not user_path.exists():
            cls._raise_config_not_found()

        # 1. Load package default as base
        if not pkg_path.exists():
            raise FileNotFoundError(
                f"Package default config not found at {pkg_path}. Reinstall the ask-llm package."
            )
        base_data = cls._load_yaml(pkg_path)

        # 2. Merge user config over base (if user config is different from package)
        if user_path != pkg_path:
            logger.debug(f"Loading user config from: {user_path}")
            user_data = cls._load_yaml(user_path)
            data = _deep_merge(base_data, user_data)
        else:
            data = base_data

        # 3. Merge providers.yml as fallback for provider runtime config.
        #    Priority: default_config.yml (user) > providers.yml > package built-in.
        providers_yml_data = _load_providers_yml()
        if providers_yml_data:
            # providers.yml fills in missing providers but does NOT override
            # providers already defined in default_config.yml.
            data = _deep_merge(providers_yml_data, data)

        # 4. Apply environment variable overrides
        data = _apply_env_overrides(data)

        # 5. Validate required sections
        if "providers" not in data:
            raise ValueError(
                "Config must contain 'providers' key. "
                "Please add provider configuration.\n\n"
                "  1. Run 'ask-llm config init' to generate a template in ~/.config/ask_llm/\n"
                "  2. Or use --config /path/to/default_config.yml to specify location"
            )

        if not isinstance(data["providers"], dict):
            raise ValueError("'providers' must be a dictionary")

        if not data["providers"]:
            raise ValueError(
                "Config must contain at least one provider.\n\n"
                "  1. Add providers to providers.yml in your project root, or\n"
                "  2. Add providers to your default_config.yml, or\n"
                "  3. Run 'ask-llm config init' to generate a template"
            )

        # Convert providers to canonical (api_*) shape, then validate everything
        # into a single UnifiedConfig in one pass.
        provider_data = cls._convert_providers_format(data)
        merged_data = {**data, **provider_data}

        try:
            unified_config = UnifiedConfig.model_validate(merged_data)
        except Exception as e:
            raise ValueError(
                f"Invalid configuration: {e}\n"
                f"Please check providers.api_key (use ${{ENV_VAR}} for env vars) "
                "and providers.base_url in your config"
            ) from e

        app_config = cls._app_config_from_unified(unified_config)

        logger.info(f"Configuration loaded successfully from: {user_path}")
        return LoadResult(
            app_config=app_config, unified_config=unified_config, config_path=user_path
        )

    @classmethod
    def _raise_config_not_found(cls) -> None:
        """Raise FileNotFoundError with helpful guidance."""
        searched = [str(p) for p in cls.DEFAULT_CONFIG_PATHS]
        pkg_path = cls._get_package_config_path()
        if pkg_path.exists():
            searched.append(str(pkg_path))
        raise FileNotFoundError(
            "Configuration file not found.\n\n"
            "Searched paths:\n  " + "\n  ".join(searched) + "\n\n"
            "Please create default_config.yml:\n"
            "  1. Run 'ask-llm config init' to generate a template in ~/.config/ask_llm/\n"
            "  2. Or copy from docs/default_config.example.yml in the project\n"
            "  3. Or use --config /path/to/default_config.yml to specify location"
        )

    @classmethod
    def _resolve_config_path(cls, config_path: str | Path | None = None) -> Path:
        """Resolve configuration file path."""
        if config_path:
            return Path(config_path)

        for path in cls.DEFAULT_CONFIG_PATHS:
            if path.exists():
                logger.debug(f"Found config at: {path}")
                return path

        # Fallback to package built-in
        pkg_path = cls._get_package_config_path()
        if pkg_path.exists():
            logger.debug(f"Using package default config: {pkg_path}")
            return pkg_path

        return cls.DEFAULT_CONFIG_PATHS[0]

    @classmethod
    def _convert_providers_format(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Convert providers section to canonical (api_*) shape.

        Supports models as list of dicts {"name": "..."} or list of strings.
        """
        if "providers" not in data:
            return data

        providers = data["providers"]
        if not isinstance(providers, dict):
            return data

        default_provider = data.get("default_provider")
        default_model = data.get("default_model")

        converted_providers = {}
        for name, provider_config in providers.items():
            if not isinstance(provider_config, dict):
                continue

            base_url = provider_config.get("base_url", "")
            if not base_url:
                try:
                    from llm_engine.config_loader import load_providers_config

                    providers_config = load_providers_config()
                    if providers_config and name in providers_config.get("providers", {}):
                        base_url = providers_config["providers"][name].get("base_url", "")
                except Exception:
                    pass

            if not base_url:
                raise ValueError(
                    f"Provider '{name}' has no base_url configured and it cannot be resolved "
                    f"from llm_engine. Please set base_url for provider '{name}' in your config."
                )

            converted_config = {
                "api_provider": name,
                "api_key": provider_config.get("api_key") or "",
                "api_base": base_url,
            }

            models = provider_config.get("models", [])
            provider_default_model = provider_config.get("default_model")

            if models:
                model_names = []
                for model in models:
                    if isinstance(model, dict):
                        model_name = model.get("name")
                        if model_name:
                            model_names.append(model_name)
                    elif isinstance(model, str):
                        model_names.append(model)

                if provider_default_model and provider_default_model in model_names:
                    model_names.remove(provider_default_model)
                    model_names.insert(0, provider_default_model)

                converted_config["models"] = model_names

                if not default_model and provider_default_model:
                    default_model = provider_default_model
                elif not default_model and model_names:
                    default_model = model_names[0]
            elif provider_default_model:
                converted_config["models"] = [provider_default_model]
                if not default_model:
                    default_model = provider_default_model
            else:
                converted_config["models"] = []

            if not default_provider:
                default_provider = name

            if "api_temperature" in provider_config:
                converted_config["api_temperature"] = provider_config["api_temperature"]
            if "api_top_p" in provider_config:
                converted_config["api_top_p"] = provider_config["api_top_p"]
            if "max_tokens" in provider_config:
                converted_config["max_tokens"] = provider_config["max_tokens"]
            if "timeout" in provider_config:
                converted_config["timeout"] = provider_config["timeout"]

            converted_providers[name] = converted_config

        return {
            "default_provider": default_provider,
            "default_model": default_model,
            "providers": converted_providers,
        }

    @classmethod
    def _app_config_from_unified(cls, unified_config: UnifiedConfig) -> AppConfig:
        """Derive the provider-facing AppConfig view from a validated UnifiedConfig."""
        providers = unified_config.providers
        default_provider = unified_config.default_provider
        if not default_provider:
            default_provider = next(iter(providers.keys()))
            logger.warning(f"No default_provider specified, using: {default_provider}")

        return AppConfig(
            default_provider=default_provider,
            default_model=unified_config.default_model,
            providers=providers,
        )
