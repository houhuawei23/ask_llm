"""Configuration loading utilities.

Parameter priority: CLI args > environment variables > user config > package default.
"""

import copy
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import yaml
from loguru import logger

from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.models import AppConfig, ProviderConfig

# Environment variable to config path mapping. Env vars overlay user config.
# Format: "ASK_LLM_<SECTION>_<KEY>" -> ("section", "key") or "ASK_LLM_<KEY>" -> ("key",)
# Log each missing ${VAR} at most once per process (avoid duplicate warnings).
_WARNED_UNSET_ENV_VARS: set[str] = set()

ENV_TO_CONFIG: ClassVar[Dict[str, Tuple[str, ...]]] = {
    "ASK_LLM_DEFAULT_PROVIDER": ("default_provider",),
    "ASK_LLM_DEFAULT_MODEL": ("default_model",),
    "ASK_LLM_TRANSLATION_TARGET_LANGUAGE": ("translation", "target_language"),
    "ASK_LLM_TRANSLATION_SOURCE_LANGUAGE": ("translation", "source_language"),
    "ASK_LLM_TRANSLATION_STYLE": ("translation", "style"),
    "ASK_LLM_TRANSLATION_THREADS": ("translation", "threads"),
    "ASK_LLM_TRANSLATION_MAX_PARALLEL_FILES": ("translation", "max_parallel_files"),
    "ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS": (
        "translation",
        "max_concurrent_api_calls",
    ),
    "ASK_LLM_TRANSLATION_RETRIES": ("translation", "retries"),
    "ASK_LLM_TRANSLATION_BALANCE_CHUNK_TOKENS": ("translation", "balance_translation_chunks"),
    "ASK_LLM_TRANSLATION_MAX_CHUNK_TOKENS": ("translation", "max_chunk_tokens"),
    "ASK_LLM_TRANSLATION_MAX_OUTPUT_TOKENS": ("translation", "max_output_tokens"),
    "ASK_LLM_TRANSLATION_MIN_CHUNK_MERGE_TOKENS": ("translation", "min_chunk_merge_tokens"),
    "ASK_LLM_TRANSLATION_PRESERVE_FORMAT": ("translation", "preserve_format"),
    "ASK_LLM_TRANSLATION_INCLUDE_ORIGINAL": ("translation", "include_original"),
    "ASK_LLM_TRANSLATION_TEMPERATURE": ("translation", "temperature"),
    "ASK_LLM_TRANSLATION_DEFAULT_PROMPT_FILE": ("translation", "default_prompt_file"),
    "ASK_LLM_TRANSLATION_RECURSIVE_DIR": ("translation", "recursive_dir"),
    "ASK_LLM_BATCH_THREADS": ("batch", "threads"),
    "ASK_LLM_BATCH_RETRIES": ("batch", "retries"),
    "ASK_LLM_BATCH_RETRY_DELAY": ("batch", "retry_delay"),
    "ASK_LLM_BATCH_RETRY_DELAY_MAX": ("batch", "retry_delay_max"),
}


def _parse_env_value(value: str, key_path: Tuple[str, ...]) -> Any:
    """Parse env var string to appropriate type for the config key."""
    if value.lower() in ("null", "none", ""):
        return None
    last_key = key_path[-1].lower()
    if (
        "threads" in last_key
        or "retries" in last_key
        or "max_chunk_size" in last_key
        or "max_chunk_tokens" in last_key
        or "max_output_tokens" in last_key
        or "min_chunk_merge_tokens" in last_key
    ):
        return int(value)
    if "retry_delay" in last_key or "temperature" in last_key:
        return float(value)
    if (
        "preserve_format" in last_key
        or "include_original" in last_key
        or "recursive" in last_key
        or "balance_translation_chunks" in last_key
    ):
        return value.lower() in ("true", "1", "yes")
    return value


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge overlay into base. Overlay values take precedence.
    For 'providers', user config replaces base entirely when provided.
    """
    result = copy.deepcopy(base)
    replace_keys = {"providers"}  # These keys are replaced, not merged
    for key, overlay_val in overlay.items():
        if key in replace_keys and isinstance(overlay_val, dict):
            result[key] = copy.deepcopy(overlay_val)
        elif key in result and isinstance(result[key], dict) and isinstance(overlay_val, dict):
            result[key] = _deep_merge(result[key], overlay_val)
        else:
            result[key] = copy.deepcopy(overlay_val)
    return result


def _set_nested(data: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    """Set a nested key in data. Creates intermediate dicts as needed."""
    current = data
    for part in path[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[path[-1]] = value


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ASK_LLM_* environment variable overrides to config data."""
    result = copy.deepcopy(data)
    legacy_chunk = os.getenv("ASK_LLM_TRANSLATION_MAX_CHUNK_SIZE")
    if legacy_chunk is not None and legacy_chunk != "":
        logger.warning(
            "ASK_LLM_TRANSLATION_MAX_CHUNK_SIZE is deprecated and ignored; use "
            "ASK_LLM_TRANSLATION_MAX_CHUNK_TOKENS for token-based chunking."
        )
    for env_var, key_path in ENV_TO_CONFIG.items():
        env_val = os.getenv(env_var)
        if env_val is not None and env_val != "":
            try:
                parsed = _parse_env_value(env_val, key_path)
                _set_nested(result, key_path, parsed)
                logger.debug(f"Config override from {env_var}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid env {env_var}={env_val!r}: {e}")
    return result


def resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variable references.

    Supports ${VAR_NAME} format environment variable references.

    Args:
        value: Value that may contain environment variable references

    Returns:
        Resolved value
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} format
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)

        if matches:
            for var_name in matches:
                env_value = os.getenv(var_name)
                if env_value:
                    value = value.replace(f"${{{var_name}}}", env_value)
                else:
                    if var_name not in _WARNED_UNSET_ENV_VARS:
                        _WARNED_UNSET_ENV_VARS.add(var_name)
                        logger.warning(f"Environment variable {var_name} not set")

        return value
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    else:
        return value


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
    def _load_yaml(cls, path: Path) -> Dict[str, Any]:
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
                return resolve_env_vars(data)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in config file: {e}\nPlease check the syntax of {path}"
            ) from e
        except Exception as e:
            raise OSError(f"Failed to read config file: {e}") from e

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> LoadResult:
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
                f"Package default config not found at {pkg_path}. " "Reinstall the ask-llm package."
            )
        base_data = cls._load_yaml(pkg_path)

        # 2. Merge user config over base (if user config is different from package)
        if user_path != pkg_path:
            logger.debug(f"Loading user config from: {user_path}")
            user_data = cls._load_yaml(user_path)
            data = _deep_merge(base_data, user_data)
        else:
            data = base_data

        # 3. Apply environment variable overrides
        data = _apply_env_overrides(data)

        # 4. Validate required sections
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
                "Config must contain at least one provider. "
                "Please configure providers in your default_config.yml"
            )

        # Parse unified config (with defaults for missing sections)
        unified_config = UnifiedConfig.from_dict(data)

        # Convert providers to AppConfig format
        provider_data = cls._convert_providers_format(data)

        try:
            app_config = cls._parse_app_config(provider_data)
        except Exception as e:
            raise ValueError(
                f"Invalid provider configuration: {e}\n"
                f"Please check providers.api_key (use ${{ENV_VAR}} for env vars) "
                "and providers.api_base in your config"
            ) from e

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
    def _resolve_config_path(cls, config_path: Optional[Union[str, Path]] = None) -> Path:
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
    def _convert_providers_format(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert providers section to AppConfig format.

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
                base_url = "https://api.example.com/v1"

            converted_config = {
                "api_provider": name,
                "api_key": provider_config.get("api_key", ""),
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
    def _parse_app_config(cls, data: Dict[str, Any]) -> AppConfig:
        """Parse provider data into AppConfig."""
        if "providers" not in data:
            raise ValueError("Config must contain 'providers' key")

        default_provider = data.get("default_provider")
        if not default_provider:
            default_provider = next(iter(data["providers"].keys()))
            logger.warning(f"No default_provider specified, using: {default_provider}")

        default_model = data.get("default_model")

        providers = {}
        for name, config_data in data["providers"].items():
            config_data = {**config_data, "api_provider": name}
            config_data.pop("api_model", None)
            try:
                providers[name] = ProviderConfig.model_validate(config_data)
            except Exception as e:
                logger.error(f"Failed to validate config for provider '{name}': {e}")
                raise ValueError(f"Invalid config for provider '{name}': {e}") from e

        return AppConfig(
            default_provider=default_provider, default_model=default_model, providers=providers
        )
