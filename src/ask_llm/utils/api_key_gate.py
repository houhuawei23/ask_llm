"""Pure API-key checks shared by the CLI gate and the batch runner.

No ``typer`` / console I/O here: the interactive CLI gate lives in
``config.cli_session``; this module only classifies keys and raises
:class:`UnresolvedAPIKeyError` for service-layer fail-fast paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import SecretStr

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager

_INVALID_PLACEHOLDERS = frozenset(
    {
        "",
        "your-api-key-here",
        "placeholder",
    }
)

# Providers that run local / keyless services and must skip API-key gating.
# Single source for api_key_gate, interactive_config, and `config test`.
PROVIDERS_WITHOUT_API_KEYS = frozenset({"ollama"})


class UnresolvedAPIKeyError(ValueError):
    """Raised when a provider API key is missing/unresolved before network calls.

    A service-layer (non-CLI) signal so batch/trans/paper paths can fail fast
    with a clear message instead of shipping an unresolved ``${VAR}``
    placeholder or empty key to the provider. See ARCHITECTURE_REVIEW.md bug B3.
    """


def provider_env_var_name(provider_name: str) -> str:
    """Conventional env var for a provider (e.g. deepseek -> DEEPSEEK_API_KEY)."""
    return f"{provider_name.upper().replace('-', '_')}_API_KEY"


def api_key_is_missing_or_unresolved(api_key: str | SecretStr | None) -> bool:
    """
    True if key is empty, placeholder, or still contains unresolved ${VAR} after YAML load.

    Accepts plain strings or ``SecretStr`` (``ProviderConfig.api_key``).
    """
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()
    s = (api_key or "").strip()
    if not s or s.lower() in _INVALID_PLACEHOLDERS:
        return True
    return bool("${" in s and "}" in s)


def ensure_resolved_provider_keys(config_manager: ConfigManager, provider_names: list[str]) -> None:
    """Raise :class:`UnresolvedAPIKeyError` if any provider key is unresolved.

    Service-layer chokepoint (no ``typer.Exit`` / console output): used by
    ``run_global_batch_tasks`` and ``run_batch_from_config`` so the
    batch/trans/paper paths fail fast with a single clear error instead of
    letting an unresolved ``${VAR}`` or empty key reach the provider across
    many concurrent calls. Ollama (no key) is skipped.
    """
    for name in provider_names:
        if name in PROVIDERS_WITHOUT_API_KEYS:
            continue
        pc = config_manager.get_provider_config(name)
        if api_key_is_missing_or_unresolved(pc.api_key):
            env_hint = provider_env_var_name(name)
            raise UnresolvedAPIKeyError(
                f"Provider '{name}' API key is missing or unresolved. "
                f"Set environment variable {env_hint} or providers.{name}.api_key "
                f"before running."
            )
