"""Shared CLI bootstrap: load config, ConfigManager, overrides, API key gate."""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

import typer

from ask_llm.config.context import set_config
from ask_llm.config.loader import ConfigLoader, LoadResult
from ask_llm.config.manager import ConfigManager
from ask_llm.utils.api_key_gate import (
    PROVIDERS_WITHOUT_API_KEYS,
    UnresolvedAPIKeyError,
    api_key_is_missing_or_unresolved,
    ensure_resolved_provider_keys,
    provider_env_var_name,
)
from ask_llm.utils.console import console


def load_cli_session(
    config_path: str | Path | None = None,
) -> tuple[LoadResult, ConfigManager]:
    """
    Load default_config.yml (search paths per ConfigLoader), set global config context,
    and return LoadResult plus a fresh ConfigManager for the loaded app config.
    """
    load_result = ConfigLoader.load(config_path)
    set_config(load_result)
    config_manager = ConfigManager(load_result.app_config, load_result.unified_config)
    return load_result, config_manager


def resolve_and_prepare(
    config_manager: ConfigManager,
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    temperature: float | None = None,
) -> tuple[str, str]:
    """
    Single provider/model resolution entry for CLI commands.

    Priority:
        1. CLI --provider / --model
        2. Configured default provider / model

    Validates provider membership, applies the model/temperature overrides
    exactly once, and returns the effective pair. Follow up with
    :func:`gate_api_key_or_exit` (commands with a pre-network dry-run path
    gate after that path instead).

    Args:
        config_manager: Active config manager.
        cli_provider: Provider passed on the CLI.
        cli_model: Model passed on the CLI.
        temperature: Effective temperature (CLI flag or command config).

    Returns:
        Tuple of (resolved_provider, resolved_model).

    Raises:
        typer.Exit: If provider or model cannot be resolved.
    """
    provider = cli_provider or config_manager.config.default_provider
    if not provider:
        console.print_error(
            "No provider specified. Use --provider or configure a default provider."
        )
        raise typer.Exit(1)

    try:
        config_manager.set_provider(provider)
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e

    try:
        model = cli_model or config_manager.get_default_model()
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e

    if not model:
        console.print_error(
            "No model specified. Use --model or configure a default model for the provider."
        )
        raise typer.Exit(1)

    config_manager.apply_overrides(model=model, temperature=temperature)
    return provider, model


def _interactive_api_key_gate(
    config_manager: ConfigManager,
    provider_name: str,
    *,
    skip_api_key_check: bool = False,
) -> bool:
    """Interactive pre-flight gate for single-provider commands.

    Options (interactive TTY):
      1 — paste API key (session-only via ConfigManager override)
      2 — exit
      3 — skip check and continue (may fail at API with duplicate errors reduced elsewhere)

    Non-interactive: print hint and exit.

    Returns:
        True if the caller should run the strict resolved-key check.
        False if the user passed the skip flag or chose option 3.
    """
    if skip_api_key_check:
        return False

    # Keyless providers (local services) are exempt from the gate
    if provider_name in PROVIDERS_WITHOUT_API_KEYS:
        return False

    pc = config_manager.get_provider_config(provider_name)
    if not api_key_is_missing_or_unresolved(pc.api_key):
        return True

    env_hint = provider_env_var_name(provider_name)
    console.print_error(
        f"未检测到 {provider_name} 的有效 API 密钥(环境变量 {env_hint} 未设置, 或配置里 ${{...}} 未解析)。"
    )

    if not sys.stdin.isatty():
        console.print_error("当前为非交互式环境: 请设置环境变量或编辑配置文件后重试。")
        raise typer.Exit(1)

    console.print()
    console.print("请选择:")
    console.print("  [1] 输入 API 密钥(仅本次进程有效, 推荐)")
    console.print("  [2] 退出")
    console.print("  [3] 跳过检测并继续(不推荐: 调用 API 时可能失败)")
    choice = console.input("请输入 1 / 2 / 3(默认 2): ").strip() or "2"

    if choice == "2":
        raise typer.Exit(1)
    if choice == "3":
        console.print_warning("已跳过 API 密钥检测, 后续请求可能失败。")
        return False
    if choice == "1":
        key = getpass.getpass("API Key: ").strip()
        if api_key_is_missing_or_unresolved(key):
            console.print_error("密钥无效或为空, 已退出。")
            raise typer.Exit(1)
        config_manager.apply_overrides(api_key=key)
        # llm_engine reloads providers.yml via load_providers_config(); ${DEEPSEEK_API_KEY}
        # must resolve there too — sync session key to the conventional env var.
        os.environ[env_hint] = key
        # Invalidate any cached adapter built from the old/empty key so the next
        # call rebuilds with the new credential. See ARCHITECTURE_REVIEW.md (secrets).
        from ask_llm.utils.provider_cache import ProviderAdapterCache

        ProviderAdapterCache.clear()
        return True

    console.print_error("无效选择, 已退出。")
    raise typer.Exit(1)


def gate_api_key_or_exit(
    config_manager: ConfigManager,
    provider: str,
    *,
    skip_api_key_check: bool = False,
) -> None:
    """
    Run the interactive API-key gate, then the resolved-key check when strict.

    Args:
        config_manager: Active config manager.
        provider: Resolved provider name.
        skip_api_key_check: Skip the gate when the user passed the flag.
    """
    strict_gate = _interactive_api_key_gate(
        config_manager,
        provider,
        skip_api_key_check=skip_api_key_check,
    )
    if strict_gate:
        try:
            ensure_resolved_provider_keys(config_manager, [provider])
        except UnresolvedAPIKeyError as e:
            console.print_error(str(e))
            raise typer.Exit(1) from e


def load_pricing_with_hint(
    explicit_path: str | Path | None = None,
) -> tuple[dict, Path | None]:
    """Load providers.yml pricing and print the standard CLI hint (P4.4).

    Single home for the previously byte-identical 6-line pricing block in
    the batch/trans/paper commands.

    Returns:
        Tuple of (pricing_map, pricing_source_path_or_None).
    """
    from ask_llm.utils.pricing import load_providers_pricing

    pricing_map, pricing_source = load_providers_pricing(explicit_path)
    if pricing_source:
        console.print_info(f"API pricing loaded from: {pricing_source}")
    else:
        console.print_info(
            "No providers.yml with pricing found; token counts will still be shown, "
            "cost estimate unavailable (add pricing_per_million_tokens or use --providers-pricing)"
        )
    return pricing_map, pricing_source


def bootstrap_command(
    config_path: str | Path | None = None,
    *,
    pricing_path: str | Path | None = None,
) -> tuple[LoadResult, ConfigManager, dict, Path | None]:
    """One-call CLI bootstrap (P4.4).

    Composes :func:`load_cli_session` + :func:`load_pricing_with_hint` into the
    standard command preamble shared by trans/paper (and future commands).
    Provider/model resolution is NOT included: callers follow up with
    :func:`resolve_and_prepare` once the command-specific effective temperature
    (CLI flag or command config section) is known, then :func:`gate_api_key_or_exit`.

    Returns:
        ``(load_result, config_manager, pricing_map, pricing_source)``.
    """
    load_result, config_manager = load_cli_session(config_path)
    pricing_map, pricing_source = load_pricing_with_hint(pricing_path)
    return load_result, config_manager, pricing_map, pricing_source
