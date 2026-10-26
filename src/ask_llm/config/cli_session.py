"""Shared CLI bootstrap: load config, ConfigManager, overrides, API key gate."""

from __future__ import annotations

import getpass
import sys
from collections.abc import Sequence
from pathlib import Path

import typer
from loguru import logger

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
from ask_llm.utils.interactive_config import apply_interactive_key
from ask_llm.utils.preflight import check_provider_pairs


def load_cli_session(
    config_path: str | Path | None = None,
) -> tuple[LoadResult, ConfigManager]:
    """
    Load default_config.yml (search paths per ConfigLoader), set global config context,
    and return LoadResult plus a fresh ConfigManager for the loaded app config.
    """
    load_result = ConfigLoader.load(config_path)
    set_config(load_result)
    config_manager = ConfigManager(load_result.unified_config)
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
    provider = cli_provider or config_manager.unified_config.default_provider
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
    # Make the effective pair visible: a stale ~/.config/ask_llm/providers.yml
    # silently changes the default model, and users could previously only see
    # it indirectly (e.g. via the preflight table).
    logger.info(f"Resolved provider/model: {provider}/{model}")
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
        # M12: single shared injection path (override + env sync + adapter
        # cache invalidation) — identical to the interactive_config flow.
        apply_interactive_key(config_manager, provider_name, key)
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


def preflight_or_exit(
    config_manager: ConfigManager,
    pairs: Sequence[tuple[str, str]],
    *,
    command: str,
    skip: bool = False,
) -> None:
    """Pre-run API health gate shared by trans / paper / format.

    Probes each (provider, model) pair with a tiny real API call
    (:func:`ask_llm.utils.preflight.check_provider_pairs`). On failure:

    - interactive TTY (not --quiet): ask continue/exit (default exit);
    - non-interactive or --quiet: warn and continue, so unattended scripts
      keep their retry-driven behavior.

    Skippable via the ``--no-preflight`` flag or ``general.preflight_check:
    false``. This is CLI-layer code: the only exits are ``typer.Exit`` here.

    Args:
        config_manager: Active config manager.
        pairs: (provider, model) pairs about to be used by the run.
        command: Command name for the log/table header.
        skip: True when the caller passed ``--no-preflight``.
    """
    if skip:
        logger.debug(f"[{command}] Preflight skipped (--no-preflight).")
        return
    unified = config_manager.unified_config
    if unified is not None and not unified.general.preflight_check:
        console.print_info(f"[{command}] API 预检已禁用 (general.preflight_check: false)。")
        return
    if not pairs:
        return

    console.print_info(f"[{command}] API 预检: 探测 {len(pairs)} 个 provider/model ...")
    outcomes = check_provider_pairs(config_manager, pairs)
    console.print_table(
        headers=["Provider", "Model", "状态", "延迟", "备注"],
        rows=[
            [
                o.provider,
                o.model,
                "✓" if o.ok else "✗",
                f"{o.latency:.2f}s" if o.ok and o.latency is not None else "-",
                o.message,
            ]
            for o in outcomes
        ],
    )

    failed = [o for o in outcomes if not o.ok]
    if not failed:
        return

    summary = "; ".join(f"{o.provider}/{o.model}: {o.message}" for o in failed)
    if sys.stdin.isatty() and not console.quiet:
        console.print_warning(f"API 预检发现 {len(failed)} 个异常端点。")
        if not console.confirm("API 预检失败, 仍要继续执行吗?", default=False):
            raise typer.Exit(1)
        console.print_warning("已选择继续: 运行中可能再次失败, 失败任务将自动重试并汇报。")
    else:
        console.print_warning(
            f"API 预检失败(非交互环境, 警告后继续): {summary} "
            f"(可加 --no-preflight 跳过预检, 或调高 translation.retries 增强重试。)"
        )
