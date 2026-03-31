"""Pre-flight checks for API keys before network calls (ask / trans / batch / chat)."""

from __future__ import annotations

import getpass
import os
import sys
from typing import TYPE_CHECKING

from ask_llm.utils.console import console

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager

_INVALID_PLACEHOLDERS = frozenset(
    {
        "",
        "your-api-key-here",
        "placeholder",
    }
)


def provider_env_var_name(provider_name: str) -> str:
    """Conventional env var for a provider (e.g. deepseek -> DEEPSEEK_API_KEY)."""
    # Special case: kimi uses KIMI_CODE_API_KEY (Kimi Code API)
    if provider_name.lower() == "kimi-code":
        return "KIMI_CODE_API_KEY"
    return f"{provider_name.upper().replace('-', '_')}_API_KEY"


def api_key_is_missing_or_unresolved(api_key: str) -> bool:
    """
    True if key is empty, placeholder, or still contains unresolved ${VAR} after YAML load.
    """
    s = (api_key or "").strip()
    if not s or s.lower() in _INVALID_PLACEHOLDERS:
        return True
    if "${" in s and "}" in s:
        return True
    return False


def require_resolved_api_key(config_manager: ConfigManager, provider_name: str) -> None:
    """
    Second line of defense before batch / parallel API calls.
    Exit with a single clear message if key is still missing.
    """
    pc = config_manager.get_provider_config(provider_name)
    if api_key_is_missing_or_unresolved(pc.api_key):
        env_hint = provider_env_var_name(provider_name)
        console.print_error(
            f"API 密钥不可用: 请设置环境变量 {env_hint} 或在配置中填写 providers.{provider_name}.api_key。"
        )
        raise SystemExit(1)


def ensure_api_key_for_provider(
    config_manager: ConfigManager,
    provider_name: str,
    *,
    skip_api_key_check: bool = False,
) -> bool:
    """
    Before commands that call the API: if key is missing, prompt or exit.

    Options (interactive TTY):
      1 — paste API key (session-only via ConfigManager override)
      2 — exit
      3 — skip check and continue (may fail at API with duplicate errors reduced elsewhere)

    Non-interactive: print hint and exit.

    Returns:
        True if caller should run ``require_resolved_api_key`` before parallel/batch calls.
        False if user passed ``--skip-api-key-check`` or chose option 3 (skip interactive gate).
    """
    if skip_api_key_check:
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
        raise SystemExit(1)

    console.print()
    console.print("请选择:")
    console.print("  [1] 输入 API 密钥(仅本次进程有效, 推荐)")
    console.print("  [2] 退出")
    console.print("  [3] 跳过检测并继续(不推荐: 调用 API 时可能失败)")
    choice = console.input("请输入 1 / 2 / 3(默认 2): ").strip() or "2"

    if choice == "2":
        raise SystemExit(1)
    if choice == "3":
        console.print_warning("已跳过 API 密钥检测, 后续请求可能失败。")
        return False
    if choice == "1":
        key = getpass.getpass("API Key: ").strip()
        if api_key_is_missing_or_unresolved(key):
            console.print_error("密钥无效或为空, 已退出。")
            raise SystemExit(1)
        config_manager.apply_overrides(api_key=key)
        # llm_engine reloads providers.yml via load_providers_config(); ${DEEPSEEK_API_KEY}
        # must resolve there too — sync session key to the conventional env var.
        os.environ[env_hint] = key
        return True

    console.print_error("无效选择, 已退出。")
    raise SystemExit(1)
