"""Configuration context for current command - set at CLI entry, used by deep modules.

Deliberately a runtime leaf: only typing imports config.loader (TYPE_CHECKING),
so ``utils.* -> config.context`` edges can never pull the loader/engine/core
import chain (the historical config <-> utils <-> core cycle fuse).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ask_llm.config.loader import LoadResult

_current: LoadResult | None = None


def set_config(load_result: LoadResult) -> None:
    """Set the current configuration for this command."""
    global _current
    _current = load_result


def get_config_or_none() -> LoadResult | None:
    """Get the current configuration, or None if not set."""
    return _current
