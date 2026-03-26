"""Configuration context for current command - set at CLI entry, used by deep modules."""

from typing import Optional

from ask_llm.config.loader import LoadResult

_current: Optional[LoadResult] = None


def set_config(load_result: LoadResult) -> None:
    """Set the current configuration for this command."""
    global _current
    _current = load_result


def get_config() -> LoadResult:
    """
    Get the current configuration.

    Raises:
        RuntimeError: If config was not set (no command has called set_config)
    """
    if _current is None:
        raise RuntimeError(
            "Configuration not initialized. "
            "Ensure ConfigLoader.load() was called and set_config() was invoked."
        )
    return _current


def get_config_or_none() -> Optional[LoadResult]:
    """Get the current configuration, or None if not set."""
    return _current
