"""Unified prompt file resolution and loading.

Supports explicit paths and ``@``-prefixed paths relative to the detected project root.
"""

from pathlib import Path

from loguru import logger

from ask_llm.config.context import get_config
from ask_llm.utils.file_handler import FileHandler


def resolve_prompt_file(prompt_path: str) -> Path:
    """Resolve a prompt file path.

    Paths starting with ``@`` are resolved relative to the project root, which is
    discovered using ``project_root_markers`` from the active configuration. If no
    project root is found, the current working directory is used.

    Args:
        prompt_path: Prompt file path, optionally prefixed with ``@``.

    Returns:
        Absolute, resolved path to the prompt file.
    """
    if prompt_path.startswith("@"):
        relative_path = prompt_path[1:].lstrip("/")
        current_dir = Path.cwd()
        project_root: Path | None = None
        try:
            markers = get_config().unified_config.project_root_markers
        except Exception:
            markers = []
        for marker in markers:
            for parent in [current_dir, *list(current_dir.parents)]:
                if (parent / marker).exists():
                    project_root = parent
                    break
            if project_root:
                break

        prompt_file = (
            project_root / relative_path.lstrip("/") if project_root else Path(relative_path)
        )
    else:
        prompt_file = Path(prompt_path)

    if not prompt_file.is_absolute():
        prompt_file = prompt_file.resolve()

    return prompt_file


def load_prompt_template(prompt_path: str) -> str:
    """Load and return the contents of a prompt file.

    Args:
        prompt_path: Prompt file path, optionally prefixed with ``@``.

    Returns:
        Stripped prompt template content.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        OSError: If the prompt file cannot be read.
    """
    prompt_file = resolve_prompt_file(prompt_path)

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    logger.debug(f"Loading prompt template from: {prompt_file}")
    try:
        content = FileHandler.read(str(prompt_file))
    except Exception as e:
        raise OSError(f"Failed to read prompt file {prompt_file}: {e}") from e

    return content.strip()
