"""Unified prompt file resolution and loading.

Supports explicit paths and ``@``-prefixed paths relative to the detected project root.
"""

from pathlib import Path

from loguru import logger

from ask_llm.config.context import get_config_or_none
from ask_llm.utils.file_handler import FileHandler

# Built-in default matching default_config.yml so @-path resolution works
# without an active CLI config (e.g. library / embedded use).
_DEFAULT_PROJECT_ROOT_MARKERS = (
    "pyproject.toml",
    "setup.py",
    ".git",
    "default_config.yml",
)

# Prompts shipped inside the package (``ask_llm/prompts``, package-data; a
# symlink to the repo ``prompts/`` in a dev checkout). Used as the last-resort
# fallback for ``@``-paths so pip-installed users get the built-in defaults
# even when their cwd is not inside an ask_llm checkout.
_PACKAGE_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def resolve_prompt_file(prompt_path: str) -> Path:
    """Resolve a prompt file path.

    Paths starting with ``@`` are resolved relative to the project root, which is
    discovered using ``project_root_markers`` from the active configuration. If no
    project root is found, the current working directory is used. When the
    resolved path does not exist, the packaged ``ask_llm/prompts/`` copy is used
    as a fallback so installed (non-checkout) environments keep working; the
    project root always wins over the packaged copy so user-customized prompts
    are never shadowed.

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
            lr = get_config_or_none()
            markers = (
                lr.unified_config.project_root_markers
                if lr is not None
                else _DEFAULT_PROJECT_ROOT_MARKERS
            )
        except Exception:
            markers = _DEFAULT_PROJECT_ROOT_MARKERS
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
        if not prompt_file.exists():
            # The package ships the prompts/ directory itself, so drop the
            # leading ``prompts/`` component: ``@prompts/x.md`` maps to
            # ``ask_llm/prompts/x.md``, mirroring ``<root>/prompts/x.md``.
            pkg_relative = relative_path.lstrip("/")
            if pkg_relative.startswith("prompts/"):
                pkg_relative = pkg_relative[len("prompts/") :]
            packaged = _PACKAGE_PROMPTS_DIR / pkg_relative
            if packaged.exists():
                return packaged.resolve()
    else:
        prompt_file = Path(prompt_path)

    if not prompt_file.is_absolute():
        prompt_file = prompt_file.resolve()

    return prompt_file


def expand_prompt(template: str, content: str) -> str:
    """Substitute ``{content}`` into *template* using replace semantics.

    Replace, not ``str.format``: prompts often contain literal ``{``/``}``
    (LaTeX, JSON examples, ``{variable}`` in code samples). Only ``{content}``
    is a placeholder; a template without it gets the content appended.
    """
    if "{content}" in template:
        return template.replace("{content}", content)
    return f"{template}\n\n{content}"


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


def resolve_prompt_or_template(prompt: str | None) -> str | None:
    """Resolve a ``--prompt`` argument that may be a file path or a literal.

    Shared by ask/chat, whose ``--prompt`` doubles as an inline template
    (M4). Resolution rules:

    - ``@path`` → project-root / packaged prompt file (missing ⇒ error)
    - ``~/path`` → explicit home path (missing ⇒ error — previously the whole
      quoted path string was silently sent to the LLM as literal text)
    - existing file path → read as file
    - anything else → returned verbatim as the literal template
    """
    if not prompt:
        return None

    if prompt.startswith("@"):
        return load_prompt_template(prompt)

    if prompt.startswith("~"):
        expanded = Path(prompt).expanduser()
        if not expanded.is_file():
            raise FileNotFoundError(f"Prompt file not found: {expanded}")
        return FileHandler.read(expanded).strip()

    if Path(prompt).is_file():
        return FileHandler.read(prompt).strip()

    return prompt
