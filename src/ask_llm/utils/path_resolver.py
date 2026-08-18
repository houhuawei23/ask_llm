"""Input/output path resolution for translation-style commands (P4.3).

Home of translation input resolution (files/globs/directories) and the shared
output-path mapping used by the text and notebook translators.
"""

from __future__ import annotations

import glob
from pathlib import Path

from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler


def resolve_trans_input_paths(
    files: list[str],
    translatable_extensions: list[str],
    recursive_dir: bool,
) -> list[str]:
    """
    Resolve input paths to a list of translatable files.

    Supports: directory (expands to matching files), file path, glob pattern.
    """
    resolved: list[str] = []
    for pattern in files:
        p = Path(pattern)
        if p.is_dir():
            for ext in translatable_extensions:
                ext_clean = ext if ext.startswith(".") else f".{ext}"
                if recursive_dir:
                    resolved.extend(str(f) for f in p.rglob(f"*{ext_clean}"))
                else:
                    resolved.extend(str(f) for f in p.glob(f"*{ext_clean}"))
        elif p.exists() and p.is_file():
            resolved.append(str(p.resolve()))
        else:
            matched = glob.glob(pattern)
            if matched:
                for m in matched:
                    mp = Path(m)
                    if mp.is_file():
                        resolved.append(str(mp.resolve()))
            elif p.exists():
                resolved.append(str(p.resolve()))
            else:
                console.print_warning(f"File not found: {pattern}")
    return sorted(set(resolved))


def resolve_translation_output_path(
    file_path: str,
    output: str | None,
    output_is_dir: bool,
    *,
    suffix: str,
) -> str:
    """Resolve a translation output path (shared by text and notebook translators).

    ``output`` that is (or is declared) a directory maps the input name into it
    with ``suffix`` inserted before the extension; otherwise it is used as-is;
    with no ``output`` a default is generated next to the input.
    """
    if output:
        if output_is_dir or Path(output).is_dir():
            input_file = Path(file_path)
            output_name = f"{input_file.stem}{suffix}{input_file.suffix}"
            return str(Path(output) / output_name)
        return output
    return FileHandler.generate_output_path(file_path, suffix=suffix)


def is_directory_output(output: str, files: list[str], resolved_count: int) -> bool:
    """Heuristically decide whether ``output`` is meant as a directory.

    A path is considered a directory when:
    - It already exists as a directory.
    - It ends with a path separator (``/`` or ``\\``).
    - It does not exist, has no file extension, and the input consists of
      multiple files or a directory.
    """
    output_path = Path(output)
    if output_path.is_dir():
        return True
    if output.endswith(("/", "\\")):
        return True
    if not output_path.exists() and not output_path.suffix:
        if resolved_count > 1:
            return True
        for pattern in files:
            if Path(pattern).is_dir():
                return True
    return False
