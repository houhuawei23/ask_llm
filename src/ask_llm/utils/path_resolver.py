"""Input/output path resolution for translation-style commands (P4.3).

Moved out of ``cli/common.py`` so the service layer can resolve paths without
importing from the CLI layer. ``cli/common.py`` re-exports these for backward
compatibility.
"""

from __future__ import annotations

import glob
from pathlib import Path

from ask_llm.utils.console import console


def _resolve_trans_input_paths(
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


def _is_directory_output(output: str, files: list[str], resolved_count: int) -> bool:
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
