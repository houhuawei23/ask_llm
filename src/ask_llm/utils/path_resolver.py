"""Input/output path resolution for translation-style commands (P4.3).

Home of translation input resolution (files/globs/directories) and the shared
output-path mapping used by the text and notebook translators.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler


class OutputTargetError(ValueError):
    """Raised when the output configuration would collide or clobber (audit 2.1).

    Service-level signal so both the CLI and library callers can fail fast
    before any paid API work.
    """


def is_single_file_output(output: str) -> bool:
    """True if ``-o`` clearly targets one file rather than a directory."""
    p = Path(output)
    if p.exists():
        return p.is_file()
    # Non-existent path: treat as file if it looks like a single markdown file.
    return p.suffix.lower() in (".md", ".markdown") and not output.endswith(os.sep)


def validate_multi_input_output(output: str | None, file_count: int, *, inplace: bool) -> None:
    """Reject a single-file ``-o`` for a multi-file run (targets would collide).

    Raises:
        OutputTargetError: If *output* names a single file and more than one
            input file is being processed.
    """
    if inplace or file_count <= 1 or not output:
        return
    if is_single_file_output(output):
        raise OutputTargetError(
            f"Multiple input files cannot use a single file as -o/--output: {output}. "
            "Specify a directory, or omit -o for default per-file naming."
        )


# L10/2.25: explicitly-passed files whose extension is not in the
# translatable list used to be appended unconditionally, so the "supports
# .txt/.md/.ipynb" contract was unenforced and a stray binary failed later
# with a confusing UnicodeDecodeError. Known binaries are blocked; unknown
# text-like extensions are warned about but kept (permissive contract).
_KNOWN_BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".ico",
    ".svgz",
    ".pdf",
    ".zip",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".rar",
    ".7z",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wav",
    ".flac",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".iso",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".docx",
    ".xlsx",
    ".pptx",
    ".sqlite",
    ".db",
    ".parquet",
    ".pkl",
}


def resolve_trans_input_paths(
    files: list[str],
    translatable_extensions: list[str],
    recursive_dir: bool,
) -> list[str]:
    """
    Resolve input paths to a list of translatable files.

    Supports: directory (expands to matching files), file path, glob pattern.
    ``~`` is expanded, so quoted ``'~/docs/*.md'`` patterns work; every entry
    is returned as an absolute resolved path so downstream output mapping is
    uniform.
    """
    resolved: list[str] = []
    for pattern in files:
        p = Path(pattern).expanduser()
        if p.is_dir():
            for ext in translatable_extensions:
                ext_clean = ext if ext.startswith(".") else f".{ext}"
                if recursive_dir:
                    resolved.extend(str(f.resolve()) for f in p.rglob(f"*{ext_clean}"))
                else:
                    resolved.extend(str(f.resolve()) for f in p.glob(f"*{ext_clean}"))
        elif p.exists() and p.is_file():
            suffix = p.suffix.lower()
            ext_clean_set = {
                ext if ext.startswith(".") else f".{ext}" for ext in translatable_extensions
            }
            if suffix in ext_clean_set:
                resolved.append(str(p.resolve()))
            elif suffix in _KNOWN_BINARY_EXTENSIONS:
                console.print_warning(
                    f"Skipped binary/unsupported file: {pattern} ({suffix or 'no extension'})"
                )
            else:
                console.print_warning(
                    f"{pattern}: extension '{suffix or '(none)'}' is not in the translatable "
                    f"list ({', '.join(translatable_extensions)}); including it as plain text."
                )
                resolved.append(str(p.resolve()))
        else:
            matched = glob.glob(str(p))
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


def validate_output_targets(
    output_paths: list[str],
    *,
    force: bool,
    resume: bool = False,
) -> None:
    """Refuse colliding or pre-existing output targets BEFORE paid API work.

    Args:
        output_paths: Resolved target path for every job in the run.
        force: ``--force`` was given; existing targets may be overwritten.
        resume: Resume run; it legitimately targets its own partial output.

    Raises:
        OutputTargetError: On duplicate targets (several inputs resolving to
            one output — later writes then fail only after the full spend, or
            with ``--force`` degrade to last-writer-wins), or on an existing
            target without ``--force`` (the old per-job check fired at export
            time, after the run had already been paid for).
    """
    seen: dict[str, str] = {}
    for out in output_paths:
        key = str(Path(out).expanduser().resolve())
        if key in seen:
            raise OutputTargetError(
                f"Multiple input files resolve to the same output path: {out} "
                f"(already targeted for {seen[key]}). Use a directory as -o, or "
                "omit -o, when processing multiple files."
            )
        seen[key] = out
    if force or resume:
        return
    for out in output_paths:
        p = Path(out).expanduser()
        if p.exists():
            raise OutputTargetError(f"Output file already exists: {out}. Use --force to overwrite.")


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
