"""Shared CLI helpers (config init, translation path resolution).

The path-resolution helpers live in ``ask_llm.utils.path_resolver`` (P4.3)
and are re-exported here for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ask_llm.core.batch import BatchTask
from ask_llm.core.text_splitter import TextChunk
from ask_llm.utils.console import console
from ask_llm.utils.path_resolver import (  # noqa: F401  (re-export)
    _is_directory_output,
    _resolve_trans_input_paths,
)


def _offset_task_ids(
    tasks: list[BatchTask], chunks: list[TextChunk], offset: int
) -> tuple[list[BatchTask], list[TextChunk]]:
    """Shift task and chunk IDs by ``offset`` so tasks from multiple files are globally unique.

    Returns new task and chunk lists where each ``task_id``/``chunk_id`` is the original
    value plus ``offset``. This preserves the one-to-one mapping required by
    :class:`TranslationExporter` when results from all files are processed in a single
    global batch.
    """
    id_map: dict[int, int] = {chunk.chunk_id: chunk.chunk_id + offset for chunk in chunks}
    new_chunks = [chunk.model_copy(update={"chunk_id": id_map[chunk.chunk_id]}) for chunk in chunks]
    new_tasks = [task.model_copy(update={"task_id": id_map[task.task_id]}) for task in tasks]
    return new_tasks, new_chunks


def _config_init(output_path: str | None = None) -> None:
    """Generate default_config.yml and providers.yml templates."""
    pkg_dir = Path(__file__).resolve().parent.parent / "config"
    pkg_config = pkg_dir / "default_config.yml"
    pkg_providers = pkg_dir / "providers.yml"
    if not pkg_config.exists():
        console.print_error("Package default config not found")
        raise typer.Exit(1)

    if output_path:
        dest = Path(output_path)
    else:
        dest = Path.home() / ".config" / "ask_llm" / "default_config.yml"

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        console.print_warning(f"File exists: {dest}")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(0)

    try:
        content = pkg_config.read_text(encoding="utf-8")
        dest.write_text(content, encoding="utf-8")
        console.print_success(f"Configuration template written to: {dest}")
        console.print("Edit the file to set your API keys (use ${VAR} for environment variables).")
    except Exception as e:
        console.print_error(f"Failed to write config: {e}")
        raise typer.Exit(1) from e

    if pkg_providers.exists():
        providers_dest = dest.parent / "providers.yml"
        if providers_dest.exists():
            console.print_warning(f"File exists: {providers_dest}")
            if not typer.confirm("Overwrite providers.yml?"):
                return
        try:
            providers_dest.write_text(pkg_providers.read_text(encoding="utf-8"), encoding="utf-8")
            console.print_success(f"Provider catalog written to: {providers_dest}")
        except Exception as e:
            console.print_error(f"Failed to write providers.yml: {e}")
            raise typer.Exit(1) from e
