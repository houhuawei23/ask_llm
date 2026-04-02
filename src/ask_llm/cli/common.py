"""Shared CLI helpers (config init, translation path resolution)."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import typer

from ask_llm.config.context import get_config
from ask_llm.core.batch import (
    ModelConfig,
)
from ask_llm.core.translator import Translator
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.notebook_translator import NotebookTranslator
from ask_llm.utils.pricing import format_cost_estimate


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


def _config_init(output_path: str | None = None) -> None:
    """Generate default_config.yml template."""
    pkg_config = Path(__file__).resolve().parent.parent / "config" / "default_config.yml"
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


def _process_notebook_translation(
    file_path: str,
    output: str | None,
    trans_config: Any,
    config_manager: Any,
    final_provider: str,
    final_model: str,
    force: bool,
    stream: bool,
    pricing_map: Any,
    pricing_source: Any,
) -> tuple[int, int] | None:
    """Process .ipynb notebook translation (markdown cells only). Returns (input_tokens, output_tokens) if any."""
    # Determine output path
    if output:
        output_path = output
        if Path(output).is_dir():
            input_file = Path(file_path)
            output_name = f"{input_file.stem}{get_config().unified_config.file.translated_suffix}{input_file.suffix}"
            output_path = str(Path(output) / output_name)
    else:
        output_path = FileHandler.generate_output_path(
            file_path, suffix=get_config().unified_config.file.translated_suffix
        )

    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )

    translator = Translator(
        target_language=trans_config.target_language,
        source_language=trans_config.source_language,
        style=trans_config.style,
        custom_prompt_template=trans_config.prompt_template,
        prompt_file=trans_config.prompt_file,
    )

    model_config = ModelConfig(
        provider=final_provider,
        model=final_model,
        temperature=trans_config.temperature,
        max_tokens=trans_config.max_output_tokens,
    )

    notebook_translator = NotebookTranslator(
        translator=translator,
        model_config=model_config,
    )

    try:
        successful, failed, total_in, total_out = notebook_translator.translate_notebook(
            input_path=file_path,
            output_path=output_path,
            config_manager=config_manager,
            max_workers=trans_config.threads,
            max_retries=trans_config.retries,
            show_progress=not stream,
            balance_chunks=trans_config.balance_translation_chunks,
            max_chunk_tokens=trans_config.max_chunk_tokens,
            min_chunk_merge_tokens=trans_config.min_chunk_merge_tokens,
        )
    except RuntimeError as e:
        if "API authentication failed" in str(e):
            console.print_error(str(e))
            raise typer.Exit(1) from e
        raise

    total = successful + failed
    console.print_success(f"Translation saved to: {output_path}")
    console.print(f"  Successful: {successful}/{total}")
    if failed > 0:
        console.print_warning(f"  Failed: {failed}/{total}")
    console.print(
        format_cost_estimate(
            final_provider,
            final_model,
            total_in,
            total_out,
            pricing_map,
            pricing_source=pricing_source,
        )
    )
    return (total_in, total_out)
