"""Shared export-format detection (P4.7).

Single extension→format mapping for all exporters. Previously each exporter
carried its own (batch: json/yaml/csv/markdown; translation: json/markdown/
text), which drifted.
"""

from __future__ import annotations

from pathlib import Path

EXTENSION_TO_FORMAT: dict[str, str] = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".csv": "csv",
    ".md": "markdown",
    ".markdown": "markdown",
}


def detect_export_format(output_path: str, *, default: str = "json", strict: bool = False) -> str:
    """Detect output format from file extension.

    Args:
        output_path: Output file path.
        default: Format returned when the extension is unknown/absent
            (batch uses ``"json"``; translation uses ``"text"``).

    Raises:
        ValueError: When ``strict`` is set and the extension (if any) is not a
            known export extension (batch auto-detection behavior, audit 4.5).
    """
    suffix = Path(output_path).suffix.lower()
    fmt = EXTENSION_TO_FORMAT.get(suffix, default)
    if strict and suffix and EXTENSION_TO_FORMAT.get(suffix) is None:
        raise ValueError(
            f"Cannot infer output format from extension '{suffix}' of "
            f"'{output_path}'. Pass --format explicitly."
        )
    return fmt
