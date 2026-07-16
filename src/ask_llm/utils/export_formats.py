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


def detect_export_format(output_path: str, *, default: str = "json") -> str:
    """Detect output format from file extension.

    Args:
        output_path: Output file path.
        default: Format returned when the extension is unknown/absent
            (batch uses ``"json"``; translation uses ``"text"``).
    """
    suffix = Path(output_path).suffix.lower()
    return EXTENSION_TO_FORMAT.get(suffix, default)
