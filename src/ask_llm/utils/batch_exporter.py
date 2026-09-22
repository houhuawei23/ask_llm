"""Batch result exporter for multiple output formats."""

import csv
import json
import re
from pathlib import Path
from typing import Any, ClassVar, cast

import yaml
from loguru import logger

from ask_llm.core.batch_models import BatchResult, BatchStatistics, TaskStatus
from ask_llm.utils.file_handler import FileHandler


class BatchResultExporter:
    """Export batch processing results to various formats.

    Audit 4.5: exports respect ``force`` — an existing target raises
    ``FileExistsError`` instead of being silently overwritten; auto-detected
    formats refuse unrecognized extensions rather than defaulting to JSON
    (JSON content never lands in a ``.txt``); CSV cells are neutralized
    against spreadsheet formula injection; LLM responses are fenced in
    Markdown exports.
    """

    SUPPORTED_FORMATS: ClassVar[list[str]] = ["json", "yaml", "csv", "markdown"]

    # Auto-detection extension map; unknown extensions are rejected instead of
    # silently exporting JSON (audit 4.5).
    _FORMAT_EXTENSIONS: ClassVar[dict[str, tuple[str, ...]]] = {
        "json": (".json",),
        "yaml": (".yaml", ".yml"),
        "csv": (".csv",),
        "markdown": (".md", ".markdown"),
    }

    def __init__(
        self,
        results: list[BatchResult],
        statistics: BatchStatistics,
        batch_mode: str | None = None,
    ):
        """
        Initialize exporter.

        Args:
            results: List of batch results
            statistics: Batch statistics
            batch_mode: Batch mode ('prompt-contents' or 'prompt-content-pairs')
        """
        self.results = results
        self.statistics = statistics
        self.batch_mode = batch_mode or self._detect_batch_mode()

    def _detect_batch_mode(self) -> str:
        """
        Detect batch mode from results.

        Returns:
            Detected batch mode
        """
        if not self.results:
            return "prompt-content-pairs"

        # Check if all prompts are the same (prompt-contents mode)
        first_prompt = self.results[0].prompt
        all_same_prompt = all(r.prompt == first_prompt for r in self.results)

        return "prompt-contents" if all_same_prompt else "prompt-content-pairs"

    def export(
        self, output_path: str, format_type: str | None = None, *, force: bool = False
    ) -> str:
        """
        Export results to file.

        Args:
            output_path: Output file path
            format_type: Output format (json, yaml, csv, markdown).
                If None, will be auto-detected from file extension.
            force: Overwrite an existing output file. When False (audit 4.5),
                an existing target raises ``FileExistsError``.

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported, or the extension is not
                recognized while auto-detecting.
            FileExistsError: If the target exists and ``force`` is False.
        """
        output_file = Path(output_path)

        # Auto-detect format from file extension if not specified
        if format_type is None:
            format_type = self._detect_format_from_suffix(output_file)
            logger.debug(f"Auto-detected format '{format_type}' from extension")
        else:
            format_type = format_type.lower()

        if format_type not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format_type}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Generate output path if needed (add extension if missing)
        if not output_file.suffix:
            format_to_extension = {
                "json": ".json",
                "yaml": ".yaml",
                "csv": ".csv",
                "markdown": ".md",
            }
            extension = format_to_extension.get(format_type, ".json")
            output_file = output_file.with_suffix(extension)

        # Export based on format
        if format_type == "json":
            self._export_json(str(output_file), force=force)
        elif format_type == "yaml":
            content = self._export_yaml()
            FileHandler.write(str(output_file), content, force=force)
        elif format_type == "csv":
            content = self._export_csv()
            FileHandler.write(str(output_file), content, force=force)
        elif format_type == "markdown":
            content = self._export_markdown()
            FileHandler.write(str(output_file), content, force=force)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.info(f"Exported {len(self.results)} results to {output_file}")
        return str(output_file)

    @classmethod
    def _detect_format_from_suffix(cls, output_file: Path) -> str:
        """Map a file extension to an export format; unknown ones are an error.

        The previous behavior defaulted unknown extensions to JSON, quietly
        writing JSON payload into e.g. a ``.txt`` file (audit 4.5).
        """
        suffix = output_file.suffix.lower()
        for fmt, extensions in cls._FORMAT_EXTENSIONS.items():
            if suffix in extensions:
                return fmt
        shown = suffix if suffix else "(none)"
        raise ValueError(
            f"Cannot infer export format from extension '{shown}' of "
            f"'{output_file.name}'. Supported extensions: "
            + ", ".join(ext for exts in cls._FORMAT_EXTENSIONS.values() for ext in exts)
            + " — or pass --format explicitly."
        )

    def _export_json(self, output_path: str, *, force: bool = False) -> None:
        """Export results as JSON using a streaming encoder.

        For large result sets this avoids materializing the entire JSON string in
        memory before writing it to disk. Streams into a tmp file and swaps at
        the end, matching FileHandler's atomic-write + parent-mkdir semantics
        used by the yaml/csv/markdown exports (H10).
        """
        output_file = Path(output_path)
        if output_file.exists() and not force:
            raise FileExistsError(
                f"Output file already exists: {output_path}. Use --force to overwrite."
            )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
        encoder = json.JSONEncoder(indent=2, ensure_ascii=False, default=str)
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                for chunk in encoder.iterencode(self._prepare_data()):
                    f.write(chunk)
            tmp_file.replace(output_file)
        except Exception:
            tmp_file.unlink(missing_ok=True)
            raise

    def _export_yaml(self) -> str:
        """Export results as YAML."""
        data = self._prepare_data()
        return cast(
            str, yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )

    # Audit 4.5: spreadsheet formula-injection guard. A cell starting with one
    # of these characters is executed by Excel/Sheets when the CSV is opened;
    # prefixing a single quote forces text interpretation.
    _CSV_INJECTION_PREFIXES: ClassVar[tuple[str, ...]] = ("=", "+", "-", "@", "\t", "\r")

    @classmethod
    def _csv_safe(cls, value: str) -> str:
        """Neutralize spreadsheet formula injection in a CSV cell."""
        if value.startswith(cls._CSV_INJECTION_PREFIXES):
            return "'" + value
        return value

    def _export_csv(self) -> str:
        """Export results as CSV (full content, injection-neutralized)."""
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "Task ID",
                "Status",
                "Provider",
                "Model",
                "Prompt",
                "Content",
                "Response",
                "Error",
                "Latency (s)",
                "Input Tokens",
                "Output Tokens",
                "Timestamp",
            ]
        )

        # Write rows — full prompt/content/response, no truncation (audit 4.5):
        # the 100-character cut silently dropped most of the paid-for output.
        for result in self.results:
            writer.writerow(
                [
                    result.task_id,
                    self._csv_safe(result.status.value),
                    self._csv_safe(result.model_settings.provider),
                    self._csv_safe(result.model_settings.model),
                    self._csv_safe(result.prompt),
                    self._csv_safe(result.content),
                    self._csv_safe(result.response or ""),
                    self._csv_safe(result.error or ""),
                    f"{result.metadata.latency:.2f}" if result.metadata else "",
                    result.metadata.input_tokens if result.metadata else "",
                    result.metadata.output_tokens if result.metadata else "",
                    result.timestamp.isoformat() if result.timestamp else "",
                ]
            )

        return output.getvalue()

    @staticmethod
    def _markdown_fence(payload: str) -> list[str]:
        """Return *payload* wrapped in a code fence that survives embedded backticks.

        The fence grows one backtick beyond the longest run inside the payload,
        so LLM output containing its own code blocks cannot break out (audit 4.5).
        """
        runs = re.findall(r"`+", payload)
        longest = max((len(r) for r in runs), default=0)
        fence = "`" * max(3, longest + 1)
        return [fence, payload, fence]

    def _export_markdown(self) -> str:
        """Export results as Markdown with format-aware structure."""
        lines = []

        # Header
        lines.append("# Batch Processing Results")
        lines.append("")
        if self.results:
            timestamp = (
                self.results[0].timestamp.isoformat() if self.results[0].timestamp else "N/A"
            )
            lines.append(f"**Generated:** {timestamp}")
        lines.append("")

        # Statistics
        lines.append("## Statistics")
        lines.append("")
        lines.append(f"- **Total Tasks:** {self.statistics.total_tasks}")
        lines.append(f"- **Successful:** {self.statistics.successful_tasks}")
        lines.append(f"- **Failed:** {self.statistics.failed_tasks}")
        if self.statistics.total_tasks > 0:
            lines.append(
                f"- **Success Rate:** {self.statistics.successful_tasks / self.statistics.total_tasks * 100:.1f}%"
            )
            lines.append(f"- **Average Latency:** {self.statistics.average_latency:.2f}s")
            lines.append(f"- **Total Input Tokens:** {self.statistics.total_input_tokens:,}")
            lines.append(f"- **Total Output Tokens:** {self.statistics.total_output_tokens:,}")
        lines.append("")

        # Group results by model
        results_by_model: dict[str, list[BatchResult]] = {}
        for result in self.results:
            model_key = f"{result.model_settings.provider}/{result.model_settings.model}"
            if model_key not in results_by_model:
                results_by_model[model_key] = []
            results_by_model[model_key].append(result)

        # Results by model
        for model_key, model_results in results_by_model.items():
            lines.append(f"## Model: {model_key}")
            lines.append("")

            successful = [r for r in model_results if r.status == TaskStatus.SUCCESS]
            failed = [r for r in model_results if r.status == TaskStatus.FAILED]

            lines.append(f"- **Total:** {len(model_results)}")
            lines.append(f"- **Successful:** {len(successful)}")
            lines.append(f"- **Failed:** {len(failed)}")
            lines.append("")

            # Format results based on batch mode
            if self.batch_mode == "prompt-contents":
                self._export_markdown_prompt_contents(lines, successful, failed)
            else:
                self._export_markdown_prompt_content_pairs(lines, successful, failed)

        return "\n".join(lines)

    def _export_markdown_prompt_contents(
        self, lines: list[str], successful: list[BatchResult], failed: list[BatchResult]
    ) -> None:
        """Export prompt-contents format: one prompt + multiple content+answer pairs."""
        if successful:
            # Get the common prompt (should be the same for all)
            common_prompt = successful[0].prompt if successful else ""

            lines.append("### Prompt")
            lines.append("")
            lines.append(common_prompt)
            lines.append("")
            lines.append("---")
            lines.append("")

            # Content + Answer pairs
            lines.append("### Results")
            lines.append("")
            for result in sorted(successful, key=lambda r: r.task_id):
                lines.append(f"#### {result.task_id}. Content")
                lines.append("")
                lines.append(result.content)
                lines.append("")
                lines.append("**Answer:**")
                lines.append("")
                lines.extend(self._markdown_fence(result.response or ""))
                lines.append("")
                if result.metadata:
                    lines.append(
                        f"*Latency: {result.metadata.latency:.2f}s | "
                        f"Tokens: {result.metadata.input_tokens} → {result.metadata.output_tokens}*"
                    )
                    lines.append("")
                lines.append("---")
                lines.append("")

        # Failed results
        if failed:
            lines.append("### Failed Tasks")
            lines.append("")
            for result in sorted(failed, key=lambda r: r.task_id):
                lines.append(f"#### {result.task_id}. Content")
                lines.append("")
                lines.append(result.content)
                lines.append("")
                lines.append(f"**Error:** {result.error}")
                lines.append("")
                lines.append("---")
                lines.append("")

    def _export_markdown_prompt_content_pairs(
        self, lines: list[str], successful: list[BatchResult], failed: list[BatchResult]
    ) -> None:
        """Export prompt-content-pairs format: multiple prompt+content+answer pairs."""
        if successful:
            lines.append("### Results")
            lines.append("")
            for result in sorted(successful, key=lambda r: r.task_id):
                lines.append(f"#### {result.task_id}. Prompt")
                lines.append("")
                lines.append(result.prompt)
                lines.append("")
                lines.append("**Content:**")
                lines.append("")
                lines.append(result.content)
                lines.append("")
                lines.append("**Answer:**")
                lines.append("")
                lines.extend(self._markdown_fence(result.response or ""))
                lines.append("")
                if result.metadata:
                    lines.append(
                        f"*Latency: {result.metadata.latency:.2f}s | "
                        f"Tokens: {result.metadata.input_tokens} → {result.metadata.output_tokens}*"
                    )
                    lines.append("")
                lines.append("---")
                lines.append("")

        # Failed results
        if failed:
            lines.append("### Failed Tasks")
            lines.append("")
            for result in sorted(failed, key=lambda r: r.task_id):
                lines.append(f"#### {result.task_id}. Prompt")
                lines.append("")
                lines.append(result.prompt)
                lines.append("")
                lines.append("**Content:**")
                lines.append("")
                lines.append(result.content)
                lines.append("")
                lines.append(f"**Error:** {result.error}")
                lines.append("")
                lines.append("---")
                lines.append("")

    def _prepare_data(self) -> dict[str, Any]:
        """
        Prepare data structure for export.

        Returns:
            Dictionary with results and statistics
        """
        return {
            "statistics": {
                "total_tasks": self.statistics.total_tasks,
                "successful_tasks": self.statistics.successful_tasks,
                "failed_tasks": self.statistics.failed_tasks,
                "average_latency": self.statistics.average_latency,
                "total_input_tokens": self.statistics.total_input_tokens,
                "total_output_tokens": self.statistics.total_output_tokens,
            },
            "results": [result.project() for result in self.results],
        }

    @classmethod
    def export_multiple_models(
        cls,
        results_by_model: dict[str, list[BatchResult]],
        statistics_by_model: dict[str, BatchStatistics],
        output_dir: str,
        format_type: str = "json",
        batch_mode: str | None = None,
        *,
        force: bool = False,
    ) -> list[str]:
        """
        Export results grouped by model to separate files.

        Args:
            results_by_model: Dictionary mapping model keys to results
            statistics_by_model: Dictionary mapping model keys to statistics
            output_dir: Output directory
            format_type: Output format
            batch_mode: Batch mode ('prompt-contents' or 'prompt-content-pairs')
            force: Overwrite existing files (audit 4.5)

        Returns:
            List of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = []

        for model_key, results in results_by_model.items():
            # Sanitize model key for filename
            safe_model_key = model_key.replace("/", "_").replace(" ", "_")
            filename = f"batch_results_{safe_model_key}.{format_type}"

            statistics = statistics_by_model.get(
                model_key, BatchStatistics(total_tasks=len(results))
            )

            exporter = cls(results, statistics, batch_mode)
            file_path = exporter.export(str(output_path / filename), format_type, force=force)
            exported_files.append(file_path)

        return exported_files

    @classmethod
    def export_split_files(
        cls,
        results: list[BatchResult],
        output_dir: str,
        batch_mode: str | None = None,  # noqa: ARG003
        *,
        force: bool = False,
    ) -> list[str]:
        """
        Export each task result to a separate file.
        File content contains only the LLM response.

        Args:
            results: List of batch results
            output_dir: Output directory path
            batch_mode: Batch mode (not used in split mode, kept for API consistency)
            force: Overwrite existing files (audit 4.5). Filename conflicts
                within this export still resolve via ``_N`` suffixes.

        Returns:
            List of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = []
        filename_counter: dict[str, int] = {}  # Track filename usage for conflict resolution

        for idx, result in enumerate(results):
            # Determine filename
            if result.output_filename:
                # Use configured output filename
                filename = result.output_filename
                # Sanitize filename: replace dangerous characters with underscore
                filename = re.sub(r"[^\w\-.]", "_", filename)
                # Remove path traversal attempts (..)
                filename = filename.replace("..", "_")
                # Remove any leading/trailing dots and spaces
                filename = filename.strip(". ")
                if not filename:
                    # Fallback if filename becomes empty after sanitization
                    # Use index + 1 for user-friendly numbering (starts from 1)
                    filename = f"task_{idx + 1}.md"
            else:
                # Generate default filename using index + 1 for user-friendly numbering (starts from 1)
                filename = f"task_{idx + 1}.md"

            # Ensure filename has .md extension if not specified
            if not filename.endswith((".md", ".txt", ".markdown")):
                filename = f"{filename}.md"

            # Handle filename conflicts
            if filename in filename_counter:
                filename_counter[filename] += 1
                base_name = Path(filename).stem
                extension = Path(filename).suffix
                filename = f"{base_name}_{filename_counter[filename]}{extension}"
            else:
                filename_counter[filename] = 0

            # Create full file path
            file_path = output_path / filename

            # Write only the response content (if available)
            if result.response:
                FileHandler.write(str(file_path), result.response, force=force)
                exported_files.append(str(file_path))
                logger.debug(f"Exported task {result.task_id} to {file_path}")
            else:
                # If no response (failed task), create empty file or skip
                # Option: create empty file to indicate task was processed
                FileHandler.write(str(file_path), "", force=force)
                exported_files.append(str(file_path))
                logger.warning(
                    f"Task {result.task_id} has no response, created empty file: {file_path}"
                )

        logger.info(f"Exported {len(exported_files)} files to {output_dir}")
        return exported_files
