"""Batch result exporter for multiple output formats."""

import csv
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import yaml
from loguru import logger

from ask_llm.core.batch import BatchResult, BatchStatistics, TaskStatus
from ask_llm.utils.file_handler import FileHandler


class BatchResultExporter:
    """Export batch processing results to various formats."""

    SUPPORTED_FORMATS: ClassVar[List[str]] = ["json", "yaml", "csv", "markdown"]

    def __init__(
        self,
        results: List[BatchResult],
        statistics: BatchStatistics,
        batch_mode: Optional[str] = None,
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

    def export(self, output_path: str, format_type: Optional[str] = None) -> str:
        """
        Export results to file.

        Args:
            output_path: Output file path
            format_type: Output format (json, yaml, csv, markdown).
                If None, will be auto-detected from file extension.

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported
        """
        output_file = Path(output_path)

        # Auto-detect format from file extension if not specified
        if format_type is None:
            suffix = output_file.suffix.lower()
            extension_to_format = {
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".csv": "csv",
                ".md": "markdown",
                ".markdown": "markdown",
            }
            format_type = extension_to_format.get(suffix, "json")
            logger.debug(f"Auto-detected format '{format_type}' from extension '{suffix}'")
        else:
            format_type = format_type.lower()

        if format_type not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format_type}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Generate output path if needed (add extension if missing)
        if not output_file.suffix:
            # Map format to extension
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
            content = self._export_json()
        elif format_type == "yaml":
            content = self._export_yaml()
        elif format_type == "csv":
            content = self._export_csv()
        elif format_type == "markdown":
            content = self._export_markdown()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Write to file
        FileHandler.write(str(output_file), content, force=True)

        logger.info(f"Exported {len(self.results)} results to {output_file}")
        return str(output_file)

    def _export_json(self) -> str:
        """Export results as JSON."""
        data = self._prepare_data()
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def _export_yaml(self) -> str:
        """Export results as YAML."""
        data = self._prepare_data()
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _export_csv(self) -> str:
        """Export results as CSV."""
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

        # Write rows
        for result in self.results:
            writer.writerow(
                [
                    result.task_id,
                    result.status.value,
                    result.model_settings.provider,
                    result.model_settings.model,
                    result.prompt[:100] + "..." if len(result.prompt) > 100 else result.prompt,
                    result.content[:100] + "..." if len(result.content) > 100 else result.content,
                    result.response[:100] + "..."
                    if result.response and len(result.response) > 100
                    else (result.response or ""),
                    result.error or "",
                    f"{result.metadata.latency:.2f}" if result.metadata else "",
                    result.metadata.input_tokens if result.metadata else "",
                    result.metadata.output_tokens if result.metadata else "",
                    result.timestamp.isoformat() if result.timestamp else "",
                ]
            )

        return output.getvalue()

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
        if self.statistics.successful_tasks > 0:
            lines.append(
                f"- **Success Rate:** {self.statistics.successful_tasks / self.statistics.total_tasks * 100:.1f}%"
            )
            lines.append(f"- **Average Latency:** {self.statistics.average_latency:.2f}s")
            lines.append(f"- **Total Input Tokens:** {self.statistics.total_input_tokens:,}")
            lines.append(f"- **Total Output Tokens:** {self.statistics.total_output_tokens:,}")
        lines.append("")

        # Group results by model
        results_by_model: Dict[str, List[BatchResult]] = {}
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
        self, lines: List[str], successful: List[BatchResult], failed: List[BatchResult]
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
                lines.append(result.response or "")
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
        self, lines: List[str], successful: List[BatchResult], failed: List[BatchResult]
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
                lines.append(result.response or "")
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

    def _prepare_data(self) -> Dict[str, Any]:
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
            "results": [
                {
                    "task_id": result.task_id,
                    "prompt": result.prompt,
                    "content": result.content,
                    "model_config": {
                        "provider": result.model_settings.provider,
                        "model": result.model_settings.model,
                        "temperature": result.model_settings.temperature,
                        "top_p": result.model_settings.top_p,
                    },
                    "response": result.response,
                    "status": result.status.value,
                    "error": result.error,
                    "metadata": {
                        "provider": result.metadata.provider,
                        "model": result.metadata.model,
                        "temperature": result.metadata.temperature,
                        "input_tokens": result.metadata.input_tokens,
                        "output_tokens": result.metadata.output_tokens,
                        "latency": result.metadata.latency,
                        "timestamp": result.metadata.timestamp.isoformat(),
                    }
                    if result.metadata
                    else None,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None,
                    "retry_count": result.retry_count,
                }
                for result in self.results
            ],
        }

    @classmethod
    def export_multiple_models(
        cls,
        results_by_model: Dict[str, List[BatchResult]],
        statistics_by_model: Dict[str, BatchStatistics],
        output_dir: str,
        format_type: str = "json",
        batch_mode: Optional[str] = None,
    ) -> List[str]:
        """
        Export results grouped by model to separate files.

        Args:
            results_by_model: Dictionary mapping model keys to results
            statistics_by_model: Dictionary mapping model keys to statistics
            output_dir: Output directory
            format_type: Output format
            batch_mode: Batch mode ('prompt-contents' or 'prompt-content-pairs')

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
            file_path = exporter.export(str(output_path / filename), format_type)
            exported_files.append(file_path)

        return exported_files
