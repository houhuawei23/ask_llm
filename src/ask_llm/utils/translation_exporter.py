"""Translation result exporter."""

import json
from pathlib import Path
from typing import List, Optional

from loguru import logger

from ask_llm.core.batch import BatchResult
from ask_llm.core.text_splitter import TextChunk


class TranslationExporter:
    """Export translation results to various formats."""

    def __init__(
        self,
        chunks: List[TextChunk],
        results: List[BatchResult],
        preserve_format: bool = True,
        include_original: bool = False,
    ):
        """
        Initialize translation exporter.

        Args:
            chunks: Original text chunks
            results: Translation results
            preserve_format: Whether to preserve original formatting
            include_original: Whether to include original text alongside translation
        """
        self.chunks = chunks
        self.results = results
        self.preserve_format = preserve_format
        self.include_original = include_original

        # Create mapping from chunk_id to result
        self.result_map = {result.task_id: result for result in results}

    def export(self, output_path: str, format_type: Optional[str] = None) -> str:
        """
        Export translation results to file.

        Args:
            output_path: Output file path
            format_type: Output format (auto-detected from extension if None)

        Returns:
            Path to exported file
        """
        if format_type is None:
            format_type = self._detect_format(output_path)

        if format_type == "json":
            return self._export_json(output_path)
        elif format_type == "markdown":
            return self._export_markdown(output_path)
        else:
            return self._export_text(output_path)

    def _detect_format(self, output_path: str) -> str:
        """
        Detect output format from file extension.

        Args:
            output_path: Output file path

        Returns:
            Format type ('json', 'markdown', or 'text')
        """
        path = Path(output_path)
        ext = path.suffix.lower()
        if ext == ".json":
            return "json"
        elif ext in (".md", ".markdown"):
            return "markdown"
        return "text"

    def _export_text(self, output_path: str) -> str:
        """
        Export as plain text.

        Args:
            output_path: Output file path

        Returns:
            Path to exported file
        """
        content_parts = []

        # Sort chunks by chunk_id to maintain order
        sorted_chunks = sorted(self.chunks, key=lambda c: c.chunk_id)

        for chunk in sorted_chunks:
            result = self.result_map.get(chunk.chunk_id)
            if result and result.response:
                translated_text = result.response.strip()
                if self.include_original:
                    content_parts.append(f"{chunk.content}\n---\n{translated_text}\n")
                else:
                    content_parts.append(translated_text)
            else:
                # If translation failed, include original text
                logger.warning(f"Translation failed for chunk {chunk.chunk_id}, using original")
                content_parts.append(chunk.content)

        # Join with appropriate separators
        separator = "\n\n" if self.preserve_format else "\n"
        content = separator.join(content_parts)

        # Write to file
        Path(output_path).write_text(content, encoding="utf-8")
        logger.info(f"Exported translation to: {output_path}")
        return output_path

    def _export_markdown(self, output_path: str) -> str:
        """
        Export as Markdown (preserving structure).

        Args:
            output_path: Output file path

        Returns:
            Path to exported file
        """
        content_parts = []

        # Sort chunks by chunk_id to maintain order
        sorted_chunks = sorted(self.chunks, key=lambda c: c.chunk_id)

        for chunk in sorted_chunks:
            result = self.result_map.get(chunk.chunk_id)
            if result and result.response:
                translated_text = result.response.strip()

                # If chunk has heading metadata, preserve it
                if "heading_level" in chunk.metadata:
                    level = chunk.metadata["heading_level"]
                    heading_prefix = "#" * level + " "
                    # Extract heading from translated text if present
                    # Otherwise use original heading title
                    heading_title = chunk.metadata.get("heading_title", "")
                    # Try to find heading in translated text
                    lines = translated_text.split("\n")
                    if lines and lines[0].startswith("#"):
                        # Heading already in translated text
                        content_parts.append(translated_text)
                    else:
                        # Add heading if we have title
                        if heading_title:
                            content_parts.append(f"{heading_prefix}{heading_title}\n")
                        content_parts.append(translated_text)
                else:
                    content_parts.append(translated_text)

                if self.include_original:
                    content_parts.append(f"\n\n<!-- Original:\n{chunk.content}\n-->")
            else:
                # If translation failed, include original text
                logger.warning(f"Translation failed for chunk {chunk.chunk_id}, using original")
                content_parts.append(chunk.content)

        content = "\n\n".join(content_parts)

        # Write to file
        Path(output_path).write_text(content, encoding="utf-8")
        logger.info(f"Exported translation to: {output_path}")
        return output_path

    def _export_json(self, output_path: str) -> str:
        """
        Export as JSON (structured format with metadata).

        Args:
            output_path: Output file path

        Returns:
            Path to exported file
        """
        export_data = {
            "chunks": [],
            "statistics": {
                "total_chunks": len(self.chunks),
                "successful_translations": sum(
                    1 for r in self.results if r.status.value == "success" and r.response
                ),
                "failed_translations": sum(1 for r in self.results if r.status.value == "failed"),
            },
        }

        # Sort chunks by chunk_id
        sorted_chunks = sorted(self.chunks, key=lambda c: c.chunk_id)

        for chunk in sorted_chunks:
            result = self.result_map.get(chunk.chunk_id)
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "original": chunk.content,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "metadata": chunk.metadata,
            }

            if result:
                chunk_data["translation"] = result.response if result.response else None
                chunk_data["status"] = result.status.value
                chunk_data["error"] = result.error
                if result.metadata:
                    chunk_data["translation_metadata"] = {
                        "provider": result.metadata.provider,
                        "model": result.metadata.model,
                        "input_tokens": result.metadata.input_tokens,
                        "output_tokens": result.metadata.output_tokens,
                        "latency": result.metadata.latency,
                    }
            else:
                chunk_data["translation"] = None
                chunk_data["status"] = "missing"
                chunk_data["error"] = "No result found for this chunk"

            export_data["chunks"].append(chunk_data)

        # Write JSON file
        Path(output_path).write_text(
            json.dumps(export_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Exported translation to: {output_path}")
        return output_path
