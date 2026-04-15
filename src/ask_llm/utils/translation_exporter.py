"""Translation result exporter."""

import json
from pathlib import Path
from typing import List, Optional

from loguru import logger

from ask_llm.core.batch import BatchResult
from ask_llm.core.text_splitter import TextChunk


def _try_parse_json_with_latex_escapes(raw: str, brace: int) -> dict | None:
    """Attempt to parse a JSON-like blob that contains invalid LaTeX escapes.

    LLMs sometimes emit ``{"translation": "text with $\\mathcal{V}$"}`` where
    ``\\mathcal`` is not a valid JSON escape.  They may also include literal
    newlines inside JSON strings.  This function:

    1. Extracts the JSON-like substring from *brace* to the last ``}``.
    2. Within string values, fixes two classes of problems:
       a. Doubles backslashes before non-JSON-escape characters
          (e.g. ``\\mathcal`` → ``\\\\mathcal``).
       b. Replaces literal control characters (newlines, tabs, etc.)
          with their JSON escape equivalents (``\\n``, ``\\t``, etc.).
    3. Parses the fixed string.
    """
    last = raw.rfind("}")
    if last <= brace:
        return None
    blob = raw[brace : last + 1]

    # Valid single-char JSON escapes after backslash
    _valid_escape_chars = set('"\\bfnrt/')
    _control_replace = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }

    fixed_parts: list[str] = []
    i = 0
    in_string = False
    while i < len(blob):
        ch = blob[i]

        # Track string boundaries (respect already-escaped quotes)
        if ch == '"' and (i == 0 or blob[i - 1] != "\\"):
            in_string = not in_string
            fixed_parts.append(ch)
            i += 1
        elif in_string and ch == "\\":
            # Check if this is already a valid JSON escape
            if i + 1 < len(blob) and blob[i + 1] in _valid_escape_chars:
                # Already valid — keep as-is
                fixed_parts.append(blob[i : i + 2])
                i += 2
            elif i + 1 < len(blob) and blob[i + 1] == "u":
                # Possible \uXXXX — keep as-is if valid hex
                if i + 5 < len(blob) and all(
                    c in "0123456789abcdefABCDEF" for c in blob[i + 2 : i + 6]
                ):
                    fixed_parts.append(blob[i : i + 6])
                    i += 6
                else:
                    # Invalid \u — double the backslash
                    fixed_parts.append("\\\\")
                    i += 1
            else:
                # Invalid escape (like \m in \mathcal) — double the backslash
                # so JSON sees \\m which decodes to \m
                fixed_parts.append("\\\\")
                i += 1
        elif in_string and ch in _control_replace:
            # Literal control character inside string — replace with JSON escape
            fixed_parts.append(_control_replace[ch])
            i += 1
        else:
            fixed_parts.append(ch)
            i += 1

    fixed = "".join(fixed_parts)
    try:
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    return None


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

    @staticmethod
    def _unwrap_translation_payload(text: str) -> str:
        """
        Unwrap common JSON-wrapped translation payloads returned by some prompts/models.

        Supports payloads like:
        - {"translation": "..."}
        - ```json {"translation":"..."} ```
        - {"translation": "..."} plus trailing text (models often append notes after ``}``)
        - JSON with invalid escape sequences (e.g. ``\\mathcal`` inside string values)

        Falls back to original text when parsing fails.

        Note: Do not require ``raw.endswith("}")`` — that rejects valid JSON objects when
        the model adds characters after the closing brace, which previously left the file
        as raw JSON.
        """
        original = text
        raw = text.strip()
        if not raw:
            return raw

        if raw.startswith("\ufeff"):
            raw = raw.lstrip("\ufeff").strip()

        # Strip optional fenced code block wrapper.
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
                raw = "\n".join(lines[1:-1]).strip()

        brace = raw.find("{")
        if brace < 0:
            return original

        obj = None

        # 1) First JSON value from first ``{`` (handles trailing garbage after ``}``).
        try:
            obj, _end = json.JSONDecoder().raw_decode(raw, brace)
        except json.JSONDecodeError:
            pass

        # 2) Slice from first ``{`` to last ``}`` (some models emit extra junk only at end).
        if obj is None:
            try:
                last = raw.rfind("}")
                if last > brace:
                    obj = json.loads(raw[brace : last + 1])
            except json.JSONDecodeError:
                pass

        # 3) Whole blob (e.g. already trimmed to a single object).
        if obj is None and raw.startswith("{"):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                pass

        # 4) JSON with invalid escape sequences (e.g. \mathcal, \beta in LaTeX).
        #    LLMs sometimes wrap translations in {"translation": "..."} even when
        #    instructed not to, and the content contains raw LaTeX whose backslashes
        #    are not valid JSON escapes.  Try to fix the escapes and re-parse.
        if obj is None:
            obj = _try_parse_json_with_latex_escapes(raw, brace)

        if isinstance(obj, dict):
            for key in ("translation", "translated_text", "content", "text", "result"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return original

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
                translated_text = self._unwrap_translation_payload(result.response).strip()
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
                translated_text = self._unwrap_translation_payload(result.response).strip()

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
