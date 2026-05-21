"""Checkpoint persistence for format command resume capability.

When a chunk/batch API call fails after all retries, the formatter saves a
checkpoint file containing the full context needed to retry only the failed
items later via ``ask-llm format --resume``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

CHECKPOINT_VERSION = 1


@dataclass
class FailedChunkInfo:
    """Information about a single failed chunk for checkpoint/resume."""

    chunk_id: int
    content: str
    prompt_template: str
    error: str
    retry_count: int


@dataclass
class SuccessfulChunkInfo:
    """Information about a successfully formatted chunk."""

    chunk_id: int
    formatted_content: str


@dataclass
class FormatCheckpoint:
    """Full checkpoint state for a format operation."""

    version: int
    source_file: str
    format_type: str  # "body" or "title"
    model: str
    prompt_template: str
    max_chunk_tokens: int | None
    created_at: str
    failed_chunks: list[FailedChunkInfo]
    successful_chunks: list[SuccessfulChunkInfo]

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "version": self.version,
            "source_file": self.source_file,
            "format_type": self.format_type,
            "model": self.model,
            "prompt_template": self.prompt_template,
            "max_chunk_tokens": self.max_chunk_tokens,
            "created_at": self.created_at,
            "failed_chunks": [
                {
                    "chunk_id": fc.chunk_id,
                    "content": fc.content,
                    "prompt_template": fc.prompt_template,
                    "error": fc.error,
                    "retry_count": fc.retry_count,
                }
                for fc in self.failed_chunks
            ],
            "successful_chunks": [
                {
                    "chunk_id": sc.chunk_id,
                    "formatted_content": sc.formatted_content,
                }
                for sc in self.successful_chunks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FormatCheckpoint:
        """Deserialize checkpoint from dictionary."""
        return cls(
            version=data.get("version", CHECKPOINT_VERSION),
            source_file=data["source_file"],
            format_type=data["format_type"],
            model=data["model"],
            prompt_template=data["prompt_template"],
            max_chunk_tokens=data.get("max_chunk_tokens"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            failed_chunks=[
                FailedChunkInfo(
                    chunk_id=fc["chunk_id"],
                    content=fc["content"],
                    prompt_template=fc["prompt_template"],
                    error=fc["error"],
                    retry_count=fc["retry_count"],
                )
                for fc in data.get("failed_chunks", [])
            ],
            successful_chunks=[
                SuccessfulChunkInfo(
                    chunk_id=sc["chunk_id"],
                    formatted_content=sc["formatted_content"],
                )
                for sc in data.get("successful_chunks", [])
            ],
        )

    def save(self, path: str | Path) -> None:
        """Save checkpoint to JSON file."""
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> FormatCheckpoint:
        """Load checkpoint from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def generate_checkpoint_path(source_file: str, format_type: str) -> Path:
    """Generate a default checkpoint file path next to the source file.

    Example: ``doc.md`` → ``doc.md.format_checkpoint.json``
    """
    p = Path(source_file)
    return p.parent / f"{p.name}.{format_type}_checkpoint.json"
