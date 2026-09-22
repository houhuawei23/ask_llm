"""Checkpoint persistence for format command resume capability.

When a chunk/batch API call fails after all retries, the formatter saves a
checkpoint file containing the full context needed to retry only the failed
items later via ``ask-llm format --resume``.

v2 (D5): stores ``original_text`` + per-chunk ``chunk_spans`` so resume can
re-assemble with the same position-aware joiner as a fresh run, instead of the
lossy ``\\n\\n`` fallback. v1 files load unchanged (spans absent → legacy join).

v3 (H5): stores the carved ``frontmatter`` so resume can reattach it; v2 files
load unchanged (frontmatter absent → resume re-extracts from the source file).

v4 (M8): carries ``config_digest`` (sha256 over the source *content*, prompt
template, model, chunk budget and format type) per the checkpoint contract
(``checkpoint.py``). Resume refuses v≤3 files and refuses v4 files whose
digest no longer matches the current source — a stale resume silently
rebuilding output from the old body (and, with ``--inplace``, clobbering the
edited source) is worse than rerunning.

v4 also carries per-failed-chunk ``context_headings`` (audit 3.5): heading
batches retry with the same level-reference context a fresh run used, and the
``take_last_only`` parse stays symmetric. The field defaults to empty, so
early v4 files load unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ask_llm.core.checkpoint import atomic_write_text

CHECKPOINT_VERSION = 4


def compute_format_digest(
    source_file: str | Path,
    *,
    prompt_template: str,
    model: str,
    max_chunk_tokens: int | None,
    format_type: str,
) -> str:
    """Digest a format run's defining inputs for checkpoint consistency checks.

    Hashes the source file *content* (never just its path) plus the prompt
    template, model, chunk budget and format type — editing any of them
    between runs invalidates the old checkpoint instead of letting a stale
    resume mis-map prior results onto changed input.
    """
    h = hashlib.sha256()
    p = Path(source_file)
    if p.is_file():
        h.update(p.read_bytes())
    else:
        h.update(str(source_file).encode("utf-8"))
    h.update(b"\x1f")
    h.update((prompt_template or "").encode("utf-8"))
    h.update(b"\x1e")
    h.update((model or "").encode("utf-8"))
    h.update(b"\x1e")
    h.update(str(max_chunk_tokens).encode("utf-8"))
    h.update(b"\x1e")
    h.update((format_type or "").encode("utf-8"))
    return h.hexdigest()


@dataclass
class FailedChunkInfo:
    """Information about a single failed chunk for checkpoint/resume.

    ``context_headings`` (audit 3.5): heading batches retry with the same
    last-N previous-batch context a fresh run used; empty for body chunks and
    for checkpoints written before the field existed.
    """

    chunk_id: int
    content: str
    prompt_template: str
    error: str
    retry_count: int
    context_headings: list[str] = field(default_factory=list)


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
    # D5: original body text + per-chunk spans so resume can use the
    # position-aware joiner (lossless) instead of the "\n\n" fallback.
    original_text: str = ""
    chunk_spans: list[dict[str, Any]] = field(default_factory=list)
    # H5: the frontmatter carved out before chunking, reattached verbatim on
    # resume. Empty for pre-v3 checkpoints (resume re-extracts it instead).
    frontmatter: str = ""
    # M8: digest of the run's defining inputs (see compute_format_digest).
    # Empty for pre-v4 checkpoints; resume refuses those outright.
    config_digest: str = ""

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
            "config_digest": self.config_digest,
            "failed_chunks": [
                {
                    "chunk_id": fc.chunk_id,
                    "content": fc.content,
                    "prompt_template": fc.prompt_template,
                    "error": fc.error,
                    "retry_count": fc.retry_count,
                    "context_headings": list(fc.context_headings),
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
            "original_text": self.original_text,
            "chunk_spans": list(self.chunk_spans),
            "frontmatter": self.frontmatter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FormatCheckpoint:
        """Deserialize checkpoint from dictionary.

        Tolerates v1 files (no ``original_text`` / ``chunk_spans``): the resume
        path falls back to the legacy ``\\n\\n`` joiner when spans are absent.
        """
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
                    context_headings=list(fc.get("context_headings", [])),
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
            original_text=data.get("original_text", ""),
            chunk_spans=list(data.get("chunk_spans", [])),
            frontmatter=data.get("frontmatter", ""),
            config_digest=data.get("config_digest", ""),
        )

    def save(self, path: str | Path) -> None:
        """Atomically save checkpoint to JSON file."""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        atomic_write_text(path, payload)
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
