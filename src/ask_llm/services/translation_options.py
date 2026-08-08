"""Shared translation option/result dataclasses (P4.5).

Home of the value types used by ``TranslationService`` and its per-file
collaborators (``TextFileTranslator`` / ``NotebookFileTranslator``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ask_llm.core.batch import BatchResult
from ask_llm.core.execution_report import ExecutionReport


@dataclass
class TranslationOptions:
    """Resolved translation options used by TranslationService."""

    target_language: str
    source_language: str
    style: str
    threads: int
    max_parallel_files: int
    retries: int
    balance_translation_chunks: bool
    max_chunk_tokens: int
    min_chunk_merge_tokens: int
    max_output_tokens: int
    preserve_format: bool
    include_original: bool
    temperature: float | None
    translatable_extensions: list[str]
    recursive_dir: bool
    prompt_file: str | None = None
    resume: bool = False
    use_fallback: bool = True


@dataclass
class TranslationJobResult:
    """Result of translating a single input file.

    ``results`` carries the per-chunk ``BatchResult`` list for session-report
    aggregation (P4.5: results travel with the job result instead of being
    accumulated into shared mutable state from worker threads).
    """

    file_path: str
    output_path: str | None
    input_tokens: int
    output_tokens: int
    success: bool
    error: str | None = None
    retries: int = 0
    results: list[BatchResult] = field(default_factory=list)


@dataclass
class TranslationSessionResult:
    """Aggregate result of a full translation session."""

    job_results: list[TranslationJobResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_retries: int = 0
    report: ExecutionReport | None = None
