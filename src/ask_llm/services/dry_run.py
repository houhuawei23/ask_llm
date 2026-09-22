"""Zero-network cost previews for ``trans``/``batch`` (``--dry-run``, plan 5.2).

The estimates reuse the exact chunking pipeline the real run uses — the memoized
providers catalog for pricing, the shared splitter/rebalance for chunk counts,
and the batch output multipliers for expected output — so a dry run is a real
budget preview of the same code path that would spend money, not a parallel
approximation that drifts from reality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import nbformat

from ask_llm.core.constants import OUTPUT_TOKEN_MULTIPLIERS, TaskKind
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.text_splitter import TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.utils.chunk_balance import (
    plain_text_chunks_by_tokens,
    rebalance_translation_chunks,
)
from ask_llm.utils.notebook_translator import plan_notebook_translation
from ask_llm.utils.pricing import estimate_cost_cny, lookup_pricing
from ask_llm.utils.prompt_resolver import expand_prompt
from ask_llm.utils.token_counter import TokenCounter


def _build_translator(
    target_language: str,
    source_language: str | None,
    style: str | None,
    prompt_file: str | None,
    glossary_pairs: list[tuple[str, str]] | None,
) -> Translator:
    """Build the same Translator the paid run builds (M10/2.25: glossary-aware)."""
    from ask_llm.core.translator import TranslationStyle

    return Translator(
        target_language=target_language,
        source_language=source_language or "auto",
        style=style if style else TranslationStyle.FORMAL,
        custom_prompt_template=None,
        prompt_file=prompt_file,
        glossary_pairs=glossary_pairs or [],
    )


@dataclass
class DryRunFileEstimate:
    """Per-file chunk/token estimate for a translation dry run."""

    path: str
    chunks: int
    input_tokens: int


@dataclass
class DryRunReport:
    """Aggregated zero-network estimate for a would-be paid run."""

    provider: str
    model: str
    kind: str  # "translation" | "batch" | "format"
    files: list[DryRunFileEstimate] = field(default_factory=list)
    task_count: int = 0
    input_tokens: int = 0
    est_output_tokens: int = 0
    est_cost_cny: float | None = None

    def render(self, *, pricing_source: Path | None = None) -> list[str]:
        """Console lines for the report (no leading/trailing blanks)."""
        src = f" ({pricing_source.name})" if pricing_source else ""
        lines = [
            f"[bold]Dry run[/bold] — no API calls were made{src}",
            f"  Provider/model: {self.provider}/{self.model}",
        ]
        if self.kind in ("translation", "format"):
            for est in self.files:
                lines.append(f"  {est.path}: {est.chunks} chunk(s), ≈{est.input_tokens:,} in")
        else:
            lines.append(f"  Tasks: {self.task_count}")
        lines.append(f"  Chunks/requests: {self._request_count():,}")
        lines.append(
            f"  Estimated tokens: input ≈{self.input_tokens:,}, output ≈{self.est_output_tokens:,}"
        )
        if self.est_cost_cny is not None:
            lines.append(f"  Estimated cost: ¥{self.est_cost_cny:.4f}")
        else:
            lines.append("  Estimated cost: unavailable — no pricing entry for this model")
        return lines

    def _request_count(self) -> int:
        if self.kind in ("translation", "format"):
            return sum(est.chunks for est in self.files)
        return self.task_count


def estimate_translation_file(
    file_path: str | Path,
    model: str,
    *,
    target_language: str,
    source_language: str | None = None,
    style: str | None = None,
    prompt_file: str | None = None,
    glossary_pairs: list[tuple[str, str]] | None = None,
    max_chunk_tokens: int,
    balance_chunks: bool = True,
) -> DryRunFileEstimate | None:
    """Estimate chunks/tokens for one translation input without any API call.

    Mirrors the real run's chunking pipeline (same splitters, same
    prompt-overhead reservation, same rebalance pass) for text/markdown via
    ``TextFileTranslator.prepare`` and for notebooks via
    ``plan_notebook_translation`` (M10/2.25: notebooks were silently skipped,
    and the glossary — which widens the prompt and shrinks the chunk budget —
    was ignored, so estimates diverged from the paid run).
    """
    file_path = Path(file_path)
    file_type = TextSplitter.detect_file_type(str(file_path))
    if file_type not in ("markdown", "text", "notebook"):
        return None

    try:
        if file_type == "notebook":
            translator = _build_translator(
                target_language,
                source_language,
                style,
                prompt_file,
                glossary_pairs,
            )
            # Planner raises FileNotFoundError for missing notebooks; a
            # missing/unreadable file estimates as zero rather than failing
            # the whole dry run, matching the text path below.
            try:
                task_data = plan_notebook_translation(
                    str(file_path),
                    model,
                    prompt_template=translator.prompt_template_for_batch(),
                    max_chunk_tokens=max_chunk_tokens,
                    balance_chunks=balance_chunks,
                )
            except (OSError, ValueError, nbformat.ValidationError):
                return None
            if not task_data:
                return None
            input_tokens = sum(
                TokenCounter.count_tokens(content, model) for _, content in task_data
            )
            return DryRunFileEstimate(
                path=str(file_path), chunks=len(task_data), input_tokens=input_tokens
            )

        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if not content.strip():
        return None

    translator = _build_translator(
        target_language, source_language, style, prompt_file, glossary_pairs
    )
    prompt_overhead = TokenCounter.count_tokens(translator.prompt_template_for_batch(), model)

    if file_type == "markdown":
        chunks = MarkdownTokenSplitter(
            model, max_chunk_tokens, prompt_overhead_tokens=prompt_overhead
        ).split(content)
    else:
        chunks = plain_text_chunks_by_tokens(content, model, max_chunk_tokens, prompt_overhead)

    chunks = rebalance_translation_chunks(
        chunks,
        model,
        max_chunk_tokens=max_chunk_tokens,
        enabled=balance_chunks,
        prompt_overhead=prompt_overhead,
    )
    if not chunks:
        return None

    input_tokens = sum(TokenCounter.count_tokens(c.content, model) for c in chunks)
    return DryRunFileEstimate(path=str(file_path), chunks=len(chunks), input_tokens=input_tokens)


def _finalize(
    report: DryRunReport,
    pricing_map: dict[tuple[str, str], dict[str, float]],
) -> DryRunReport:
    kind_enum = TaskKind.TRANSLATION if report.kind == "translation" else TaskKind.BATCH
    multiplier = OUTPUT_TOKEN_MULTIPLIERS.get(kind_enum, OUTPUT_TOKEN_MULTIPLIERS[TaskKind.BATCH])
    report.est_output_tokens = int(report.input_tokens * multiplier)
    row = lookup_pricing(pricing_map, report.provider, report.model)
    if row is not None:
        report.est_cost_cny = estimate_cost_cny(row, report.input_tokens, report.est_output_tokens)
    return report


def estimate_translation_run(
    file_paths: Sequence[str | Path],
    model: str,
    provider: str,
    *,
    target_language: str,
    source_language: str | None = None,
    style: str | None = None,
    prompt_file: str | None = None,
    glossary_pairs: list[tuple[str, str]] | None = None,
    max_chunk_tokens: int,
    balance_chunks: bool = True,
    pricing_map: dict[tuple[str, str], dict[str, float]],
) -> DryRunReport:
    """Build the translation dry-run report for resolved input files."""
    report = DryRunReport(provider=provider, model=model, kind="translation")
    for path in file_paths:
        est = estimate_translation_file(
            path,
            model,
            target_language=target_language,
            source_language=source_language,
            style=style,
            prompt_file=prompt_file,
            glossary_pairs=glossary_pairs,
            max_chunk_tokens=max_chunk_tokens,
            balance_chunks=balance_chunks,
        )
        if est is not None:
            report.files.append(est)
            report.input_tokens += est.input_tokens
    return _finalize(report, pricing_map)


def estimate_batch_run(
    tasks: list[tuple[str, str]],
    provider: str,
    model: str,
    *,
    pricing_map: dict[tuple[str, str], dict[str, float]],
) -> DryRunReport:
    """Build the batch dry-run report from (prompt, content) task pairs.

    Batch tasks are single requests — no chunking — so each task is one
    request with ``expand_prompt``-composed input, matching
    ``GlobalBatchProcessor.process_global_tasks``' tokenization.
    """
    report = DryRunReport(provider=provider, model=model, kind="batch")
    report.task_count = len(tasks)
    for prompt, content in tasks:
        composed = expand_prompt(prompt, content)
        report.input_tokens += TokenCounter.estimate_tokens(composed, model)["token_count"]
    return _finalize(report, pricing_map)


def estimate_format_run(
    file_paths: Sequence[str | Path],
    model: str,
    provider: str,
    *,
    format_type: str = "body",
    max_chunk_tokens: int,
    heading_batch_size: int = 160,
    pricing_map: dict[tuple[str, str], dict[str, float]],
) -> DryRunReport:
    """Build the format dry-run report (E3/2.25).

    Body files reuse the same ``MarkdownTokenSplitter`` chunking the paid run
    uses; title files count heading batches via ``HeadingExtractor`` (one LLM
    request per ``heading_batch_size`` headings). Output uses the FORMAT
    multiplier from ``OUTPUT_TOKEN_MULTIPLIERS``.
    """
    report = DryRunReport(provider=provider, model=model, kind="format")
    output_multiplier = OUTPUT_TOKEN_MULTIPLIERS.get(
        TaskKind.FORMAT, OUTPUT_TOKEN_MULTIPLIERS[TaskKind.BATCH]
    )
    from ask_llm.core.md_heading_formatter import HeadingExtractor

    for path in file_paths:
        path = Path(path)
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if not content.strip():
            continue

        if format_type == "title":
            # One request per heading batch, matching HeadingFormatter.
            headings = HeadingExtractor.extract(content)
            requests = max(1, -(-len(headings) // max(1, heading_batch_size)))
            input_tokens = TokenCounter.count_tokens(content, model)
        else:
            chunks = MarkdownTokenSplitter(model, max_chunk_tokens).split(content)
            requests = len(chunks)
            input_tokens = sum(TokenCounter.count_tokens(c.content, model) for c in chunks)

        report.files.append(
            DryRunFileEstimate(path=str(path), chunks=requests, input_tokens=input_tokens)
        )
        report.input_tokens += input_tokens
    report.task_count = len(report.files)
    report.est_output_tokens = int(report.input_tokens * output_multiplier)
    row = lookup_pricing(pricing_map, provider, model)
    if row is not None:
        report.est_cost_cny = estimate_cost_cny(row, report.input_tokens, report.est_output_tokens)
    return report
