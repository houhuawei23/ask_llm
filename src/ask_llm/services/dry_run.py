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

from ask_llm.core.constants import OUTPUT_TOKEN_MULTIPLIERS, TaskKind
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.text_splitter import TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.utils.chunk_balance import (
    plain_text_chunks_by_tokens,
    rebalance_translation_chunks,
)
from ask_llm.utils.pricing import estimate_cost_cny, lookup_pricing
from ask_llm.utils.prompt_resolver import expand_prompt
from ask_llm.utils.token_counter import TokenCounter


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
    kind: str  # "translation" | "batch"
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
        if self.kind == "translation":
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
        if self.kind == "translation":
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
    max_chunk_tokens: int,
    balance_chunks: bool = True,
) -> DryRunFileEstimate | None:
    """Estimate chunks/tokens for one translation input without any API call.

    Mirrors ``TextFileTranslator._prepare_text_file``'s chunking pipeline
    (same splitters, same prompt-overhead reservation, same rebalance pass).
    """
    file_path = Path(file_path)
    file_type = TextSplitter.detect_file_type(str(file_path))
    if file_type not in ("markdown", "text"):
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if not content.strip():
        return None

    from ask_llm.core.translator import TranslationStyle

    translator = Translator(
        target_language=target_language,
        source_language=source_language or "auto",
        style=style if style else TranslationStyle.FORMAL,
        custom_prompt_template=None,
        prompt_file=prompt_file,
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
