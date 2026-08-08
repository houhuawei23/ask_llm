"""Translation orchestration service.

Moves the core translation workflow (resolve inputs, split, translate chunks,
export) out of the CLI command so the command module stays focused on argument
parsing and user-facing error handling.

P4.5: the service is an aggregator. Per-file work lives in the collaborators
``TextFileTranslator`` and ``NotebookFileTranslator``; per-chunk results
travel inside ``TranslationJobResult.results`` and are aggregated on the main
thread, so the old cross-thread ``_batch_results`` mutation is gone.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.batch import BatchResult
from ask_llm.core.execution_report import build_report_from_batch_results
from ask_llm.core.models import AppConfig
from ask_llm.core.text_splitter import TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.services.notebook_file_translator import NotebookFileTranslator
from ask_llm.services.text_file_translator import (
    TextFileTranslator,
    TextTranslationJob,
)
from ask_llm.services.translation_options import (
    TranslationJobResult,
    TranslationOptions,
    TranslationSessionResult,
)
from ask_llm.utils.console import console
from ask_llm.utils.pricing import format_cost_estimate

PricingMap = dict[tuple[str, str], dict[str, float]]

# Backward-compat aliases (tests / external imports).
_TextTranslationJob = TextTranslationJob

__all__ = [
    "TranslationJobResult",
    "TranslationOptions",
    "TranslationService",
    "TranslationSessionResult",
    "_TextTranslationJob",
]


class TranslationService:
    """High-level service for translating files via LLM APIs."""

    def __init__(
        self,
        config_manager: ConfigManager,
        unified_config: UnifiedConfig,
        *,
        provider: str,
        model: str,
        pricing_map: PricingMap | None = None,
        pricing_source: Path | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        """Initialize the translation service.

        Args:
            config_manager: Active config manager (provider/model already resolved).
            unified_config: Loaded unified configuration.
            provider: Resolved provider name.
            model: Resolved model name.
            pricing_map: Optional pricing data for cost estimates.
            pricing_source: Optional path/label of the pricing source.
            app_config: Loaded application config; required for provider fallback chain.
        """
        self.config_manager = config_manager
        self.unified_config = unified_config
        self.provider = provider
        self.model = model
        self.pricing_map = pricing_map or {}
        self.pricing_source = pricing_source
        self.app_config = app_config
        # Per-chunk results for the session report. Written ONLY on the main
        # thread (via _accumulate) — P4.5 removed cross-thread mutation.
        self._batch_results: list[BatchResult] = []
        self._text_translator = TextFileTranslator(
            config_manager,
            provider=provider,
            model=model,
            pricing_map=pricing_map,
            pricing_source=pricing_source,
            app_config=app_config,
        )
        self._notebook_file_translator = NotebookFileTranslator(
            config_manager,
            provider=provider,
            model=model,
            pricing_map=pricing_map,
            pricing_source=pricing_source,
            app_config=app_config,
        )

    def translate_files(
        self,
        files: list[str],
        options: TranslationOptions,
        *,
        output: str | None = None,
        force: bool = False,
        stream: bool = False,
        stream_api: bool = True,
        glossary: str | None = None,
        translated_suffix: str | None = None,
    ) -> TranslationSessionResult:
        """Translate one or more files and export the results.

        Args:
            files: Input file path(s), glob patterns, or directories.
            options: Resolved translation options.
            output: Optional output file or directory path.
            force: Whether to overwrite existing output files.
            stream: Stream progress to console (progress bars only).
            stream_api: Use streaming API calls.
            glossary: Optional path to a glossary file.
            translated_suffix: Optional suffix for output filenames.

        Returns:
            TranslationSessionResult aggregating per-file outcomes.

        Raises:
            FileNotFoundError: If no input files are found.
            ValueError: If input or configuration is invalid.
        """
        from ask_llm.utils.path_resolver import _is_directory_output, _resolve_trans_input_paths

        session_result = TranslationSessionResult()
        _t0 = time.perf_counter()

        glossary_pairs: list[tuple[str, str]] = []
        if glossary:
            glossary_pairs = Translator.load_glossary(glossary)
            console.print_info(
                f"Glossary: {len(glossary_pairs)} term pair(s) loaded from {glossary}"
            )

        resolved_files = _resolve_trans_input_paths(
            files,
            translatable_extensions=options.translatable_extensions,
            recursive_dir=options.recursive_dir,
        )
        if not resolved_files:
            raise FileNotFoundError("No files found to translate")

        console.print_info(f"Found {len(resolved_files)} file(s) to translate")
        console.print_info(
            f"File-level concurrency: {options.max_parallel_files} file(s) in parallel"
        )
        console.print_info(
            f"Chunk-level concurrency: up to {options.threads} concurrent API call(s) per file"
        )

        if output and _is_directory_output(output, files, len(resolved_files)):
            output_is_dir = True
            Path(output).mkdir(parents=True, exist_ok=True)
        else:
            output_is_dir = False

        effective_suffix = translated_suffix or self.unified_config.file.translated_suffix

        notebook_files: list[str] = []
        text_jobs: list[TextTranslationJob] = []

        for file_path in resolved_files:
            file_type = TextSplitter.detect_file_type(file_path)
            if file_type == "notebook":
                notebook_files.append(file_path)
            elif file_type in ("markdown", "text"):
                job = self._text_translator.prepare(
                    file_path,
                    options,
                    output=output,
                    output_is_dir=output_is_dir,
                    effective_suffix=effective_suffix,
                    glossary_pairs=glossary_pairs,
                    stream=stream,
                )
                if job:
                    text_jobs.append(job)
            else:
                console.print_warning(
                    f"Unsupported file type: {Path(file_path).suffix}. "
                    f"Only .txt, .md, and .ipynb files are supported. Skipping."
                )

        if not text_jobs and not notebook_files:
            raise FileNotFoundError("No translatable files found")

        self._print_prompt_preview(options, glossary_pairs)

        if options.max_parallel_files <= 1:
            for job in text_jobs:
                result = self._text_translator.translate_and_export(
                    job, options, force=force, stream=stream, stream_api=stream_api
                )
                self._accumulate(session_result, result)
            for file_path in notebook_files:
                result = self._notebook_file_translator.translate(
                    file_path,
                    options,
                    output=output,
                    output_is_dir=output_is_dir,
                    effective_suffix=effective_suffix,
                    force=force,
                    stream=stream,
                    stream_api=stream_api,
                )
                self._accumulate(session_result, result)
        else:
            with ThreadPoolExecutor(max_workers=options.max_parallel_files) as executor:
                futures: dict[Future, str] = {}
                for job in text_jobs:
                    futures[
                        executor.submit(
                            self._text_translator.translate_and_export,
                            job,
                            options,
                            force=force,
                            stream=stream,
                            stream_api=stream_api,
                        )
                    ] = job.file_path
                for file_path in notebook_files:
                    futures[
                        executor.submit(
                            self._notebook_file_translator.translate,
                            file_path,
                            options,
                            output=output,
                            output_is_dir=output_is_dir,
                            effective_suffix=effective_suffix,
                            force=force,
                            stream=stream,
                            stream_api=stream_api,
                        )
                    ] = file_path

                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        console.print_error(f"Failed to translate {file_path}: {e}")
                        logger.exception("Translation error")
                        result = TranslationJobResult(
                            file_path=file_path,
                            output_path=None,
                            input_tokens=0,
                            output_tokens=0,
                            success=False,
                            error=str(e),
                        )
                    self._accumulate(session_result, result)

        self._print_session_total(session_result)
        session_result.report = build_report_from_batch_results(
            "trans",
            self._batch_results,
            metadata={"files": files},
        )
        logger.debug("TranslationService wall time: {:.2f}s", time.perf_counter() - _t0)
        return session_result

    # --- Backward-compat delegates (tests / external callers) ---

    def _prepare_text_file(self, file_path, options, **kwargs):
        """Delegate to TextFileTranslator.prepare (backward compat)."""
        return self._text_translator.prepare(file_path, options, **kwargs)

    def _translate_and_export_text_file(self, job, options, **kwargs):
        """Delegate to TextFileTranslator.translate_and_export (backward compat)."""
        return self._text_translator.translate_and_export(job, options, **kwargs)

    # --- Presentation helpers ---

    def _print_prompt_preview(
        self,
        options: TranslationOptions,
        glossary_pairs: list[tuple[str, str]],
    ) -> None:
        """Print a token-count preview of the prompt template."""
        prompt_preview = Translator(
            target_language=options.target_language,
            source_language=options.source_language,
            style=options.style,
            custom_prompt_template=None,
            prompt_file=options.prompt_file,
            glossary_pairs=glossary_pairs,
        )
        prompt_template_tokens = prompt_preview.count_prompt_template_tokens(self.model)
        pf = options.prompt_file
        if pf:
            prompt_label = pf if pf.startswith("@") else str(Path(pf).expanduser().name)
        else:
            prompt_label = f"内置样式 ({options.style})"
        console.print_info(
            f"提示词模板「{prompt_label}」指令部分(已替换语言占位、不含待译正文)≈ "
            f"{prompt_template_tokens} tokens (tiktoken · {self.model})"
        )

    def _print_session_total(self, session_result: TranslationSessionResult) -> None:
        """Print aggregate token/cost summary when more than one file was processed."""
        processed_file_count = len([r for r in session_result.job_results if r.success]) + len(
            [r for r in session_result.job_results if not r.success]
        )
        if processed_file_count > 1:
            console.print()
            console.print("[bold]Session total (all files)[/bold]")
            console.print(
                f"  Files: {session_result.successful_files} succeeded, "
                f"{session_result.failed_files} failed, "
                f"{session_result.total_retries} retries"
            )
            console.print(
                format_cost_estimate(
                    self.provider,
                    self.model,
                    session_result.total_input_tokens,
                    session_result.total_output_tokens,
                    self.pricing_map,
                    pricing_source=self.pricing_source,
                )
            )

    def _accumulate(
        self,
        session_result: TranslationSessionResult,
        job_result: TranslationJobResult,
    ) -> None:
        """Add a single-file result to the session aggregate (main thread only)."""
        session_result.job_results.append(job_result)
        session_result.total_retries += job_result.retries
        if job_result.success:
            session_result.successful_files += 1
            session_result.total_input_tokens += job_result.input_tokens
            session_result.total_output_tokens += job_result.output_tokens
        else:
            session_result.failed_files += 1
        self._batch_results.extend(job_result.results)

    def export_report(self, report_path: str | None) -> str | None:
        """Export the execution report to ``report_path`` if available.

        Args:
            report_path: Destination path for the JSON report.

        Returns:
            The exported path, or ``None`` if no report was generated or no path
            was requested.
        """
        if not report_path:
            return None
        if self._batch_results:
            report = build_report_from_batch_results(
                "trans",
                self._batch_results,
                metadata={"provider": self.provider, "model": self.model},
            )
            report.to_json_file(report_path)
            console.print_info(f"Execution report saved to: {report_path}")
            return report_path
        return None
