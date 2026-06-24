"""Translation orchestration service.

Moves the core translation workflow (resolve inputs, split, translate chunks,
export) out of the CLI command so the command module stays focused on argument
parsing and user-facing error handling.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ask_llm.cli.common import _is_directory_output, _resolve_trans_input_paths
from ask_llm.config.manager import ConfigManager
from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.text_splitter import TextChunk, TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.utils.chunk_balance import plain_text_chunks_by_tokens, rebalance_translation_chunks
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.notebook_translator import NotebookTranslator
from ask_llm.utils.pricing import format_cost_estimate
from ask_llm.utils.translation_exporter import TranslationExporter

PricingMap = dict[tuple[str, str], dict[str, float]]


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


@dataclass
class TranslationJobResult:
    """Result of translating a single input file."""

    file_path: str
    output_path: str | None
    input_tokens: int
    output_tokens: int
    success: bool
    error: str | None = None
    retries: int = 0


@dataclass
class TranslationSessionResult:
    """Aggregate result of a full translation session."""

    job_results: list[TranslationJobResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_retries: int = 0


@dataclass
class _TextTranslationJob:
    """Internal container for a prepared text/markdown translation job."""

    file_path: str
    file_type: str
    chunks: list[TextChunk]
    tasks: list[BatchTask]
    output_path: str


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
    ) -> None:
        """Initialize the translation service.

        Args:
            config_manager: Active config manager (provider/model already resolved).
            unified_config: Loaded unified configuration.
            provider: Resolved provider name.
            model: Resolved model name.
            pricing_map: Optional pricing data for cost estimates.
            pricing_source: Optional path/label of the pricing source.
        """
        self.config_manager = config_manager
        self.unified_config = unified_config
        self.provider = provider
        self.model = model
        self.pricing_map = pricing_map or {}
        self.pricing_source = pricing_source

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
        text_jobs: list[_TextTranslationJob] = []

        for file_path in resolved_files:
            file_type = TextSplitter.detect_file_type(file_path)
            if file_type == "notebook":
                notebook_files.append(file_path)
            elif file_type in ("markdown", "text"):
                job = self._prepare_text_file(
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
                result = self._translate_and_export_text_file(
                    job, options, force=force, stream=stream, stream_api=stream_api
                )
                self._accumulate(session_result, result)
            for file_path in notebook_files:
                result = self._translate_notebook_file(
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
                            self._translate_and_export_text_file,
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
                            self._translate_notebook_file,
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
        logger.debug("TranslationService wall time: {:.2f}s", time.perf_counter() - _t0)
        return session_result

    def _prepare_text_file(
        self,
        file_path: str,
        options: TranslationOptions,
        *,
        output: str | None,
        output_is_dir: bool,
        effective_suffix: str,
        glossary_pairs: list[tuple[str, str]],
        stream: bool,
    ) -> _TextTranslationJob | None:
        """Read, split, and build translation tasks for a text/markdown file."""
        console.print()
        console.print(f"[bold]Preparing: {file_path}[/bold]")

        file_type = TextSplitter.detect_file_type(file_path)
        if file_type not in ("markdown", "text"):
            console.print_warning(
                f"Unsupported file type: {Path(file_path).suffix}. "
                f"Only .txt, .md, and .ipynb files are supported. Skipping."
            )
            return None

        try:
            content = FileHandler.read(file_path, show_progress=not stream)
        except Exception as e:
            console.print_error(f"Failed to read file {file_path}: {e}")
            return None

        if not content.strip():
            console.print_warning(f"File {file_path} is empty. Skipping.")
            return None

        if file_type == "markdown":
            chunks = MarkdownTokenSplitter(self.model, options.max_chunk_tokens).split(content)
        else:
            chunks = plain_text_chunks_by_tokens(content, self.model, options.max_chunk_tokens)

        if not chunks:
            console.print_warning(f"No chunks created from {file_path}. Skipping.")
            return None

        n_before = len(chunks)
        chunks = rebalance_translation_chunks(
            chunks,
            self.model,
            max_chunk_tokens=options.max_chunk_tokens,
            min_merge_tokens=options.min_chunk_merge_tokens,
            enabled=options.balance_translation_chunks,
        )
        if len(chunks) != n_before:
            console.print_info(
                f"Token rebalance: {n_before} -> {len(chunks)} chunk(s) "
                f"(cap ≈ {options.max_chunk_tokens} tokens)"
            )
        else:
            console.print_info(f"Split into {len(chunks)} chunk(s)")

        translator = Translator(
            target_language=options.target_language,
            source_language=options.source_language,
            style=options.style,
            custom_prompt_template=None,
            prompt_file=options.prompt_file,
            glossary_pairs=glossary_pairs,
        )

        model_config = ModelConfig(
            provider=self.provider,
            model=self.model,
            temperature=options.temperature,
            max_tokens=options.max_output_tokens,
        )

        tasks = translator.create_translation_tasks(chunks, model_config)
        output_path = self._resolve_output_path(
            file_path, output=output, output_is_dir=output_is_dir, suffix=effective_suffix
        )

        return _TextTranslationJob(
            file_path=file_path,
            file_type=file_type,
            chunks=chunks,
            tasks=tasks,
            output_path=output_path,
        )

    def _translate_and_export_text_file(
        self,
        job: _TextTranslationJob,
        options: TranslationOptions,
        *,
        force: bool,
        stream: bool,
        stream_api: bool,
    ) -> TranslationJobResult:
        """Translate and export a single text/markdown file."""
        console.print()
        console.print(f"[bold]Translating: {job.file_path}[/bold]")

        try:
            results, processor = run_global_batch_tasks(
                job.tasks,
                self.config_manager,
                max_workers=options.threads,
                max_retries=options.retries,
                show_progress=not stream,
                clamp_workers_to_task_count=True,
                stream_api=stream_api,
            )
        except Exception as e:
            console.print_error(f"Failed to translate {job.file_path}: {e}")
            logger.exception("Translation error")
            return TranslationJobResult(
                file_path=job.file_path,
                output_path=job.output_path,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error=str(e),
            )

        failed_count = sum(1 for r in results if r.status.value == "failed")
        successful_chunks = sum(1 for r in results if r.status.value == "success")
        if failed_count > 0:
            console.print_warning(f"{failed_count} chunk(s) failed to translate")
        if (
            successful_chunks == 0
            and failed_count > 0
            and getattr(processor, "_auth_error_logged", False)
        ):
            console.print_error(f"翻译失败: API 认证错误, {job.file_path} 未产生有效译文。")
            return TranslationJobResult(
                file_path=job.file_path,
                output_path=job.output_path,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error="API authentication error",
            )

        retries = getattr(processor, "last_metrics", None)
        retry_count = retries.retried if retries is not None else 0

        return self._export_text_file(
            job,
            results,
            options.preserve_format,
            options.include_original,
            force=force,
            retries=retry_count,
        )

    def _export_text_file(
        self,
        job: _TextTranslationJob,
        results: list[BatchResult],
        preserve_format: bool,
        include_original: bool,
        *,
        force: bool,
        retries: int = 0,
    ) -> TranslationJobResult:
        """Export translated chunks for a single text/markdown file."""
        failed_count = sum(1 for r in results if r.status.value == "failed")
        successful_chunks = sum(1 for r in results if r.status.value == "success")

        if failed_count > 0:
            console.print_warning(f"{failed_count} chunk(s) failed to translate")
        if successful_chunks == 0 and failed_count > 0:
            console.print_error(f"翻译失败: {job.file_path} 所有分块均失败。")
            return TranslationJobResult(
                file_path=job.file_path,
                output_path=job.output_path,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error="All chunks failed",
            )

        exporter = TranslationExporter(
            chunks=job.chunks,
            results=results,
            preserve_format=preserve_format,
            include_original=include_original,
        )

        output_ext = Path(job.output_path).suffix.lower()
        output_format = None
        if output_ext == ".json":
            output_format = "json"
        elif output_ext in (".md", ".markdown"):
            output_format = "markdown"

        try:
            output_file = Path(job.output_path)
            if output_file.exists() and not force:
                raise FileExistsError(
                    f"Output file already exists: {job.output_path}. Use --force to overwrite."
                )

            exported_path = exporter.export(job.output_path, format_type=output_format)
            console.print_success(f"Translation saved to: {exported_path}")

            console.print(f"  Successful: {successful_chunks}/{len(results)}")
            if failed_count > 0:
                console.print_warning(f"  Failed: {failed_count}/{len(results)}")

            total_in = sum(
                r.metadata.input_tokens
                for r in results
                if r.metadata and r.status.value == "success"
            )
            total_out = sum(
                r.metadata.output_tokens
                for r in results
                if r.metadata and r.status.value == "success"
            )
            console.print(
                format_cost_estimate(
                    self.provider,
                    self.model,
                    total_in,
                    total_out,
                    self.pricing_map,
                    pricing_source=self.pricing_source,
                )
            )
            return TranslationJobResult(
                file_path=job.file_path,
                output_path=str(exported_path),
                input_tokens=total_in,
                output_tokens=total_out,
                success=True,
                retries=retries,
            )

        except FileExistsError:
            console.print_error(
                f"Output file already exists: {job.output_path}. Use --force to overwrite."
            )
            return TranslationJobResult(
                file_path=job.file_path,
                output_path=job.output_path,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error="Output file already exists",
            )
        except Exception as e:
            console.print_error(f"Failed to export translation: {e}")
            logger.exception("Export error")
            return TranslationJobResult(
                file_path=job.file_path,
                output_path=job.output_path,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error=str(e),
            )

    def _translate_notebook_file(
        self,
        file_path: str,
        options: TranslationOptions,
        *,
        output: str | None,
        output_is_dir: bool,
        effective_suffix: str,
        force: bool,
        stream: bool,
        stream_api: bool,
    ) -> TranslationJobResult:
        """Process a single Jupyter notebook (markdown cells only)."""
        console.print()
        console.print(f"[bold]Processing: {file_path}[/bold]")

        output_path: str
        if output:
            output_path = output
            if output_is_dir or Path(output).is_dir():
                input_file = Path(file_path)
                output_name = f"{input_file.stem}{effective_suffix}{input_file.suffix}"
                output_path = str(Path(output) / output_name)
        else:
            output_path = FileHandler.generate_output_path(file_path, suffix=effective_suffix)

        output_file = Path(output_path)
        if output_file.exists() and not force:
            console.print_error(
                f"Output file already exists: {output_path}. Use --force to overwrite."
            )
            return TranslationJobResult(
                file_path=file_path,
                output_path=output_path,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error="Output file already exists",
            )

        translator = Translator(
            target_language=options.target_language,
            source_language=options.source_language,
            style=options.style,
            custom_prompt_template=None,
            prompt_file=options.prompt_file,
        )

        model_config = ModelConfig(
            provider=self.provider,
            model=self.model,
            temperature=options.temperature,
            max_tokens=options.max_output_tokens,
        )

        notebook_translator = NotebookTranslator(
            translator=translator,
            model_config=model_config,
        )

        try:
            successful, failed, total_in, total_out = notebook_translator.translate_notebook(
                input_path=file_path,
                output_path=output_path,
                config_manager=self.config_manager,
                max_workers=options.threads,
                max_retries=options.retries,
                show_progress=not stream,
                balance_chunks=options.balance_translation_chunks,
                max_chunk_tokens=options.max_chunk_tokens,
                min_chunk_merge_tokens=options.min_chunk_merge_tokens,
                stream_api=stream_api,
            )
        except RuntimeError as e:
            if "API authentication failed" in str(e):
                console.print_error(str(e))
                return TranslationJobResult(
                    file_path=file_path,
                    output_path=output_path,
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=str(e),
                )
            raise

        total = successful + failed
        console.print_success(f"Translation saved to: {output_path}")
        console.print(f"  Successful: {successful}/{total}")
        if failed > 0:
            console.print_warning(f"  Failed: {failed}/{total}")
        console.print(
            format_cost_estimate(
                self.provider,
                self.model,
                total_in,
                total_out,
                self.pricing_map,
                pricing_source=self.pricing_source,
            )
        )
        return TranslationJobResult(
            file_path=file_path,
            output_path=output_path,
            input_tokens=total_in,
            output_tokens=total_out,
            success=True,
        )

    def _resolve_output_path(
        self,
        file_path: str,
        *,
        output: str | None,
        output_is_dir: bool,
        suffix: str,
    ) -> str:
        """Determine the output path for a single input file."""
        if output:
            output_path = output
            if output_is_dir:
                input_file = Path(file_path)
                output_name = f"{input_file.stem}{suffix}{input_file.suffix}"
                output_path = str(Path(output) / output_name)
        else:
            output_path = FileHandler.generate_output_path(file_path, suffix=suffix)
        return output_path

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
        """Add a single-file result to the session aggregate."""
        session_result.job_results.append(job_result)
        session_result.total_retries += job_result.retries
        if job_result.success:
            session_result.successful_files += 1
            session_result.total_input_tokens += job_result.input_tokens
            session_result.total_output_tokens += job_result.output_tokens
        else:
            session_result.failed_files += 1
