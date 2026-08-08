"""Per-file text/markdown translation collaborator (P4.5).

Owns the read → split → build tasks → checkpointed translate → export flow
for a single text or markdown file. Split out of ``TranslationService``,
which is now an immutable-ish aggregator delegating per-file work here.

Thread-safety: instances carry only immutable run configuration (provider,
model, pricing, app config). Per-run results are returned to the caller
(inside ``TranslationJobResult.results``) instead of being accumulated into
shared mutable state, so files can be processed concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import BatchResult, BatchTask, ModelConfig
from ask_llm.core.command_runner import run_with_checkpoint
from ask_llm.core.markdown_token_splitter import MarkdownTokenSplitter
from ask_llm.core.models import AppConfig
from ask_llm.core.text_splitter import TextChunk, TextSplitter
from ask_llm.core.translator import Translator
from ask_llm.services.translation_options import (
    TranslationJobResult,
    TranslationOptions,
)
from ask_llm.utils.chunk_balance import plain_text_chunks_by_tokens, rebalance_translation_chunks
from ask_llm.utils.console import console
from ask_llm.utils.fallback_chain import build_fallback_chain
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.pricing import format_cost_estimate
from ask_llm.utils.translation_exporter import TranslationExporter

PricingMap = dict[tuple[str, str], dict[str, float]]


@dataclass
class TextTranslationJob:
    """Container for a prepared text/markdown translation job."""

    file_path: str
    file_type: str
    chunks: list[TextChunk]
    tasks: list[BatchTask]
    output_path: str


class TextFileTranslator:
    """Translate a single text/markdown file end to end."""

    def __init__(
        self,
        config_manager: ConfigManager,
        *,
        provider: str,
        model: str,
        pricing_map: PricingMap | None = None,
        pricing_source: Path | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        self.config_manager = config_manager
        self.provider = provider
        self.model = model
        self.pricing_map = pricing_map or {}
        self.pricing_source = pricing_source
        self.app_config = app_config

    def prepare(
        self,
        file_path: str,
        options: TranslationOptions,
        *,
        output: str | None,
        output_is_dir: bool,
        effective_suffix: str,
        glossary_pairs: list[tuple[str, str]],
        stream: bool,
    ) -> TextTranslationJob | None:
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
        if options.use_fallback and self.app_config is not None:
            fallback_configs = build_fallback_chain(self.app_config, model_config)
            for task in tasks:
                task.fallback_model_configs = fallback_configs
        output_path = self.resolve_output_path(
            file_path, output=output, output_is_dir=output_is_dir, suffix=effective_suffix
        )

        return TextTranslationJob(
            file_path=file_path,
            file_type=file_type,
            chunks=chunks,
            tasks=tasks,
            output_path=output_path,
        )

    @staticmethod
    def resolve_output_path(
        file_path: str,
        *,
        output: str | None,
        output_is_dir: bool,
        suffix: str,
    ) -> str:
        """Resolve the output path for a text/markdown translation."""
        if output:
            if output_is_dir or Path(output).is_dir():
                input_file = Path(file_path)
                output_name = f"{input_file.stem}{suffix}{input_file.suffix}"
                return str(Path(output) / output_name)
            return output
        return FileHandler.generate_output_path(file_path, suffix=suffix)

    @staticmethod
    def checkpoint_path(output_path: str) -> str:
        """Return the default checkpoint path for a translation output file."""
        return f"{output_path}.trans_checkpoint.json"

    def translate_and_export(
        self,
        job: TextTranslationJob,
        options: TranslationOptions,
        *,
        force: bool,
        stream: bool,
        stream_api: bool,
    ) -> TranslationJobResult:
        """Translate and export a single text/markdown file."""
        console.print()
        console.print(f"[bold]Translating: {job.file_path}[/bold]")

        checkpoint_path = self.checkpoint_path(job.output_path)

        # Shared checkpoint lifecycle (P4.1). Canonical drift resolution:
        # early-return with all chunks previously completed now KEEPS the
        # checkpoint (batch behavior) instead of unlinking it.
        try:
            outcome = run_with_checkpoint(
                command="trans",
                config_digest=job.file_path,
                checkpoint_path=checkpoint_path,
                tasks=job.tasks,
                config_manager=self.config_manager,
                resume=bool(options.resume and Path(checkpoint_path).exists()),
                max_retries=options.retries,
                max_workers=options.threads,
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

        if outcome.all_previously_completed:
            console.print_info("All chunks already translated according to checkpoint.")
            results = sorted(outcome.results, key=lambda r: r.task_id)
            job_result = self.export_text_file(
                job,
                results,
                options.preserve_format,
                options.include_original,
                force=force,
                retries=0,
            )
            job_result.results = results
            return job_result

        processor = outcome.processor
        results = sorted(outcome.results, key=lambda r: r.task_id)

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
                results=results,
            )

        metrics = getattr(processor, "last_metrics", None)
        retry_count = metrics.retried if metrics is not None else 0

        if outcome.interrupted:
            console.print_warning(
                f"翻译中断: 已保存进度到 {checkpoint_path}，使用 --resume 继续。"
            )

        job_result = self.export_text_file(
            job,
            results,
            options.preserve_format,
            options.include_original,
            force=force,
            retries=retry_count,
        )
        job_result.results = results
        return job_result

    def export_text_file(
        self,
        job: TextTranslationJob,
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
