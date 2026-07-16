"""Per-file Jupyter notebook translation collaborator (P4.5).

Owns the single-notebook flow (markdown cells only) via
``utils.notebook_translator.NotebookTranslator``. Split out of
``TranslationService``; per-run results are returned inside
``TranslationJobResult.results`` instead of mutating shared state.
"""

from __future__ import annotations

from pathlib import Path

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch import ModelConfig
from ask_llm.core.models import AppConfig
from ask_llm.core.translator import Translator
from ask_llm.services.translation_options import TranslationJobResult, TranslationOptions
from ask_llm.utils.console import console
from ask_llm.utils.fallback_chain import build_fallback_chain
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.notebook_translator import NotebookTranslator
from ask_llm.utils.pricing import format_cost_estimate

PricingMap = dict[tuple[str, str], dict[str, float]]


class NotebookFileTranslator:
    """Translate a single Jupyter notebook (markdown cells only)."""

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

    def translate(
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

        fallback_configs: list[ModelConfig] = []
        if options.use_fallback and self.app_config is not None:
            fallback_configs = build_fallback_chain(self.app_config, model_config)

        notebook_translator = NotebookTranslator(
            translator=translator,
            model_config=model_config,
            fallback_configs=fallback_configs,
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
            results=list(notebook_translator.last_results),
        )
