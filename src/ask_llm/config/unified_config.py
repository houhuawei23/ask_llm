"""Unified configuration model for default_config.yml."""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class GeneralConfig(BaseModel):
    """General/global default configuration."""

    default_prompt_template: str = Field(
        default="Please process the following text:\n\n{content}",
        description="Default prompt template for ask command",
    )
    default_output_filename: str = Field(
        default="output.txt",
        description="Default output filename for non-file input in ask command",
    )
    stream_default: bool = Field(
        default=True,
        description="Default streaming mode for ask command",
    )


class TranslationConfig(BaseModel):
    """Translation default configuration."""

    model_config = ConfigDict(extra="ignore")

    target_language: str = Field(default="zh", description="Target language code")
    source_language: str = Field(
        default="auto",
        description="Source language code (auto for auto-detection)",
    )
    style: str = Field(
        default="formal",
        description="Translation style: formal, casual, or technical",
    )
    threads: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of concurrent threads (max_concurrent_api_calls per file)",
    )
    max_parallel_files: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Max number of files to process in parallel when translating a directory",
    )
    max_concurrent_api_calls: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Max concurrent LLM API calls per file (alias for threads)",
    )
    retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    balance_translation_chunks: bool = Field(
        default=True,
        description="Re-split/merge chunks by estimated tokens for more uniform parallel API cost",
    )
    max_chunk_tokens: int = Field(
        default=2400,
        gt=256,
        le=128000,
        description="Target max body tokens per translation request after rebalance (tiktoken estimate)",
    )
    max_output_tokens: int = Field(
        default=8192,
        ge=256,
        le=128000,
        description="Completion max_tokens per translation API call (must cover zh expansion vs body)",
    )
    min_chunk_merge_tokens: int = Field(
        default=400,
        ge=0,
        description="Deprecated: rebalance merge is greedy up to max_chunk_tokens; value ignored (kept for YAML compat)",
    )
    preserve_format: bool = Field(
        default=True,
        description="Whether to preserve original formatting in translation export",
    )
    include_original: bool = Field(
        default=False,
        description="Whether to include original text in translation export",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for translation (null uses provider default)",
    )
    default_prompt_file: Optional[str] = Field(
        default=None,
        description="Default prompt template file for translation (null for style-based)",
    )
    translatable_extensions: List[str] = Field(
        default_factory=lambda: [".md", ".markdown", ".txt", ".ipynb"],
        description="File extensions to include when translating a directory",
    )
    recursive_dir: bool = Field(
        default=False,
        description="When translating a directory, include files from subdirectories",
    )


class BatchConfig(BaseModel):
    """Batch processing default configuration."""

    mode: str = Field(
        default="prompt-content-pairs",
        description="Batch mode: prompt-contents or prompt-content-pairs",
    )
    threads: int = Field(default=5, ge=1, le=50, description="Number of concurrent threads")
    retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0,
        gt=0,
        description="Initial delay between retries in seconds",
    )
    retry_delay_max: float = Field(
        default=10.0,
        gt=0,
        description="Maximum retry delay in seconds",
    )
    default_output_format: str = Field(
        default="json",
        description="Default output format when not specified",
    )
    batch_output_dir: str = Field(
        default="batch_output",
        description="Default output directory for split mode",
    )
    batch_results_dir: str = Field(
        default="batch_results",
        description="Default directory for separate-files mode with multiple models",
    )
    output_suffix: str = Field(
        default="_results",
        description="Suffix for combined output filename",
    )


class FileConfig(BaseModel):
    """File handling configuration."""

    chunk_size: int = Field(
        default=8192,
        gt=0,
        description="Chunk size in bytes for file I/O",
    )
    default_output_suffix: str = Field(
        default="_output",
        description="Default suffix for auto-generated output path",
    )
    translated_suffix: str = Field(
        default="_translated",
        description="Suffix for translated output files",
    )
    formatted_suffix: str = Field(
        default="_formatted",
        description="Suffix for formatted markdown output files",
    )
    tqdm_ncols: int = Field(
        default=80,
        gt=0,
        description="Progress bar width in columns",
    )


class FormatHeadingConfig(BaseModel):
    """Markdown heading formatter configuration."""

    default_prompt_file: str = Field(
        default="@prompts/md-heading-format.md",
        description="Default prompt file path for heading formatting",
    )
    batch_size: int = Field(
        default=80,
        gt=0,
        description="Max headings per LLM API call",
    )
    concurrency: int = Field(
        default=4,
        gt=0,
        description="Max concurrent API calls for heading batches",
    )
    context_heading_count: int = Field(
        default=5,
        ge=0,
        description="Number of previous headings for context in batch processing",
    )


class TextSplitterConfig(BaseModel):
    """Text splitter configuration."""

    max_chunk_size: int = Field(
        default=2000,
        gt=0,
        description="Maximum chunk size in characters",
    )


class TokenConfig(BaseModel):
    """Token counting configuration."""

    default_encoding: str = Field(
        default="cl100k_base",
        description="Default tiktoken encoding for unknown models",
    )


class UnifiedConfig(BaseModel):
    """Unified configuration loaded from default_config.yml."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    file: FileConfig = Field(default_factory=FileConfig)
    format_heading: FormatHeadingConfig = Field(default_factory=FormatHeadingConfig)
    text_splitter: TextSplitterConfig = Field(default_factory=TextSplitterConfig)
    token: TokenConfig = Field(default_factory=TokenConfig)
    project_root_markers: List[str] = Field(
        default_factory=lambda: ["pyproject.toml", "setup.py", ".git", "default_config.yml"],
        description="Markers to detect project root for @ path resolution",
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedConfig":
        """
        Create UnifiedConfig from parsed YAML dictionary.

        Args:
            data: Configuration dictionary from YAML

        Returns:
            UnifiedConfig instance with defaults for missing sections
        """
        trans_raw = data.get("translation") or {}
        if isinstance(trans_raw, dict) and trans_raw.get("max_chunk_size") is not None:
            logger.warning(
                "translation.max_chunk_size is deprecated and ignored; splitting uses "
                "max_chunk_tokens (tiktoken). Remove max_chunk_size from your YAML."
            )
        return cls(
            general=GeneralConfig(**(data.get("general") or {})),
            translation=TranslationConfig(**trans_raw),
            batch=BatchConfig(**(data.get("batch") or {})),
            file=FileConfig(**(data.get("file") or {})),
            format_heading=FormatHeadingConfig(**(data.get("format_heading") or {})),
            text_splitter=TextSplitterConfig(**(data.get("text_splitter") or {})),
            token=TokenConfig(**(data.get("token") or {})),
            project_root_markers=data.get("project_root_markers")
            or [
                "pyproject.toml",
                "setup.py",
                ".git",
                "default_config.yml",
            ],
        )
