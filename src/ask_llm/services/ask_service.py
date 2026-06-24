"""Single-request orchestration service.

Moves the core ``ask`` workflow (load input, prepare prompt, dry-run preview,
non-streaming processing, output path resolution) out of the CLI command so the
command module stays focused on argument parsing, streaming UX, and exit codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ask_llm.config.manager import ConfigManager
from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.models import ProcessingResult, RequestMetadata
from ask_llm.core.processor import RequestProcessor
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.token_counter import TokenCounter


@dataclass
class AskResult:
    """Result of a single ``ask`` processing request."""

    content: str
    output_path: str | None = None
    metadata: RequestMetadata | None = None
    reasoning: str | None = None

    @property
    def output_content(self) -> str:
        """Content formatted for file output (metadata prepended when available)."""
        if self.metadata:
            return self.metadata.format() + self.content
        return self.content


@dataclass
class AskDryRunInfo:
    """Information produced by ``ask --dry-run``."""

    model: str
    estimated_input_tokens: int
    estimated_words: int
    final_prompt: str
    system_prompt_tokens: int | None = None


class AskService:
    """High-level service for processing a single LLM request."""

    def __init__(
        self,
        *,
        config_manager: ConfigManager,
        unified_config: UnifiedConfig,
        model: str,
        processor: RequestProcessor | None = None,
    ) -> None:
        """Initialize the ask service.

        Args:
            config_manager: Active config manager (provider/model already resolved).
            unified_config: Loaded unified configuration.
            model: Resolved model name.
            processor: Optional processor; must be set before calling process methods
                unless only prepare/dry-run helpers are used.
        """
        self.config_manager = config_manager
        self.unified_config = unified_config
        self.model = model
        self.processor = processor

    def set_processor(self, processor: RequestProcessor) -> None:
        """Set the request processor after API key checks are complete."""
        self.processor = processor

    def _ensure_processor(self) -> RequestProcessor:
        if self.processor is None:
            raise RuntimeError("RequestProcessor is not set on AskService")
        return self.processor

    def load_content(self, source: str, *, show_progress: bool = True) -> tuple[str, bool]:
        """Load input content from a file path or use the string directly.

        Args:
            source: File path or direct text input.
            show_progress: Show progress bar when reading a file.

        Returns:
            Tuple of (content, input_is_file).
        """
        input_path = Path(source)
        if input_path.exists() and input_path.is_file():
            return FileHandler.read(source, show_progress=show_progress), True
        return source, False

    def load_prompt_template(self, prompt: str | None) -> str | None:
        """Load prompt template from file or string, ensuring a {content} placeholder.

        Args:
            prompt: Prompt template file path or raw template string.

        Returns:
            Normalized prompt template or None.
        """
        if not prompt:
            return None

        prompt_path = Path(prompt)
        if prompt_path.exists() and prompt_path.is_file():
            template = FileHandler.read(prompt)
        else:
            template = prompt

        if "{content}" not in template:
            template = template + "\n\n{content}"
        return template

    def dry_run(
        self,
        content: str,
        prompt_template: str | None,
        system_prompt: str | None,
    ) -> AskDryRunInfo:
        """Preview prompt and token estimate without making an API call.

        Args:
            content: Input content.
            prompt_template: Prompt template with {content} placeholder.
            system_prompt: Optional system prompt.

        Returns:
            Dry-run information for display.
        """
        template = prompt_template or self.unified_config.general.default_prompt_template
        if "{content}" in template:
            final_prompt = template.replace("{content}", content)
        else:
            final_prompt = f"{template}\n\n{content}"

        stats = TokenCounter.estimate_tokens(final_prompt, self.model)
        system_tokens: int | None = None
        if system_prompt:
            system_tokens = TokenCounter.estimate_tokens(system_prompt, self.model)["token_count"]

        return AskDryRunInfo(
            model=self.model,
            estimated_input_tokens=stats["token_count"],
            estimated_words=stats["word_count"],
            final_prompt=final_prompt,
            system_prompt_tokens=system_tokens,
        )

    def process(
        self,
        content: str,
        *,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        return_reasoning: bool = False,
    ) -> ProcessingResult:
        """Process content non-streamingly and return a result with metadata.

        Args:
            content: Input content.
            prompt_template: Prompt template with {content} placeholder.
            system_prompt: Optional system prompt.
            return_reasoning: Request reasoning content from reasoner models.

        Returns:
            Processing result with metadata.
        """
        processor = self._ensure_processor()
        return processor.process_with_metadata(
            content=content,
            prompt_template=prompt_template,
            model=self.model,
            system_prompt=system_prompt,
            return_reasoning=return_reasoning,
        )

    def process_to_file(
        self,
        content: str,
        *,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        include_metadata: bool = False,
        return_reasoning: bool = False,
    ) -> AskResult:
        """Process content and prepare file-output result.

        Args:
            content: Input content.
            prompt_template: Prompt template.
            system_prompt: Optional system prompt.
            include_metadata: Prepend metadata to output content.
            return_reasoning: Request reasoning content.

        Returns:
            AskResult ready for writing to disk.
        """
        result = self.process(
            content,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            return_reasoning=return_reasoning,
        )
        return AskResult(
            content=result.content,
            metadata=result.metadata,
            reasoning=result.reasoning,
        )

    def determine_output_path(
        self,
        source: str,
        input_is_file: bool,
        output: str | None,
    ) -> str:
        """Determine the final output path for a file-mode request.

        Args:
            source: Original input source.
            input_is_file: Whether the source was a file path.
            output: Explicit -o/--output value, if any.

        Returns:
            Resolved output file path.
        """
        if input_is_file:
            return FileHandler.generate_output_path(source, output)
        return output or self.unified_config.general.default_output_filename

    def write_output(self, output_path: str, content: str, *, force: bool = False) -> None:
        """Write output content to disk.

        Args:
            output_path: Destination file path.
            content: Content to write.
            force: Overwrite existing file.
        """
        FileHandler.write(output_path, content, force=force)
