"""Translation core logic and models."""

from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional

from loguru import logger

from ask_llm.config.context import get_config
from ask_llm.core.batch import BatchTask, ModelConfig
from ask_llm.core.text_splitter import TextChunk
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.token_counter import TokenCounter


class TranslationStyle(str, Enum):
    """Translation style options."""

    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"


class Translator:
    """Translation processor that generates prompts and handles translation."""

    # Base prompt templates for different styles
    PROMPT_TEMPLATES: ClassVar[dict[TranslationStyle, tuple[str, str]]] = {
        TranslationStyle.FORMAL: (
            "请将以下{source_lang}文本翻译成{target_lang}，使用正式、专业的语言风格，"  # noqa: RUF001
            "保持原文的格式和结构：\n\n{content}"  # noqa: RUF001
        ),
        TranslationStyle.CASUAL: (
            "请将以下{source_lang}文本翻译成{target_lang}，使用自然、口语化的语言风格，"  # noqa: RUF001
            "保持原文的格式和结构：\n\n{content}"  # noqa: RUF001
        ),
        TranslationStyle.TECHNICAL: (
            "请将以下{source_lang}文本翻译成{target_lang}，使用技术性、准确的语言风格，"  # noqa: RUF001
            "保持专业术语的准确性，保持原文的格式和结构：\n\n{content}"  # noqa: RUF001
        ),
    }

    DEFAULT_TEMPLATE: ClassVar[
        str
    ] = "请将以下{source_lang}文本翻译成{target_lang}，保持格式和风格：\n\n{content}"  # noqa: RUF001

    def __init__(
        self,
        target_language: str = "zh",
        source_language: str = "auto",
        style: str = TranslationStyle.FORMAL,
        custom_prompt_template: Optional[str] = None,
        prompt_file: Optional[str] = None,
        glossary_pairs: Optional[list[tuple[str, str]]] = None,
    ):
        """
        Initialize translator.

        Args:
            target_language: Target language code (e.g., 'zh', 'en')
            source_language: Source language code ('auto' for auto-detection)
            style: Translation style (formal/casual/technical)
            custom_prompt_template: Custom prompt template (overrides style)
            prompt_file: Path to prompt template file (overrides custom_prompt_template and style)
            glossary_pairs: Optional list of (source, target) terminology pairs
        """
        self.target_language = target_language
        self.source_language = source_language
        self.style = style
        self.custom_prompt_template = custom_prompt_template
        self.prompt_file = prompt_file
        self.glossary_pairs = glossary_pairs or []

        # Load prompt from file if specified
        if prompt_file:
            self.custom_prompt_template = self._load_prompt_from_file(prompt_file)

    def prompt_template_for_batch(self) -> str:
        """
        Template with ``{source_lang}`` / ``{target_lang}`` substituted; ``{content}`` unchanged.

        Used by :meth:`create_translation_tasks` so :class:`~ask_llm.core.processor.RequestProcessor`
        can merge ``task.content`` via ``{content}`` once. Storing a fully expanded prompt here
        would cause :meth:`RequestProcessor._format_prompt` to append the body again.
        """
        if self.custom_prompt_template:
            template = self.custom_prompt_template
        else:
            template = self.PROMPT_TEMPLATES.get(self.style, self.DEFAULT_TEMPLATE)

        source_lang = self._format_language_name(self.source_language)
        target_lang = self._format_language_name(self.target_language)

        template = template.replace("{source_lang}", source_lang).replace("{target_lang}", target_lang)

        # Inject glossary if provided
        if self.glossary_pairs:
            lines = ["\nGlossary (use these translations consistently):"]
            for src, tgt in self.glossary_pairs:
                lines.append(f"- {src} → {tgt}")
            lines.append("")
            glossary_block = "\n".join(lines) + "\n"
            if "{content}" in template:
                template = template.replace("{content}", glossary_block + "{content}")
            else:
                template = template + glossary_block

        return template

    def generate_prompt(self, content: str) -> str:
        """
        Generate translation prompt for given content.

        Args:
            content: Content to translate

        Returns:
            Formatted prompt string
        """
        return self.prompt_template_for_batch().replace("{content}", content)

    def count_prompt_template_tokens(self, model: str) -> int:
        """
        Tiktoken count of the instruction-only part: template with language placeholders
        filled and ``{content}`` removed (same substitution rules as ``generate_prompt``).
        """
        static = self.prompt_template_for_batch().replace("{content}", "")
        return TokenCounter.count_tokens(static, model)

    def create_translation_tasks(
        self, chunks: list[TextChunk], model_config: ModelConfig
    ) -> list[BatchTask]:
        """
        Create batch tasks from text chunks.

        Args:
            chunks: List of text chunks to translate
            model_config: Model configuration

        Returns:
            List of batch tasks
        """
        tasks = []
        prompt_template = self.prompt_template_for_batch()
        for chunk in chunks:
            task = BatchTask(
                task_id=chunk.chunk_id,
                prompt=prompt_template,
                content=chunk.content,
                task_model_config=model_config,
            )
            tasks.append(task)

        logger.debug(f"Created {len(tasks)} translation tasks from {len(chunks)} chunks")
        return tasks

    @staticmethod
    def _load_prompt_from_file(prompt_path: str) -> str:
        """
        Load prompt template from file.

        Supports @ prefix for relative paths from project root.

        Args:
            prompt_path: Path to prompt file (may start with @)

        Returns:
            Prompt template content

        Raises:
            FileNotFoundError: If prompt file not found
            OSError: If file cannot be read
        """
        # Handle @ prefix (relative to project root)
        if prompt_path.startswith("@"):
            # Remove @ prefix
            relative_path = prompt_path[1:]
            # Try to find project root by looking for common markers
            current_dir = Path.cwd()
            project_root = None
            markers = get_config().unified_config.project_root_markers
            for marker in markers:
                for parent in [current_dir, *list(current_dir.parents)]:
                    if (parent / marker).exists():
                        project_root = parent
                        break
                if project_root:
                    break

            if project_root:
                prompt_file = project_root / relative_path.lstrip("/")
            else:
                # Fallback to current directory
                prompt_file = Path(relative_path.lstrip("/"))
        else:
            prompt_file = Path(prompt_path)

        # Resolve absolute path
        if not prompt_file.is_absolute():
            prompt_file = prompt_file.resolve()

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        logger.debug(f"Loading prompt template from: {prompt_file}")
        try:
            content = FileHandler.read(str(prompt_file))
            return content.strip()
        except Exception as e:
            raise OSError(f"Failed to read prompt file {prompt_file}: {e}") from e

    @staticmethod
    def load_glossary(path: str) -> list[tuple[str, str]]:
        """
        Load glossary file with terminology pairs.

        Supports:
        - YAML (.yml, .yaml): flat dict {src: tgt} or list [{src: ..., tgt: ...}]
        - JSONL (.jsonl): lines of {"src": ..., "tgt": ...} or {"source": ..., "target": ...}

        Args:
            path: Path to glossary file

        Returns:
            List of (source, target) tuples

        Raises:
            FileNotFoundError: If file not found
            ValueError: If format is unrecognized
        """
        import json

        import yaml

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Glossary file not found: {path}")

        suffix = file_path.suffix.lower()
        content = file_path.read_text(encoding="utf-8")

        if suffix in (".yml", ".yaml"):
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                return [(str(k), str(v)) for k, v in data.items()]
            elif isinstance(data, list):
                pairs = []
                for item in data:
                    if isinstance(item, dict):
                        src = item.get("src") or item.get("source")
                        tgt = item.get("tgt") or item.get("target")
                        if src and tgt:
                            pairs.append((str(src), str(tgt)))
                return pairs
            else:
                raise ValueError(f"Unrecognized YAML format in glossary: {path}")

        elif suffix == ".jsonl":
            pairs = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    src = item.get("src") or item.get("source")
                    tgt = item.get("tgt") or item.get("target")
                    if src and tgt:
                        pairs.append((str(src), str(tgt)))
            return pairs

        else:
            raise ValueError(f"Unsupported glossary format: {suffix}. Use .yml, .yaml, or .jsonl")

    @staticmethod
    def _format_language_name(lang_code: str) -> str:
        """
        Format language code to readable name.

        Args:
            lang_code: Language code (e.g., 'zh', 'en', 'auto')

        Returns:
            Formatted language name
        """
        lang_map = {
            "zh": "中文",
            "en": "英文",
            "ja": "日文",
            "ko": "韩文",
            "fr": "法文",
            "de": "德文",
            "es": "西班牙文",
            "ru": "俄文",
            "auto": "原文",
        }
        return lang_map.get(lang_code.lower(), lang_code)
