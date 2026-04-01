"""Request processing logic."""

import time
from typing import Generator, Iterator, Optional, Tuple, Union

from loguru import logger

from ask_llm.config.context import get_config
from ask_llm.core.models import (
    ChatHistory,
    MessageRole,
    ProcessingResult,
    RequestMetadata,
)
from ask_llm.core.protocols import LLMProviderProtocol
from ask_llm.utils.token_counter import TokenCounter


class RequestProcessor:
    """Process LLM API requests."""

    def __init__(
        self,
        provider: LLMProviderProtocol,
        default_prompt_template: Optional[str] = None,
    ):
        """
        Initialize processor with provider.

        Args:
            provider: LLM provider instance
            default_prompt_template: Default prompt template when none provided.
                If None, uses value from default_config.yml
        """
        self.provider = provider
        self._default_prompt_template = default_prompt_template

    def _get_default_prompt_template(self) -> str:
        """Get default prompt template from config."""
        if self._default_prompt_template is not None:
            return self._default_prompt_template
        return get_config().unified_config.general.default_prompt_template

    def _format_prompt(self, content: str, prompt_template: Optional[str] = None) -> str:
        """
        Format prompt with content.

        Args:
            content: Input content
            prompt_template: Prompt template with {content} placeholder

        Returns:
            Formatted prompt string
        """
        template = prompt_template or self._get_default_prompt_template()

        if "{content}" in template:
            return template.format(content=content)
        else:
            return f"{template}\n\n{content}"

    def process(
        self,
        content: str,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Generator[str, None, None]:
        """
        Process content with LLM.

        Args:
            content: Input content
            prompt_template: Prompt template with {content} placeholder
            temperature: Sampling temperature
            model: Model name
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream response

        Yields:
            Response text chunks (if streaming) or full response
        """
        prompt = self._format_prompt(content, prompt_template)
        logger.debug(f"Processing request with {len(prompt)} characters")

        # Prepare kwargs for provider.call()
        call_kwargs = {}
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens

        if stream:
            yield from self.provider.call(
                prompt=prompt, temperature=temperature, model=model, stream=True, **call_kwargs
            )
        else:
            response = self.provider.call(
                prompt=prompt, temperature=temperature, model=model, stream=False, **call_kwargs
            )
            yield response

    def iter_process_raw_stream(
        self,
        prompt_template: str,
        *,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        return_reasoning: bool = False,
    ) -> Iterator[Union[str, Tuple[str, str]]]:
        """
        Stream a full user prompt (paper / raw mode) and yield chunks.

        When *return_reasoning* is False, yields content string fragments (same as ``process(..., stream=True)`` style).
        When True (e.g. DeepSeek reasoner), yields ``(content_delta, reasoning_delta)`` pairs.
        """
        prompt = (prompt_template or "").strip()
        logger.debug(f"Streaming raw prompt request with {len(prompt)} characters")

        call_kw: dict = {
            "prompt": prompt,
            "temperature": temperature,
            "model": model,
            "stream": True,
        }
        if max_tokens is not None:
            call_kw["max_tokens"] = max_tokens
        if return_reasoning:
            call_kw["return_reasoning"] = True

        gen = self.provider.call(**call_kw)
        yield from gen

    def process_with_metadata(
        self,
        content: str,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        return_reasoning: bool = False,
        raw_prompt: bool = False,
    ) -> ProcessingResult:
        """
        Process content and return result with metadata.

        Args:
            content: Input content
            prompt_template: Prompt template
            temperature: Sampling temperature
            model: Model name
            max_tokens: Completion token cap (omit to use provider default)
            return_reasoning: If True, request ``reasoning_content`` (e.g. DeepSeek reasoner)
            raw_prompt: If True, ``prompt_template`` is sent as the full user message (no
                ``{content}`` merge); ``content`` is ignored.

        Returns:
            Processing result with metadata
        """
        if raw_prompt:
            prompt = (prompt_template or "").strip()
        else:
            prompt = self._format_prompt(content, prompt_template)

        # Count input tokens
        input_stats = TokenCounter.estimate_tokens(prompt, model)

        # Call API
        start_time = time.time()
        call_kw: dict = {
            "prompt": prompt,
            "temperature": temperature,
            "model": model,
            "stream": False,
        }
        if max_tokens is not None:
            call_kw["max_tokens"] = max_tokens
        if return_reasoning:
            call_kw["return_reasoning"] = True

        raw = self.provider.call(**call_kw)
        reasoning: Optional[str] = None
        if isinstance(raw, tuple) and len(raw) == 2:
            response, reasoning = raw[0], raw[1]
        else:
            response = raw if isinstance(raw, str) else str(raw)

        latency = time.time() - start_time

        # Count output tokens (main answer + optional reasoning)
        out_for_count = response
        if reasoning:
            out_for_count = f"{reasoning}\n{response}"
        output_stats = TokenCounter.estimate_tokens(out_for_count, model)

        resolved_model = model or self.provider.default_model

        # Create metadata
        metadata = RequestMetadata(
            provider=self.provider.name,
            model=resolved_model,
            temperature=temperature
            if temperature is not None
            else self.provider.config.api_temperature,
            input_words=input_stats["word_count"],
            input_tokens=input_stats["token_count"],
            output_words=output_stats["word_count"],
            output_tokens=output_stats["token_count"],
            latency=latency,
        )

        logger.info(
            f"Request completed: {metadata.input_tokens} -> {metadata.output_tokens} "
            f"tokens in {latency:.2f}s"
        )

        return ProcessingResult(content=response, metadata=metadata, reasoning=reasoning)

    def create_chat_history(
        self,
        system_prompt: Optional[str] = None,
        initial_context: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> ChatHistory:
        """
        Create chat history with optional system prompt and initial context.

        Args:
            system_prompt: System prompt message
            initial_context: Initial user context
            prompt_template: Template for formatting initial context

        Returns:
            Chat history
        """
        history = ChatHistory(provider=self.provider.name, model=self.provider.default_model)

        if system_prompt:
            history.add_message(MessageRole.SYSTEM, system_prompt)

        if initial_context:
            content = self._format_prompt(initial_context, prompt_template)
            history.add_message(MessageRole.USER, content)

        return history
