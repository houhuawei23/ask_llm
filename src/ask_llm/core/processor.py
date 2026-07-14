"""Request processing logic."""

import time
from collections.abc import Iterator

from loguru import logger

from ask_llm.config.context import get_config
from ask_llm.core.models import (
    ChatHistory,
    MessageRole,
    ProcessingResult,
    RequestMetadata,
)
from ask_llm.core.protocols import LLMProviderProtocol, ReasoningChunk
from ask_llm.utils.token_counter import TokenCounter


class RequestProcessor:
    """Process LLM API requests."""

    def __init__(
        self,
        provider: LLMProviderProtocol,
        default_prompt_template: str | None = None,
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

    def _format_prompt(self, content: str, prompt_template: str | None = None) -> str:
        """
        Format prompt with content.

        Args:
            content: Input content
            prompt_template: Prompt template with {content} placeholder

        Returns:
            Formatted prompt string
        """
        template = prompt_template or self._get_default_prompt_template()

        # Use replace, not str.format: prompts often contain literal ``{``/``}`` (LaTeX,
        # JSON examples, ``{variable}`` in code samples). Only ``{content}`` is a placeholder.
        if "{content}" in template:
            return template.replace("{content}", content)
        else:
            return f"{template}\n\n{content}"

    def process(
        self,
        content: str,
        prompt_template: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> Iterator[str | ReasoningChunk]:
        """
        Process content with LLM.

        Args:
            content: Input content
            prompt_template: Prompt template with {content} placeholder
            temperature: Sampling temperature
            model: Model name
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream response
            system_prompt: Optional system prompt to prepend as a system message.
                When set, the request uses messages format instead of prompt.

        Yields:
            Response text chunks (if streaming) or full response
        """
        prompt = self._format_prompt(content, prompt_template)
        logger.debug(f"Processing request with {len(prompt)} characters")

        # Prepare kwargs for provider.call()
        call_kwargs: dict = {}
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens

        # Use messages format if system_prompt is provided
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            call_kwargs["messages"] = messages
        else:
            call_kwargs["prompt"] = prompt

        response = self.provider.call(
            temperature=temperature, model=model, stream=stream, **call_kwargs
        )
        if isinstance(response, str):
            yield response
            return
        if isinstance(response, ReasoningChunk):
            yield response
            return
        yield from response

    def iter_process_raw_stream(
        self,
        prompt_template: str,
        *,
        temperature: float | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        return_reasoning: bool = False,
        system_prompt: str | None = None,
    ) -> Iterator[str | ReasoningChunk]:
        """
        Stream a full user prompt (paper / raw mode) and yield chunks.

        When *return_reasoning* is False, yields content string fragments (same as ``process(..., stream=True)`` style).
        When True (e.g. DeepSeek reasoner), yields ReasoningChunk pairs.
        """
        prompt = (prompt_template or "").strip()
        logger.debug(f"Streaming raw prompt request with {len(prompt)} characters")

        call_kw: dict = {
            "temperature": temperature,
            "model": model,
            "stream": True,
        }

        # Use messages format if system_prompt is provided
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            call_kw["messages"] = messages
        else:
            call_kw["prompt"] = prompt

        if max_tokens is not None:
            call_kw["max_tokens"] = max_tokens
        if return_reasoning:
            call_kw["return_reasoning"] = True

        gen = self.provider.call(**call_kw)
        if isinstance(gen, str):
            yield gen
            return
        if isinstance(gen, ReasoningChunk):
            yield gen
            return
        for item in gen:
            if isinstance(item, tuple) and len(item) == 2:
                yield ReasoningChunk(content=item[0], reasoning=item[1])
            else:
                yield item

    def process_with_metadata(
        self,
        content: str,
        prompt_template: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        return_reasoning: bool = False,
        raw_prompt: bool = False,
        system_prompt: str | None = None,
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
            system_prompt: Optional system prompt to prepend as a system message.
                When set, the request uses messages format instead of prompt.

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
            "temperature": temperature,
            "model": model,
            "stream": False,
        }

        # Use messages format if system_prompt is provided, otherwise use prompt
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            call_kw["messages"] = messages
            # Recalculate input tokens to include system prompt
            full_input = f"{system_prompt}\n{prompt}"
            input_stats = TokenCounter.estimate_tokens(full_input, model)
        else:
            call_kw["prompt"] = prompt

        if max_tokens is not None:
            call_kw["max_tokens"] = max_tokens
        if return_reasoning:
            call_kw["return_reasoning"] = True

        raw = self.provider.call(**call_kw)
        reasoning: str | None = None
        if isinstance(raw, ReasoningChunk):
            response, reasoning = raw.content, raw.reasoning
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
        metadata = RequestMetadata.from_execution(
            provider_name=self.provider.name,
            model=resolved_model,
            temperature=temperature,
            default_temperature=self.provider.config.api_temperature,
            input_stats=input_stats,
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
        system_prompt: str | None = None,
        initial_context: str | None = None,
        prompt_template: str | None = None,
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
