"""Request processing logic."""

import time
from typing import Generator, Optional

from loguru import logger

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

    DEFAULT_PROMPT_TEMPLATE = "Please process the following text:\n\n{content}"

    def __init__(self, provider: LLMProviderProtocol):
        """
        Initialize processor with provider.

        Args:
            provider: LLM provider instance
        """
        self.provider = provider

    def _format_prompt(self, content: str, prompt_template: Optional[str] = None) -> str:
        """
        Format prompt with content.

        Args:
            content: Input content
            prompt_template: Prompt template with {content} placeholder

        Returns:
            Formatted prompt string
        """
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

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
        stream: bool = False,
    ) -> Generator[str, None, None]:
        """
        Process content with LLM.

        Args:
            content: Input content
            prompt_template: Prompt template with {content} placeholder
            temperature: Sampling temperature
            model: Model name
            stream: Whether to stream response

        Yields:
            Response text chunks (if streaming) or full response
        """
        prompt = self._format_prompt(content, prompt_template)
        logger.debug(f"Processing request with {len(prompt)} characters")

        if stream:
            yield from self.provider.call(
                prompt=prompt, temperature=temperature, model=model, stream=True
            )
        else:
            response = self.provider.call(
                prompt=prompt, temperature=temperature, model=model, stream=False
            )
            yield response

    def process_with_metadata(
        self,
        content: str,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process content and return result with metadata.

        Args:
            content: Input content
            prompt_template: Prompt template
            temperature: Sampling temperature
            model: Model name

        Returns:
            Processing result with metadata
        """
        prompt = self._format_prompt(content, prompt_template)

        # Count input tokens
        input_stats = TokenCounter.estimate_tokens(prompt, model)

        # Call API
        start_time = time.time()
        response = self.provider.call(
            prompt=prompt, temperature=temperature, model=model, stream=False
        )
        latency = time.time() - start_time

        # Count output tokens
        output_stats = TokenCounter.estimate_tokens(response, model)

        # Create metadata
        metadata = RequestMetadata(
            provider=self.provider.name,
            model=model or self.provider.default_model,
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

        return ProcessingResult(content=response, metadata=metadata)

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
