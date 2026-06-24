"""Unit tests for AskService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.models import ProcessingResult, RequestMetadata
from ask_llm.services.ask_service import AskDryRunInfo, AskResult, AskService


@pytest.fixture
def mock_config_manager():
    cm = MagicMock()
    cm.config = MagicMock()
    return cm


@pytest.fixture
def mock_unified_config():
    uc = MagicMock()
    uc.general.default_prompt_template = "Please process the following text:\n\n{content}"
    uc.general.default_output_filename = "output.txt"
    return uc


@pytest.fixture
def mock_processor():
    return MagicMock()


@pytest.fixture
def service(mock_config_manager, mock_unified_config, mock_processor):
    return AskService(
        config_manager=mock_config_manager,
        unified_config=mock_unified_config,
        model="gpt-4",
        processor=mock_processor,
    )


def test_load_content_from_file(service, tmp_path):
    file_path = tmp_path / "input.txt"
    file_path.write_text("hello world", encoding="utf-8")

    with patch("ask_llm.services.ask_service.FileHandler") as mock_fh:
        mock_fh.read.return_value = "hello world"
        content, is_file = service.load_content(str(file_path))

    assert content == "hello world"
    assert is_file is True
    mock_fh.read.assert_called_once_with(str(file_path), show_progress=True)


def test_load_content_from_string(service):
    content, is_file = service.load_content("direct text")
    assert content == "direct text"
    assert is_file is False


def test_load_prompt_template_from_file(service, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Translate: {content}", encoding="utf-8")

    with patch("ask_llm.services.ask_service.FileHandler") as mock_fh:
        mock_fh.read.return_value = "Translate: {content}"
        template = service.load_prompt_template(str(prompt_file))

    assert template == "Translate: {content}"


def test_load_prompt_template_adds_content_placeholder(service):
    template = service.load_prompt_template("Summarize")
    assert template == "Summarize\n\n{content}"


def test_load_prompt_template_returns_none_when_empty(service):
    assert service.load_prompt_template(None) is None
    assert service.load_prompt_template("") is None


def test_dry_run_replaces_content(service):
    with patch("ask_llm.services.ask_service.TokenCounter") as mock_tc:
        mock_tc.estimate_tokens.return_value = {
            "token_count": 42,
            "word_count": 10,
        }
        info = service.dry_run("hello", "Translate: {content}", None)

    assert isinstance(info, AskDryRunInfo)
    assert info.model == "gpt-4"
    assert info.estimated_input_tokens == 42
    assert info.estimated_words == 10
    assert info.final_prompt == "Translate: hello"
    assert info.system_prompt_tokens is None


def test_dry_run_counts_system_prompt(service):
    with patch("ask_llm.services.ask_service.TokenCounter") as mock_tc:
        mock_tc.estimate_tokens.side_effect = [
            {"token_count": 10, "word_count": 2},
            {"token_count": 5, "word_count": 1},
        ]
        info = service.dry_run("hello", None, "You are helpful")

    assert info.system_prompt_tokens == 5


def test_process_to_file_uses_processor(service, mock_processor):
    metadata = RequestMetadata(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        input_words=1,
        input_tokens=1,
        output_words=1,
        output_tokens=1,
        latency=0.1,
    )
    mock_processor.process_with_metadata.return_value = ProcessingResult(
        content="result",
        metadata=metadata,
        reasoning="thinking",
    )

    result = service.process_to_file(
        "hello",
        prompt_template="Translate: {content}",
        include_metadata=True,
        return_reasoning=True,
    )

    assert isinstance(result, AskResult)
    assert result.content == "result"
    assert result.metadata == metadata
    assert result.reasoning == "thinking"
    assert metadata.format() in result.output_content
    mock_processor.process_with_metadata.assert_called_once_with(
        content="hello",
        prompt_template="Translate: {content}",
        model="gpt-4",
        system_prompt=None,
        return_reasoning=True,
    )


def test_determine_output_path_for_input_file(service):
    with patch("ask_llm.services.ask_service.FileHandler") as mock_fh:
        mock_fh.generate_output_path.return_value = "input_output.txt"
        path = service.determine_output_path("input.txt", True, None)
    assert path == "input_output.txt"


def test_determine_output_path_uses_explicit_output(service):
    path = service.determine_output_path("input.txt", True, "custom.txt")
    assert path == "custom.txt"


def test_determine_output_path_for_text_uses_default(service):
    path = service.determine_output_path("hello", False, None)
    assert path == "output.txt"


def test_write_output_delegates_to_file_handler(service):
    with patch("ask_llm.services.ask_service.FileHandler") as mock_fh:
        service.write_output("out.txt", "content", force=True)
        mock_fh.write.assert_called_once_with("out.txt", "content", force=True)


def test_service_requires_processor_for_process(service):
    service_no_proc = AskService(
        config_manager=service.config_manager,
        unified_config=service.unified_config,
        model="gpt-4",
    )
    with pytest.raises(RuntimeError, match="RequestProcessor is not set"):
        service_no_proc.process("hello")


def test_set_processor(service):
    new_processor = MagicMock()
    service.set_processor(new_processor)
    assert service.processor is new_processor
