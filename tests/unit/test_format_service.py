"""Unit tests for FormatService resume helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ask_llm.services.format_service import FormatService


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.unified_config.file.formatted_suffix = "_formatted"
    return cfg


@pytest.fixture
def service():
    processor = MagicMock()
    return FormatService(processor=processor, model="gpt-4")


def test_resume_body_checkpoint_success_removes_checkpoint(service, mock_config, tmp_path):
    checkpoint_path = tmp_path / "doc.md.body_checkpoint.json"
    checkpoint_path.write_text("{}", encoding="utf-8")

    mock_checkpoint = MagicMock()
    mock_checkpoint.source_file = str(tmp_path / "doc.md")
    mock_checkpoint.format_type = "body"
    mock_checkpoint.failed_chunks = []
    mock_checkpoint.successful_chunks = [MagicMock()]

    result = MagicMock()
    result.text = "formatted"
    result.failed_chunks = []
    result.checkpoint_path = None

    with (
        patch("ask_llm.services.format_service.get_config_or_none") as mock_get_config,
        patch("ask_llm.services.format_service.FormatCheckpoint") as mock_cls,
        patch("ask_llm.services.format_service.BodyFormatter") as mock_bf,
        patch("ask_llm.services.format_service.FileHandler") as mock_fh,
        patch("ask_llm.services.format_service.os.remove") as mock_remove,
    ):
        mock_get_config.return_value = mock_config
        mock_cls.load.return_value = mock_checkpoint
        mock_bf.resume_from_checkpoint.return_value = result
        service.resume_from_checkpoint(
            str(checkpoint_path),
            output=None,
            inplace=False,
            force=False,
        )

    mock_fh.write.assert_called_once()
    mock_remove.assert_called_once_with(str(checkpoint_path))


def test_resume_body_checkpoint_remove_failure_warns(service, mock_config, tmp_path):
    """B11: a failed checkpoint removal must warn, not pass silently."""
    checkpoint_path = tmp_path / "doc.md.body_checkpoint.json"
    checkpoint_path.write_text("{}", encoding="utf-8")

    mock_checkpoint = MagicMock()
    mock_checkpoint.source_file = str(tmp_path / "doc.md")
    mock_checkpoint.format_type = "body"
    mock_checkpoint.failed_chunks = []
    mock_checkpoint.successful_chunks = [MagicMock()]

    result = MagicMock()
    result.text = "formatted"
    result.failed_chunks = []
    result.checkpoint_path = None

    with (
        patch("ask_llm.services.format_service.get_config_or_none") as mock_get_config,
        patch("ask_llm.services.format_service.FormatCheckpoint") as mock_cls,
        patch("ask_llm.services.format_service.BodyFormatter") as mock_bf,
        patch("ask_llm.services.format_service.FileHandler") as mock_fh,
        patch("ask_llm.services.format_service.os.remove") as mock_remove,
        patch("ask_llm.services.format_service.console") as mock_console,
    ):
        mock_get_config.return_value = mock_config
        mock_cls.load.return_value = mock_checkpoint
        mock_bf.resume_from_checkpoint.return_value = result
        mock_remove.side_effect = OSError("permission denied")

        service.resume_from_checkpoint(
            str(checkpoint_path),
            output=None,
            inplace=False,
            force=False,
        )

    # File still written; checkpoint removal failed but surfaced as a warning.
    mock_fh.write.assert_called_once()
    mock_remove.assert_called_once_with(str(checkpoint_path))
    mock_console.print_warning.assert_called_once()
    warned = " ".join(str(a) for a in mock_console.print_warning.call_args.args)
    assert str(checkpoint_path) in warned


def test_resume_title_checkpoint_supported(service, mock_config, tmp_path):
    """P3.5: title checkpoints resume (was ValueError before P3.3/P3.5)."""
    checkpoint_path = tmp_path / "doc.md.title_checkpoint.json"
    checkpoint_path.write_text("{}", encoding="utf-8")

    mock_checkpoint = MagicMock()
    mock_checkpoint.source_file = str(tmp_path / "doc.md")
    mock_checkpoint.format_type = "title"
    mock_checkpoint.failed_chunks = []
    mock_checkpoint.successful_chunks = []

    result = MagicMock()
    result.formatted_headings = ["# A", "## B"]
    result.failed_batches = []
    result.checkpoint_path = None

    with (
        patch("ask_llm.services.format_service.get_config_or_none") as mock_get_config,
        patch("ask_llm.services.format_service.FormatCheckpoint") as mock_cls,
        patch("ask_llm.services.format_service.HeadingFormatter") as mock_hf,
        patch("ask_llm.services.format_service.HeadingExtractor") as mock_hex,
        patch("ask_llm.services.format_service.HeadingApplier") as mock_happ,
        patch("ask_llm.services.format_service.FileHandler") as mock_fh,
        patch("ask_llm.services.format_service.os.remove") as mock_remove,
    ):
        mock_get_config.return_value = mock_config
        mock_cls.load.return_value = mock_checkpoint
        mock_hf.resume_from_checkpoint.return_value = result
        mock_hex.extract.return_value = [MagicMock(), MagicMock()]
        mock_happ.return_value.apply.return_value = "merged text"
        mock_fh.read.return_value = "# A\n\n## B\n"
        service.resume_from_checkpoint(
            str(checkpoint_path),
            output=None,
            inplace=False,
            force=False,
        )

    mock_fh.write.assert_called_once()
    mock_remove.assert_called_once_with(str(checkpoint_path))


def test_resume_body_partial_failure_keeps_checkpoint(service, mock_config, tmp_path):
    checkpoint_path = tmp_path / "doc.md.body_checkpoint.json"
    checkpoint_path.write_text("{}", encoding="utf-8")

    mock_checkpoint = MagicMock()
    mock_checkpoint.source_file = str(tmp_path / "doc.md")
    mock_checkpoint.format_type = "body"
    mock_checkpoint.failed_chunks = [MagicMock()]
    mock_checkpoint.successful_chunks = [MagicMock()]

    result = MagicMock()
    result.text = "partial"
    result.failed_chunks = [MagicMock()]
    result.checkpoint_path = str(checkpoint_path)

    with (
        patch("ask_llm.services.format_service.get_config_or_none") as mock_get_config,
        patch("ask_llm.services.format_service.FormatCheckpoint") as mock_cls,
        patch("ask_llm.services.format_service.BodyFormatter") as mock_bf,
        patch("ask_llm.services.format_service.FileHandler") as mock_fh,
        patch("ask_llm.services.format_service.os.remove") as mock_remove,
    ):
        mock_get_config.return_value = mock_config
        mock_cls.load.return_value = mock_checkpoint
        mock_bf.resume_from_checkpoint.return_value = result
        service.resume_from_checkpoint(
            str(checkpoint_path),
            output=None,
            inplace=False,
            force=False,
        )

    mock_fh.write.assert_called_once()
    mock_remove.assert_not_called()
