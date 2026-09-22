"""Unit tests for FormatService resume helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.format_checkpoint import CHECKPOINT_VERSION
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
    mock_checkpoint.version = CHECKPOINT_VERSION
    mock_checkpoint.config_digest = ""
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
    mock_checkpoint.version = CHECKPOINT_VERSION
    mock_checkpoint.config_digest = ""
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
    mock_checkpoint.version = CHECKPOINT_VERSION
    mock_checkpoint.config_digest = ""
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
    mock_checkpoint.version = CHECKPOINT_VERSION
    mock_checkpoint.config_digest = ""
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


# ---------------------------------------------------------------------------
# Audit 2.3/2.6/2.7: checkpoint contract (digest), inplace backup, exit outcome
# ---------------------------------------------------------------------------


def _write_source(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nbody\n", encoding="utf-8")
    return src


def _digest_for(src):
    from ask_llm.core.format_checkpoint import compute_format_digest

    return compute_format_digest(
        src,
        prompt_template="PROMPT",
        model="gpt-4",
        max_chunk_tokens=None,
        format_type="body",
    )


def _make_body_result(text="formatted", failed=0, checkpoint_path=None):
    result = MagicMock()
    result.text = text
    result.failed_chunks = [MagicMock() for _ in range(failed)]
    result.checkpoint_path = checkpoint_path
    return result


def test_format_checkpoint_digest_mismatch_refuses_resume(service, tmp_path):
    """M8: resume must refuse when the source changed after the checkpoint."""
    src = _write_source(tmp_path)
    cp_path = tmp_path / "doc.md.body_checkpoint.json"
    cp_path.write_text("{}", encoding="utf-8")

    mock_checkpoint = MagicMock()
    mock_checkpoint.version = CHECKPOINT_VERSION
    mock_checkpoint.source_file = str(src)
    mock_checkpoint.format_type = "body"
    mock_checkpoint.prompt_template = "PROMPT"
    mock_checkpoint.model = "gpt-4"
    mock_checkpoint.max_chunk_tokens = None
    mock_checkpoint.config_digest = "stale-digest"
    mock_checkpoint.failed_chunks = []
    mock_checkpoint.successful_chunks = []

    with patch("ask_llm.services.format_service.FormatCheckpoint") as mock_cls:
        mock_cls.load.return_value = mock_checkpoint
        with pytest.raises(RuntimeError, match="不一致"):
            service.resume_from_checkpoint(str(cp_path), output=None, inplace=True, force=True)


def test_format_checkpoint_legacy_version_refused(service, tmp_path):
    """v≤3 checkpoints lack the digest and are refused outright."""
    cp_path = tmp_path / "doc.md.body_checkpoint.json"
    cp_path.write_text("{}", encoding="utf-8")

    mock_checkpoint = MagicMock()
    mock_checkpoint.version = 3

    with patch("ask_llm.services.format_service.FormatCheckpoint") as mock_cls:
        mock_cls.load.return_value = mock_checkpoint
        with pytest.raises(RuntimeError, match="版本过旧"):
            service.resume_from_checkpoint(str(cp_path), output=None, inplace=False, force=False)


def test_inplace_resume_with_failed_chunks_writes_backup(service, mock_config, tmp_path):
    """2.6: partial inplace resume keeps a one-shot .bak of the source."""
    src = _write_source(tmp_path)
    cp_path = tmp_path / "doc.md.body_checkpoint.json"
    cp_path.write_text("{}", encoding="utf-8")

    mock_checkpoint = MagicMock()
    mock_checkpoint.version = CHECKPOINT_VERSION
    mock_checkpoint.source_file = str(src)
    mock_checkpoint.format_type = "body"
    mock_checkpoint.prompt_template = "PROMPT"
    mock_checkpoint.model = "gpt-4"
    mock_checkpoint.max_chunk_tokens = None
    mock_checkpoint.config_digest = _digest_for(src)
    mock_checkpoint.failed_chunks = [MagicMock()]
    mock_checkpoint.successful_chunks = [MagicMock()]

    with (
        patch("ask_llm.services.format_service.get_config_or_none") as mock_get_config,
        patch("ask_llm.services.format_service.FormatCheckpoint") as mock_cls,
        patch("ask_llm.services.format_service.BodyFormatter") as mock_bf,
        patch("ask_llm.services.format_service.FileHandler") as mock_fh,
    ):
        mock_get_config.return_value = mock_config
        mock_cls.load.return_value = mock_checkpoint
        mock_bf.resume_from_checkpoint.return_value = _make_body_result(
            text="partial", failed=1, checkpoint_path=str(cp_path)
        )
        outcome = service.resume_from_checkpoint(
            str(cp_path), output=None, inplace=True, force=True
        )

    backup = tmp_path / "doc.md.bak"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
    assert outcome.still_failed_count == 1
    assert not outcome.ok
    mock_fh.write.assert_called_once()


def test_resume_clean_success_returns_ok_outcome(service, mock_config, tmp_path):
    """2.7: a clean resume reports ok so the CLI can exit 0."""
    src = _write_source(tmp_path)
    cp_path = tmp_path / "doc.md.body_checkpoint.json"
    cp_path.write_text("{}", encoding="utf-8")

    mock_checkpoint = MagicMock()
    mock_checkpoint.version = CHECKPOINT_VERSION
    mock_checkpoint.source_file = str(src)
    mock_checkpoint.format_type = "body"
    mock_checkpoint.prompt_template = "PROMPT"
    mock_checkpoint.model = "gpt-4"
    mock_checkpoint.max_chunk_tokens = None
    mock_checkpoint.config_digest = _digest_for(src)
    mock_checkpoint.failed_chunks = []
    mock_checkpoint.successful_chunks = [MagicMock()]

    with (
        patch("ask_llm.services.format_service.get_config_or_none") as mock_get_config,
        patch("ask_llm.services.format_service.FormatCheckpoint") as mock_cls,
        patch("ask_llm.services.format_service.BodyFormatter") as mock_bf,
        patch("ask_llm.services.format_service.FileHandler"),
        patch("ask_llm.services.format_service.os.remove") as mock_remove,
    ):
        mock_get_config.return_value = mock_config
        mock_cls.load.return_value = mock_checkpoint
        mock_bf.resume_from_checkpoint.return_value = _make_body_result()
        outcome = service.resume_from_checkpoint(
            str(cp_path), output=None, inplace=False, force=False
        )

    assert outcome.ok
    mock_remove.assert_called_once_with(str(cp_path))
