"""Regression tests: typer.Exit must not be swallowed by RuntimeError handlers.

typer.Exit subclasses RuntimeError; commands that raise typer.Exit(0) inside a
try block guarded by ``except RuntimeError`` must re-raise it (see
cli/errors.py and paper.py for the known pattern).
"""

from unittest import mock

import pytest
from typer.testing import CliRunner

from ask_llm.cli.app import app

runner = CliRunner()


def _patch_format_bootstrap(monkeypatch):
    from ask_llm.cli.commands import format_cmd

    fake_manager = mock.MagicMock()
    fake_manager.get_provider_config.return_value = {"provider": "deepseek"}
    fake_manager.get_model_override.return_value = "deepseek-chat"
    fake_manager.get_default_model.return_value = "deepseek-chat"

    load_result = mock.MagicMock()
    monkeypatch.setattr(
        format_cmd, "load_cli_session", mock.MagicMock(return_value=(load_result, fake_manager))
    )
    monkeypatch.setattr(format_cmd, "apply_cli_overrides_and_gate_api_key", mock.MagicMock())
    monkeypatch.setattr(format_cmd, "create_engine_adapter", mock.MagicMock())
    monkeypatch.setattr(format_cmd, "RequestProcessor", mock.MagicMock())

    resume = mock.MagicMock()
    fake_service = mock.MagicMock()
    fake_service.resume_from_checkpoint = resume
    monkeypatch.setattr(format_cmd, "FormatService", mock.MagicMock(return_value=fake_service))
    return resume


def test_format_resume_success_exits_zero(monkeypatch, tmp_path):
    """`format --resume` success path must exit 0, not be caught as API error."""
    resume = _patch_format_bootstrap(monkeypatch)
    doc = tmp_path / "doc.md"
    doc.write_text("# hi\n")
    ckpt = tmp_path / "doc.md.body_checkpoint.json"
    ckpt.write_text("{}")

    response = runner.invoke(app, ["format", str(doc), "--resume", str(ckpt)])
    assert response.exit_code == 0, response.output
    assert "API 错误" not in response.output
    resume.assert_called_once()
    # Bootstrap must go through the shared helper (includes the API key gate).
    from ask_llm.cli.commands import format_cmd

    format_cmd.apply_cli_overrides_and_gate_api_key.assert_called_once()


def test_format_invalid_type_reports_type_error(monkeypatch, tmp_path):
    """Exit(1) for bad --type must report the type message, not 'API 错误'."""
    _patch_format_bootstrap(monkeypatch)
    doc = tmp_path / "doc.md"
    doc.write_text("# hi\n")

    response = runner.invoke(app, ["format", str(doc), "--type", "bogus"])
    assert response.exit_code == 1, response.output
    assert "不支持的格式化类型" in response.output
    assert "API 错误" not in response.output
