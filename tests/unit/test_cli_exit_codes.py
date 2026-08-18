"""Regression tests: typer.Exit must not be swallowed by RuntimeError handlers.

typer.Exit subclasses RuntimeError; commands that raise typer.Exit(0) inside a
try block guarded by ``except RuntimeError`` must re-raise it (see
cli/errors.py and paper.py for the known pattern).
"""

from unittest import mock

import pytest
import typer
from typer.testing import CliRunner

from ask_llm.cli.app import app

runner = CliRunner()


def test_cli_errors_maps_keyboard_interrupt_to_exit_1():
    """Ctrl-C must map to exit 1 with a message, not propagate to click's Abort."""
    from ask_llm.cli.errors import cli_errors

    with pytest.raises(typer.Exit) as excinfo, cli_errors("ask"):
        raise KeyboardInterrupt()
    assert excinfo.value.exit_code == 1


def test_api_key_gate_module_does_not_import_typer():
    """Layering regression: utils/api_key_gate stays pure (no typer dependency)."""
    import ast
    from pathlib import Path

    import ask_llm.utils.api_key_gate as gate_mod

    tree = ast.parse(Path(gate_mod.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        imported = []
        if isinstance(node, ast.Import):
            imported = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [node.module.split(".")[0]]
        assert "typer" not in imported, "utils must not import typer"


def _patch_format_bootstrap(monkeypatch):
    from ask_llm.cli.commands import format_cmd

    fake_manager = mock.MagicMock()
    fake_manager.get_provider_config.return_value = {"provider": "deepseek"}

    load_result = mock.MagicMock()
    monkeypatch.setattr(
        format_cmd, "load_cli_session", mock.MagicMock(return_value=(load_result, fake_manager))
    )
    monkeypatch.setattr(
        format_cmd,
        "resolve_and_prepare",
        mock.MagicMock(return_value=("deepseek", "deepseek-chat")),
    )
    monkeypatch.setattr(format_cmd, "gate_api_key_or_exit", mock.MagicMock())
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
    # Bootstrap must resolve through the shared entry and run the API key gate.
    from ask_llm.cli.commands import format_cmd

    format_cmd.resolve_and_prepare.assert_called_once()
    format_cmd.gate_api_key_or_exit.assert_called_once()


def test_format_invalid_type_reports_type_error(monkeypatch, tmp_path):
    """Exit(1) for bad --type must report the type message, not 'API 错误'."""
    _patch_format_bootstrap(monkeypatch)
    doc = tmp_path / "doc.md"
    doc.write_text("# hi\n")

    response = runner.invoke(app, ["format", str(doc), "--type", "bogus"])
    assert response.exit_code == 1, response.output
    assert "不支持的格式化类型" in response.output
    assert "API 错误" not in response.output
