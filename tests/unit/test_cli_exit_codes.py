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


# ---------------------------------------------------------------------------
# H4: cli_errors must not swallow usage errors / OSError
# ---------------------------------------------------------------------------


def test_cli_errors_lets_usage_errors_through():
    """H4: BadParameter (click UsageError family) keeps click's usage handling
    and exit code 2 instead of becoming 'Unexpected error' + exit 1."""
    import click

    from ask_llm.cli.errors import cli_errors

    with pytest.raises(click.exceptions.UsageError), cli_errors("format"):
        raise typer.BadParameter("多个输入文件不能使用单个 Markdown 文件作为输出")


def test_cli_errors_maps_oserror_to_readable_exit_1():
    """H4: PermissionError/OSError get a readable message + exit 1, not a
    generic 'Unexpected error'."""
    from ask_llm.cli.errors import cli_errors

    with pytest.raises(typer.Exit) as excinfo, cli_errors("ask"):
        raise PermissionError("output file is not writable")
    assert excinfo.value.exit_code == 1


# ---------------------------------------------------------------------------
# H2: format must exit non-zero when any file fails
# ---------------------------------------------------------------------------


def _patch_format_run(monkeypatch):
    from ask_llm.cli.commands import format_cmd

    fake_manager = mock.MagicMock()
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
    return format_cmd


def test_format_failed_files_exit_nonzero(monkeypatch, tmp_path):
    """H2: failed_count > 0 must map to exit 1 (was: always exit 0)."""
    from ask_llm.services.format_service import FormatRunStats

    format_cmd = _patch_format_run(monkeypatch)
    doc = tmp_path / "doc.md"
    doc.write_text("# hi\n", encoding="utf-8")
    monkeypatch.setattr(
        format_cmd,
        "run_format",
        mock.MagicMock(return_value=FormatRunStats(successful_count=0, failed_count=1)),
    )

    response = runner.invoke(app, ["format", str(doc), "--type", "title"])
    assert response.exit_code == 1, response.output


def test_format_all_success_exits_zero(monkeypatch, tmp_path):
    from ask_llm.services.format_service import FormatRunStats

    format_cmd = _patch_format_run(monkeypatch)
    doc = tmp_path / "doc.md"
    doc.write_text("# hi\n", encoding="utf-8")
    monkeypatch.setattr(
        format_cmd,
        "run_format",
        mock.MagicMock(return_value=FormatRunStats(successful_count=1, failed_count=0)),
    )

    response = runner.invoke(app, ["format", str(doc), "--type", "title"])
    assert response.exit_code == 0, response.output


# ---------------------------------------------------------------------------
# H3: config test / show consistency
# ---------------------------------------------------------------------------


def _write_test_config(tmp_path):
    import yaml

    config = {
        "default_provider": "prov_a",
        "default_model": "wrong-global-model",
        "providers": {
            "prov_a": {
                "base_url": "https://a.example.com/v1",
                "api_key": "key-a",
                "default_model": "a-model",
                "models": [{"name": "a-model"}],
            },
        },
    }
    config_path = tmp_path / "default_config.yml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return config_path


def test_config_test_uses_provider_model_first(monkeypatch, tmp_path):
    """H3: with both a global default_model and provider models present, the
    provider's own model wins (matches runtime resolution)."""
    from ask_llm.cli.commands import config as config_cmd

    config_path = _write_test_config(tmp_path)
    used = {}

    class _FakeAdapter:
        def __init__(self, pc, default_model=None):
            used["model"] = default_model

        def test_connection(self):
            return True, "ok", 0.1

    monkeypatch.setattr(config_cmd, "create_engine_adapter", _FakeAdapter)

    response = runner.invoke(app, ["config", "test", "--config", str(config_path)])
    assert response.exit_code == 0, response.output
    assert used["model"] == "a-model"


def test_config_test_connection_failure_exits_nonzero(monkeypatch, tmp_path):
    from ask_llm.cli.commands import config as config_cmd

    config_path = _write_test_config(tmp_path)

    class _FailingAdapter:
        def __init__(self, pc, default_model=None):
            pass

        def test_connection(self):
            return False, "connection refused", 0.0

    monkeypatch.setattr(config_cmd, "create_engine_adapter", _FailingAdapter)

    response = runner.invoke(app, ["config", "test", "--config", str(config_path)])
    assert response.exit_code == 1, response.output


def test_config_test_unknown_provider_exits_nonzero(tmp_path):
    config_path = _write_test_config(tmp_path)
    response = runner.invoke(
        app, ["config", "test", "--provider", "nope", "--config", str(config_path)]
    )
    assert response.exit_code == 1, response.output
    assert "not found" in response.output


def test_config_show_reports_unresolved_key_as_unconfigured(monkeypatch, tmp_path):
    """H3: `config show` must not claim an unresolved ${VAR} is configured."""
    import yaml

    config = {
        "default_provider": "prov_a",
        "providers": {
            "prov_a": {
                "base_url": "https://a.example.com/v1",
                "api_key": "${DEFINITELY_UNSET_VAR_12345}",
                "models": [{"name": "a-model"}],
            },
        },
    }
    config_path = tmp_path / "default_config.yml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    monkeypatch.delenv("DEFINITELY_UNSET_VAR_12345", raising=False)

    response = runner.invoke(app, ["config", "show", "--config", str(config_path)])
    assert response.exit_code == 0, response.output
    assert "✓ Configured" not in response.output
    assert "✗ Not configured" in response.output


# ---------------------------------------------------------------------------
# H3/2.25: ask input validation must render as a clean CLI error
# ---------------------------------------------------------------------------


def test_ask_missing_input_file_is_clean_cli_error():
    """H3: a path-looking missing input must exit 1 with a readable message,
    never escape cli_errors as a raw FileNotFoundError traceback."""
    response = runner.invoke(app, ["ask", "definitely-missing-input-12345.md"])
    assert response.exit_code == 1, response.output
    assert "Input file not found" in response.output
    assert not isinstance(response.exception, FileNotFoundError)


def test_config_init_yes_overwrites_without_prompt(tmp_path):
    """L12/2.25: `config init --yes` overwrites existing files without
    prompting — non-interactive scripts must not die on typer's Abort."""
    target = tmp_path / "default_config.yml"
    target.write_text("existing: true", encoding="utf-8")

    ok = runner.invoke(app, ["config", "init", "-o", str(target), "--yes"])
    assert ok.exit_code == 0, ok.output
    assert "existing: true" not in target.read_text(encoding="utf-8")


def test_config_init_existing_target_prompts_and_aborts_cleanly(tmp_path):
    """Without --yes an existing target still prompts (aborting in
    non-interactive contexts) — behavior preserved for interactive use."""
    target = tmp_path / "default_config.yml"
    target.write_text("existing: true", encoding="utf-8")

    aborted = runner.invoke(app, ["config", "init", "-o", str(target)], input="n\n")
    assert aborted.exit_code == 0  # user declined -> clean exit, no overwrite
    assert target.read_text(encoding="utf-8") == "existing: true"


def test_config_get_distinguishes_missing_key_from_none(tmp_path):
    """M17/2.25: a typo'd key errors; a nullable key that exists prints None
    instead of a misleading 'Key not found'."""
    import yaml

    config = {
        "default_provider": "prov_a",
        "providers": {
            "prov_a": {
                "base_url": "https://a.example.com/v1",
                "api_key": "sk-test",
                "models": [{"name": "a-model"}],
            },
        },
    }
    config_path = tmp_path / "default_config.yml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    missing = runner.invoke(
        app, ["config", "get", "translation.nonexistent_key_xyz", "--config", str(config_path)]
    )
    assert missing.exit_code == 1
    assert "Key not found" in missing.output

    nullable = runner.invoke(
        app, ["config", "get", "translation.default_prompt_file", "--config", str(config_path)]
    )
    assert nullable.exit_code == 0, nullable.output
