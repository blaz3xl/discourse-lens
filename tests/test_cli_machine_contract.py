from decimal import Decimal
from types import SimpleNamespace

from typer.testing import CliRunner

from discourselens import Project
from discourselens import cli as cli_module
from discourselens.cli import app


def test_empty_batch_revision_selection_is_a_stderr_parameter_error(tmp_path):
    project = Project.init(tmp_path / "research")
    project.close()

    result = CliRunner().invoke(
        app,
        ["analyze", "batch-textual-descriptors", str(tmp_path / "research"), ", ,"],
    )

    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)
    assert result.stdout == ""
    assert "revisions must include at least one revision" in result.stderr
    assert "Traceback" not in result.stderr


def test_signaling_decimal_nan_error_keeps_machine_output_on_stderr(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "supported_providers",
        lambda: [
            SimpleNamespace(
                provider_id="invalid-provider",
                transport="fixture",
                endpoint_policy="none",
                generation=Decimal("sNaN"),
                embeddings=False,
                reranking=False,
            )
        ],
    )

    result = CliRunner().invoke(app, ["providers", "list", "--format", "json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "finite numeric values" in result.stderr
    assert "Traceback" not in result.stderr
