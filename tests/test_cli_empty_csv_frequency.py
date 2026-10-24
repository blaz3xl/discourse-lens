from __future__ import annotations

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_empty_surface_frequency_csv_keeps_column_header(tmp_path):
    project_root = tmp_path / "empty-frequency"
    project = Project.init(project_root)
    _, revision = project.import_text("empty.txt", "   \n\t")
    project.close()

    result = CliRunner().invoke(
        app,
        ["corpus", "frequency", str(project_root), revision, "--format", "csv"],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "token,count\n"
    assert "no frequency rows" in result.stderr
    assert "has_more=false" in result.stderr
