import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens import cli as cli_module
from discourselens.cli import app


def test_cli_collocation_page_uses_core_tie_order(tmp_path):
    root = tmp_path / "collocation-page-order"
    project = Project.init(root)
    _, revision = project.import_text(
        "ties.txt", "target zulu target alpha target beta target alpha"
    )
    scope_id = project.snapshot([revision])
    sdk_page = project.corpus_collocations_page(scope_id, "target", window=1, limit=3)
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "collocations",
            str(root),
            revision,
            "target",
            "--window",
            "1",
            "--offset",
            "0",
            "--limit",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    page = json.loads(result.stdout)
    assert [row["collocate"] for row in page["rows"]] == [
        row["term"] for row in sdk_page["rows"]
    ] == ["alpha", "beta", "zulu"]
    assert [row["count"] for row in page["rows"]] == [3, 2, 2]
    assert page["total"] == sdk_page["total_values"] == 3


def test_cli_collocations_rejects_window_101_before_opening_project(tmp_path, monkeypatch):
    root = tmp_path / "cli-collocation-window-bound"
    project = Project.init(root)
    _, revision = project.import_text("source.txt", "alpha target beta")
    project.close()

    opened_projects = []

    def unexpected_project_open(*args, **kwargs):
        opened_projects.append((args, kwargs))
        raise AssertionError("invalid CLI bounds must be rejected before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_project_open)
    result = CliRunner().invoke(
        app,
        ["corpus", "collocations", str(root), revision, "target", "--window", "101"],
    )

    assert result.exit_code == 2
    assert "1<=x<=100" in result.stderr
    assert "Traceback" not in result.output
    assert opened_projects == []
