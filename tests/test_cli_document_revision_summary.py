import json

from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app


def test_document_show_matches_core_summary_without_sqlite_access(tmp_path, monkeypatch):
    project_root = tmp_path / "revision-summary"
    project = Project.init(project_root)
    _, revision_id = project.import_text(
        "speech.txt",
        "private source text that must not appear in the summary",
    )
    expected = project.document_revision_summary(revision_id)

    class CoreOnlyProject:
        def document_revision_summary(self, requested_revision: str) -> dict[str, object]:
            return project.document_revision_summary(requested_revision)

        @property
        def db(self):
            raise AssertionError("document show must not read SQLite directly")

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli_module.Project, "open", lambda _path: CoreOnlyProject())
    result = CliRunner().invoke(app, ["document", "show", str(project_root), revision_id])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == expected
    assert "private source text" not in result.output
    assert str(project_root) not in result.output
    assert expected["revision_id"] == revision_id
    assert expected["display_name"] == "speech.txt"
    project.close()


def test_document_show_preserves_unknown_revision_error(tmp_path):
    project_root = tmp_path / "unknown-revision"
    project = Project.init(project_root)
    project.close()

    result = CliRunner().invoke(app, ["document", "show", str(project_root), "missing-revision"])

    assert result.exit_code == 2
    assert "unknown revision" in result.output
    assert "Traceback" not in result.output
