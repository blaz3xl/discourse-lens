import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_ngrams_opt_in_pagination_preserves_legacy_output(tmp_path):
    project = Project.init(tmp_path / "research")
    _, revision = project.import_text(
        "speech.txt", "alpha beta alpha beta alpha gamma alpha beta"
    )
    project.close()
    runner = CliRunner()
    command = ["corpus", "ngrams", str(tmp_path / "research"), revision, "--size", "2"]

    full = runner.invoke(app, command)
    first = runner.invoke(app, [*command, "--limit", "2"])
    second = runner.invoke(app, [*command, "--offset", "2", "--limit", "2"])

    assert full.exit_code == first.exit_code == second.exit_code == 0
    legacy_rows = json.loads(full.stdout)
    first_page = json.loads(first.stdout)
    second_page = json.loads(second.stdout)
    assert isinstance(legacy_rows, list)
    assert first_page == {
        "offset": 0,
        "limit": 2,
        "total": 4,
        "has_more": True,
        "rows": [{"ngram": "alpha beta", "count": 3}, {"ngram": "beta alpha", "count": 2}],
    }
    assert second_page["has_more"] is False
    assert [row["ngram"] for row in first_page["rows"] + second_page["rows"]] == [
        row[0] for row in legacy_rows
    ]


def test_collocations_opt_in_pagination_preserves_legacy_output(tmp_path):
    project = Project.init(tmp_path / "research")
    _, revision = project.import_text(
        "speech.txt", "red target blue target green target amber"
    )
    project.close()
    runner = CliRunner()
    command = [
        "corpus",
        "collocations",
        str(tmp_path / "research"),
        revision,
        "target",
        "--window",
        "1",
    ]

    full = runner.invoke(app, command)
    first = runner.invoke(app, [*command, "--limit", "2"])
    second = runner.invoke(app, [*command, "--offset", "2", "--limit", "2"])

    assert full.exit_code == first.exit_code == second.exit_code == 0
    legacy_rows = json.loads(full.stdout)
    first_page = json.loads(first.stdout)
    second_page = json.loads(second.stdout)
    assert isinstance(legacy_rows, list)
    assert first_page["offset"] == 0
    assert first_page["limit"] == 2
    assert first_page["total"] == 4
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    assert sorted(
        row["collocate"] for row in first_page["rows"] + second_page["rows"]
    ) == sorted(row[0] for row in legacy_rows)


def test_corpus_pagination_bounds_and_human_status(tmp_path):
    project = Project.init(tmp_path / "research")
    _, revision = project.import_text("speech.txt", "red target blue")
    project.close()
    runner = CliRunner()
    base = ["corpus", "collocations", str(tmp_path / "research"), revision, "target"]

    page = runner.invoke(app, [*base, "--limit", "1", "--format", "table"])
    invalid = runner.invoke(app, [*base, "--limit", "1001"])

    assert page.exit_code == 0
    assert "collocate" in page.stdout and "count" in page.stdout
    assert "has_more=true" in page.stderr
    assert invalid.exit_code == 2
