import json

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.errors import ScopeViolationError


def test_collocation_analysis_pages_preserve_rank_ties_and_scope(tmp_path):
    project = Project.init(tmp_path / "collocation-analysis-pages")
    _, included = project.import_text("included.txt", "zulu target alpha beta target gamma")
    project.import_text("outside.txt", "target outside")
    scope_id = project.snapshot([included])

    first = project.corpus_collocation_analysis(
        scope_id, "target", window=1, min_frequency=1, offset=0, limit=2
    )
    last = project.corpus_collocation_analysis(
        scope_id, "target", window=1, min_frequency=1, offset=2, limit=2
    )
    beyond = project.corpus_collocation_analysis(
        scope_id, "target", window=1, min_frequency=1, offset=4, limit=2
    )

    assert first["scope_id"] == scope_id
    assert first["offset"] == 0
    assert first["limit"] == 2
    assert first["total"] == 4
    assert first["has_more"] is first["truncated"] is True
    assert [row["term"] for row in first["rows"]] == ["alpha", "beta"]
    assert last["offset"] == 2
    assert last["limit"] == 2
    assert last["total"] == 4
    assert last["has_more"] is last["truncated"] is False
    assert [row["term"] for row in last["rows"]] == ["gamma", "zulu"]
    assert beyond["rows"] == []
    assert beyond["has_more"] is beyond["truncated"] is False
    assert [row["term"] for row in first["rows"] + last["rows"]] == [
        "alpha",
        "beta",
        "gamma",
        "zulu",
    ]
    assert all(row["term"] != "outside" for row in first["rows"] + last["rows"])
    project.close()


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (True, 1), (0, 0), (0, 10_001), (0, True)],
)
def test_collocation_analysis_page_rejects_invalid_bounds(tmp_path, offset, limit):
    project = Project.init(tmp_path / f"collocation-analysis-bounds-{offset}-{limit}")
    _, revision = project.import_text("text.txt", "target alpha")
    scope_id = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="offset|limit"):
        project.corpus_collocation_analysis(
            scope_id, "target", min_frequency=1, offset=offset, limit=limit
        )
    project.close()


def test_cli_collocation_analysis_returns_final_page_metadata(tmp_path):
    root = tmp_path / "cli-collocation-analysis-pages"
    project = Project.init(root)
    _, revision = project.import_text("included.txt", "zulu target alpha beta target gamma")
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "collocation-analysis",
            str(root),
            revision,
            "target",
            "--window",
            "1",
            "--min-frequency",
            "1",
            "--offset",
            "2",
            "--limit",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    page = json.loads(result.stdout)
    assert page["offset"] == 2
    assert page["limit"] == 2
    assert page["total"] == 4
    assert page["has_more"] is page["truncated"] is False
    assert [row["term"] for row in page["rows"]] == ["gamma", "zulu"]
