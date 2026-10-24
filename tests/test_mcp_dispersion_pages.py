from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import corpus_dispersion


def test_mcp_dispersion_is_bounded_by_default_and_matches_sdk_pages(tmp_path, monkeypatch):
    root = tmp_path / "mcp-dispersion-pages"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "migration policy")
    _, second_revision = project.import_text("second.txt", "public migration")
    _, outside_revision = project.import_text("outside.txt", "migration outside")
    scope_id = project.snapshot([first_revision, second_revision])
    sdk_default = project.corpus_dispersion_page(scope_id, "migration")
    sdk_page = project.corpus_dispersion_page(scope_id, "migration", offset=1, limit=1)
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    default = corpus_dispersion([first_revision, second_revision], "migration")
    page = corpus_dispersion([first_revision, second_revision], "migration", offset=1, limit=1)

    assert {key: value for key, value in default.items() if key != "scope_id"} == {
        key: value for key, value in sdk_default.items() if key != "scope_id"
    }
    assert {key: value for key, value in page.items() if key != "scope_id"} == {
        key: value for key, value in sdk_page.items() if key != "scope_id"
    }
    assert default["document_count"] == 2
    assert default["total_rows"] == 2
    default_documents = cast(list[dict[str, object]], default["documents"])
    page_documents = cast(list[dict[str, object]], page["documents"])
    assert len(default_documents) == 2
    assert page["total_occurrences"] == 2
    assert page["has_more"] is False
    assert outside_revision not in {str(row["revision_id"]) for row in page_documents}


@pytest.mark.parametrize("offset,limit", [(-1, 1), (0, 0), (0, 1_001)])
def test_mcp_dispersion_rejects_invalid_page_bounds(tmp_path, monkeypatch, offset, limit):
    root = tmp_path / f"mcp-dispersion-invalid-{offset}-{limit}"
    project = Project.init(root)
    _, revision = project.import_text("text.txt", "migration policy")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError):
        corpus_dispersion([revision], "migration", offset=offset, limit=limit)
