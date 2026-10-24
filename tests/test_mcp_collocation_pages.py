from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import corpus_collocations


def test_mcp_collocation_pages_match_sdk_and_remain_revision_scoped(tmp_path, monkeypatch):
    root = tmp_path / "mcp-collocation-pages"
    project = Project.init(root)
    _, revision = project.import_text(
        "included.txt", "target zulu target alpha target beta target alpha"
    )
    project.import_text("outside.txt", "target outsider target")
    scope_id = project.snapshot([revision])
    sdk_first = project.corpus_collocations_page(scope_id, "target", window=1, limit=2)
    sdk_next = project.corpus_collocations_page(
        scope_id, "target", window=1, offset=2, limit=1
    )
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_first = corpus_collocations(revision, "target", window=1, limit=2)
    mcp_next = corpus_collocations(revision, "target", window=1, offset=2, limit=1)

    assert {key: value for key, value in mcp_first.items() if key != "truncated"} == {
        "revision_id": revision,
        **{key: value for key, value in sdk_first.items() if key != "scope_id"},
    }
    assert {key: value for key, value in mcp_next.items() if key != "truncated"} == {
        "revision_id": revision,
        **{key: value for key, value in sdk_next.items() if key != "scope_id"},
    }
    assert mcp_first["truncated"] is mcp_first["has_more"] is True
    assert mcp_next["truncated"] is mcp_next["has_more"] is False
    rows = cast(list[dict[str, object]], mcp_first["rows"])
    assert all(str(row["term"]) != "outsider" for row in rows)
    assert mcp_first["total_values"] == 3


@pytest.mark.parametrize(
    ("offset", "limit"), [(-1, 1), (0, 0), (0, 1_001)]
)
def test_mcp_collocation_pages_reject_invalid_bounds(
    tmp_path, monkeypatch, offset, limit
):
    root = tmp_path / f"mcp-collocation-invalid-{offset}-{limit}"
    project = Project.init(root)
    _, revision = project.import_text("text.txt", "target alpha")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError):
        corpus_collocations(revision, "target", offset=offset, limit=limit)


@pytest.mark.parametrize("query,window", [(" ", 1), ("target", 0), ("target", 101)])
def test_mcp_collocation_pages_keep_existing_query_and_window_bounds(
    tmp_path, monkeypatch, query, window
):
    root = tmp_path / f"mcp-collocation-query-{window}"
    project = Project.init(root)
    _, revision = project.import_text("text.txt", "target alpha")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError):
        corpus_collocations(revision, query, window=window)
