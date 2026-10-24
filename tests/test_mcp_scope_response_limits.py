from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from discourselens import Project, mcp_server
from discourselens.errors import ProjectError, ScopeViolationError


class _LargeScopeProject:
    def __init__(self) -> None:
        self.scopes = {
            "scope-left": [f"left-revision-{index}" for index in range(150)],
            "scope-right": [f"right-revision-{index}" for index in range(131)],
        }

    def corpus_snapshot(self, corpus_id: str) -> str:
        assert corpus_id == "corpus"
        return "scope-left"

    def subcorpus_snapshot(self, subcorpus_id: str) -> str:
        assert subcorpus_id == "subcorpus"
        return "scope-right"

    def snapshot_by_document_metadata(self, filters: dict[str, object]) -> tuple[str, list[str]]:
        assert filters == {"language": "en"}
        return "scope-left", self.scopes["scope-left"]

    def scope_revisions(self, scope_id: str) -> list[str]:
        return self.scopes[scope_id]

    def scope_revisions_page(self, scope_id: str, *, offset: int, limit: int) -> dict[str, object]:
        revisions = self.scope_revisions(scope_id)
        page = revisions[offset : offset + limit]
        return {
            "scope_id": scope_id,
            "offset": offset,
            "limit": limit,
            "revision_count": len(revisions),
            "revision_ids": page,
            "has_more": offset + len(page) < len(revisions),
        }

    def corpus_compare_frequency_page(
        self, left_scope_id: str, right_scope_id: str, *, offset: int, limit: int
    ) -> dict[str, object]:
        return self._comparison_page("frequency", offset, limit)

    def corpus_keyness_page(
        self, left_scope_id: str, right_scope_id: str, *, offset: int, limit: int
    ) -> dict[str, object]:
        return self._comparison_page("keyness", offset, limit)

    @staticmethod
    def _comparison_page(metric: str, offset: int, limit: int) -> dict[str, object]:
        return {
            "metric": metric,
            "offset": offset,
            "limit": limit,
            "total_rows": 1,
            "has_more": False,
            "rows": [],
        }

    def close(self) -> None:
        pass


def test_mcp_scope_responses_bound_revision_id_lists(monkeypatch) -> None:
    project = _LargeScopeProject()
    monkeypatch.setattr(mcp_server, "_open_project", lambda: project)
    monkeypatch.setattr(
        mcp_server, "_frequency_v1", lambda _project, _scope_id: Counter({"token": 1})
    )

    results: list[dict[str, Any]] = [
        mcp_server.snapshot_corpus("corpus"),
        mcp_server.snapshot_subcorpus("subcorpus"),
        mcp_server.select_documents_by_metadata({"language": "en"}),
    ]
    for result in results:
        assert "revision_ids" not in result
        assert result["revision_count"] in {131, 150}
        assert len(result["revision_ids_preview"]) == mcp_server.MCP_SCOPE_REVISION_PREVIEW_LIMIT
        assert result["revision_ids_truncated"] is True

    comparisons = (
        mcp_server.corpus_compare_frequency("scope-left", "scope-right", limit=1),
        mcp_server.corpus_keyness("scope-left", "scope-right", limit=1),
    )
    for result in comparisons:
        populations = result["populations"]
        assert isinstance(populations, dict)
        for population in populations.values():
            assert isinstance(population, dict)
            assert "revision_ids" not in population
            assert len(population["revision_ids_preview"]) == (
                mcp_server.MCP_SCOPE_REVISION_PREVIEW_LIMIT
            )
            assert population["revision_ids_truncated"] is True


def test_mcp_scope_revision_pages_cover_complete_snapshot_in_order(monkeypatch) -> None:
    project = _LargeScopeProject()
    monkeypatch.setattr(mcp_server, "_open_project", lambda: project)

    first = mcp_server.get_scope_revisions("scope-left", limit=73)
    second = mcp_server.get_scope_revisions("scope-left", offset=73, limit=73)
    final = mcp_server.get_scope_revisions("scope-left", offset=146, limit=73)

    first_ids = first["revision_ids"]
    second_ids = second["revision_ids"]
    final_ids = final["revision_ids"]
    assert first_ids == project.scopes["scope-left"][:73]
    assert second_ids == project.scopes["scope-left"][73:146]
    assert final_ids == project.scopes["scope-left"][146:]
    assert first["revision_count"] == second["revision_count"] == final["revision_count"] == 150
    assert first["has_more"] is True
    assert second["has_more"] is True
    assert final["has_more"] is False
    assert first["scope_id"] == second["scope_id"] == final["scope_id"] == "scope-left"


@pytest.mark.parametrize(
    ("scope_id", "offset", "limit"),
    [
        ("scope-left", -1, 10),
        ("scope-left", 0, 0),
        ("scope-left", 0, mcp_server.MCP_COLLECTION_MAX_PAGE_SIZE + 1),
        ("scope-left", True, 10),
        ("", 0, 10),
    ],
)
def test_mcp_scope_revision_page_rejects_invalid_bounds(
    monkeypatch, scope_id: str, offset: int, limit: int
) -> None:
    monkeypatch.setattr(mcp_server, "_open_project", lambda: pytest.fail("must validate first"))
    with pytest.raises(ProjectError, match="scope revisions page"):
        mcp_server.get_scope_revisions(scope_id, offset=offset, limit=limit)


def test_mcp_scope_revision_page_rejects_unknown_scope_read_only(tmp_path, monkeypatch) -> None:
    root = tmp_path / "unknown-scope-project"
    with Project.init(root) as project:
        _, revision_id = project.import_text("source.txt", "A source.")
        project.snapshot([revision_id])

    monkeypatch.setattr(
        mcp_server,
        "_open_project",
        lambda: Project.open(root, read_only=True),
    )
    with pytest.raises(ScopeViolationError, match="unknown scope"):
        mcp_server.get_scope_revisions("unknown-scope")
