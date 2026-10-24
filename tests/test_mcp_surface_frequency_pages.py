from __future__ import annotations

from collections.abc import Mapping

import pytest

import discourselens.mcp_server as mcp_module
from discourselens.errors import ProjectError


class _PageProject:
    def __init__(self, page: Mapping[str, object]) -> None:
        self.page = dict(page)
        self.page_call: tuple[str, list[str], str, int, int] | None = None
        self.closed = False

    def snapshot(self, revision_ids: list[str]) -> str:
        assert revision_ids == ["revision-1"]
        return "scope-1"

    def corpus_surface_frequency_page(
        self,
        scope_id: str,
        artifact_ids: list[str],
        *,
        field: str,
        offset: int,
        limit: int,
    ) -> Mapping[str, object]:
        self.page_call = (scope_id, artifact_ids, field, offset, limit)
        return self.page

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("tool", "field"),
    [
        (mcp_module.corpus_normalized_frequency, "normalized"),
        (mcp_module.corpus_document_frequency, "document_frequency"),
    ],
)
def test_surface_frequency_tools_delegate_to_core_page_api(monkeypatch, tool, field) -> None:
    page = {
        "scope_id": "scope-1",
        "revision_ids": ["revision-1"],
        "artifact_ids": ["artifact-1"],
        "field": field,
        "offset": 1,
        "limit": 1,
        "total_values": 2,
        "has_more": False,
        "rows": [{"token": "café", "count": 3}],
    }
    project = _PageProject(page)
    monkeypatch.setattr(mcp_module, "_open_project", lambda: project)

    result = tool(["revision-1"], ["artifact-1"], offset=1, limit=1)

    assert result == page
    assert project.page_call == (
        "scope-1",
        ["artifact-1"],
        field,
        1,
        1,
    )
    assert project.closed is True


@pytest.mark.parametrize(
    "tool",
    [mcp_module.corpus_normalized_frequency, mcp_module.corpus_document_frequency],
)
def test_surface_frequency_tools_fail_closed_over_json_text_budget(
    monkeypatch, tool
) -> None:
    secret_token = "untrusted-token-" + ("x" * mcp_module.MAX_MCP_FREQUENCY_PAGE_BYTES)
    project = _PageProject(
        {
            "scope_id": "scope-1",
            "revision_ids": ["revision-1"],
            "artifact_ids": ["artifact-1"],
            "field": "normalized",
            "offset": 0,
            "limit": 1,
            "total_values": 1,
            "has_more": False,
            "rows": [{"token": secret_token, "count": 1}],
        }
    )
    monkeypatch.setattr(mcp_module, "_open_project", lambda: project)

    with pytest.raises(ProjectError, match="64 KiB UTF-8 JSON response budget") as exc_info:
        tool(["revision-1"], ["artifact-1"])

    assert secret_token not in str(exc_info.value)
    assert project.closed is True
