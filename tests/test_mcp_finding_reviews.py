from __future__ import annotations

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from discourselens import Project, mcp_server
from discourselens.errors import ProjectError


def _project_with_finding(project_root):
    project = Project.init(project_root)
    _, revision_id = project.import_text("source.txt", "A source sentence.")
    run_id = project.create_run(project.snapshot([revision_id]), {"fixture": "review"})
    project.start_run(run_id)
    finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "observed-pattern",
        "A bounded observation for review.",
    )
    project.finalize_run(run_id)
    project.close()
    return finding_id


def test_mcp_record_finding_review_persists_core_decision(tmp_path, monkeypatch):
    root = tmp_path / "review-project"
    finding_id = _project_with_finding(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    tool = next(
        item
        for item in asyncio.run(mcp_server.server.list_tools())
        if item.name == "record_finding_review"
    )
    status_schema = tool.model_dump(by_alias=True)["inputSchema"]["properties"]["status"]
    assert status_schema["enum"] == ["ACCEPTED", "MODIFIED", "REJECTED"]

    result = mcp_server.record_finding_review(
        finding_id, "accepted", "Aprobado tras revisar la evidencia."
    )

    assert result["finding_id"] == finding_id
    assert result["status"] == "ACCEPTED"
    assert result["review_id"]
    with Project.open(root, read_only=True) as project:
        reviews = project.finding_reviews(finding_id)
    assert len(reviews) == 1
    assert reviews[0]["id"] == result["review_id"]
    assert reviews[0]["status"] == "ACCEPTED"
    assert reviews[0]["note"] == "Aprobado tras revisar la evidencia."


def test_mcp_get_finding_reviews_pages_complete_history_in_read_only_mode(tmp_path, monkeypatch):
    root = tmp_path / "review-history-project"
    finding_id = _project_with_finding(root)
    with Project.open(root) as project:
        expected_ids = [
            project.review_finding(finding_id, status, f"review {index}")
            for index, status in enumerate(("ACCEPTED", "MODIFIED", "REJECTED"))
        ]
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", True)

    try:
        asyncio.run(mcp_server._configure_read_only_server())
        names = {tool.name for tool in asyncio.run(mcp_server.server.list_tools())}
        assert "get_finding_reviews" in names
        first_page = mcp_server.get_finding_reviews(finding_id, limit=2)
        second_page = mcp_server.get_finding_reviews(finding_id, offset=2, limit=2)
    finally:
        monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", False)
        mcp_server._restore_full_server_registry()

    first_reviews = first_page["reviews"]
    second_reviews = second_page["reviews"]
    assert isinstance(first_reviews, list)
    assert isinstance(second_reviews, list)
    assert first_page["total_count"] == second_page["total_count"] == 3
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    assert [review["id"] for review in first_reviews + second_reviews] == expected_ids
    assert [review["review_order"] for review in first_reviews + second_reviews] == [1, 2, 3]
    assert database.read_bytes() == before


def test_mcp_record_finding_review_rejects_invalid_status_and_utf8_oversize_before_write(
    tmp_path, monkeypatch
):
    root = tmp_path / "invalid-review-project"
    finding_id = _project_with_finding(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    oversized_note = "💬" * (mcp_server.MAX_MCP_FINDING_REVIEW_NOTE_BYTES // 4 + 1)

    with pytest.raises(ToolError, match="ACCEPTED, MODIFIED, or REJECTED"):
        mcp_server.record_finding_review(finding_id, "APPROVED")
    with pytest.raises(ToolError, match="UTF-8 bytes"):
        mcp_server.record_finding_review(finding_id, "REJECTED", oversized_note)

    with Project.open(root, read_only=True) as project:
        assert project.finding_reviews(finding_id) == []


def test_mcp_record_finding_review_is_hidden_and_blocked_in_read_only_mode(
    tmp_path, monkeypatch
):
    root = tmp_path / "readonly-review-project"
    finding_id = _project_with_finding(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", True)

    try:
        asyncio.run(mcp_server._configure_read_only_server())
        names = {tool.name for tool in asyncio.run(mcp_server.server.list_tools())}
        assert "record_finding_review" not in names
        with pytest.raises(ProjectError, match="read-only mode"):
            mcp_server.record_finding_review(finding_id, "MODIFIED", "Researcher edit.")
    finally:
        monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", False)
        mcp_server._restore_full_server_registry()

    with Project.open(root, read_only=True) as project:
        assert project.finding_reviews(finding_id) == []
