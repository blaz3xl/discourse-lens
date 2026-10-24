from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from typer.testing import CliRunner

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.cli import app
from discourselens.errors import ProjectError
from discourselens.mcp_server import create_research_memo, list_research_memos
from discourselens.project import (
    MAX_RESEARCH_MEMO_PAGE_SIZE,
    MAX_RESEARCH_MEMO_TEXT_CHARS,
    MAX_RESEARCH_MEMO_TITLE_CHARS,
)


def _create_memos(root) -> tuple[str, list[str]]:
    project = Project.init(root)
    _, revision = project.import_text("memo-source.txt", "Frozen memo scope.")
    scope_id = project.snapshot([revision])
    memo_ids = [
        project.create_research_memo(
            scope_id,
            "T" * (MAX_RESEARCH_MEMO_TITLE_CHARS + 25),
            "A" * (MAX_RESEARCH_MEMO_TEXT_CHARS + 50),
        ),
        project.create_research_memo(scope_id, "Second memo", "Second memo body."),
        project.create_research_memo(scope_id, "Third memo", "Third memo body."),
    ]
    for index, memo_id in enumerate(memo_ids):
        project.db.execute(
            "UPDATE research_memos SET created_at=? WHERE id=?",
            (f"2026-01-01T00:00:0{index}+00:00", memo_id),
        )
    project.db.commit()
    project.close()
    return scope_id, memo_ids


def test_project_research_memos_page_is_scoped_bounded_and_read_only(tmp_path):
    root = tmp_path / "memo-page-core"
    scope_id, memo_ids = _create_memos(root)
    project = Project.open(root)
    before_changes = project.db.total_changes

    first_page = project.research_memos_page(scope_id, offset=0, limit=2)
    second_page = project.research_memos_page(scope_id, offset=2, limit=2)

    assert first_page["scope_id"] == scope_id
    assert first_page["total_count"] == 3
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    first_rows = cast(list[dict[str, object]], first_page["rows"])
    second_rows = cast(list[dict[str, object]], second_page["rows"])
    assert [row["id"] for row in first_rows] == memo_ids[:2]
    assert [row["id"] for row in second_rows] == memo_ids[2:]
    assert first_rows[0]["title_characters"] == MAX_RESEARCH_MEMO_TITLE_CHARS + 25
    assert first_rows[0]["title_truncated"] is True
    assert first_rows[0]["content_characters"] == MAX_RESEARCH_MEMO_TEXT_CHARS + 50
    assert first_rows[0]["content_truncated"] is True
    assert len(str(first_rows[0]["title"])) == MAX_RESEARCH_MEMO_TITLE_CHARS
    assert len(str(first_rows[0]["content"])) == MAX_RESEARCH_MEMO_TEXT_CHARS
    assert project.db.total_changes == before_changes
    project.close()


def test_project_research_memos_page_rejects_invalid_bounds_and_scope(tmp_path):
    root = tmp_path / "memo-page-invalid"
    scope_id, _ = _create_memos(root)
    project = Project.open(root)
    for offset, limit in [(-1, 1), (True, 1), (0, 0), (0, MAX_RESEARCH_MEMO_PAGE_SIZE + 1)]:
        with pytest.raises(ProjectError, match="research memo page"):
            project.research_memos_page(scope_id, offset=offset, limit=limit)
    with pytest.raises(ProjectError):
        project.research_memos_page("missing-scope", limit=1)
    project.close()


def test_cli_memo_list_keeps_legacy_default_and_adds_bounded_page(tmp_path):
    root = tmp_path / "memo-page-cli"
    scope_id, memo_ids = _create_memos(root)

    runner = CliRunner()
    full = runner.invoke(app, ["coding", "memos", str(root), scope_id])
    paged = runner.invoke(
        app,
        ["coding", "memos", str(root), scope_id, "--offset", "1", "--limit", "1"],
    )

    assert full.exit_code == 0, full.output
    full_rows = json.loads(full.output)
    assert [row["id"] for row in full_rows] == memo_ids
    assert len(full_rows[0]["content"]) == MAX_RESEARCH_MEMO_TEXT_CHARS + 50
    assert paged.exit_code == 0, paged.output
    page = json.loads(paged.output)
    assert page["offset"] == 1
    assert page["limit"] == 1
    assert page["total_count"] == 3
    assert page["has_more"] is True
    rows = cast(list[dict[str, object]], page["rows"])
    assert [row["id"] for row in rows] == memo_ids[1:2]


def test_cli_memo_page_rejects_out_of_range_bounds(tmp_path):
    root = tmp_path / "memo-page-cli-invalid"
    scope_id, _ = _create_memos(root)

    result = CliRunner().invoke(
        app,
        ["coding", "memos", str(root), scope_id, "--limit", "101"],
    )

    assert result.exit_code == 2
    assert "--limit" in result.output


def test_mcp_memo_page_uses_shared_core_contract_in_read_only_mode(tmp_path, monkeypatch):
    root = tmp_path / "memo-page-mcp"
    scope_id, memo_ids = _create_memos(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    page = list_research_memos(scope_id, offset=0, limit=2)

    assert page["scope_id"] == scope_id
    assert page["total_count"] == 3
    assert page["has_more"] is True
    rows = cast(list[dict[str, object]], page["rows"])
    assert [row["id"] for row in rows] == memo_ids[:2]
    tools = asyncio.run(mcp_module.server.list_tools())
    assert any(tool.name == "list_research_memos" for tool in tools)


@pytest.mark.parametrize(("title", "content"), [("", "valid body"), ("valid title", "  ")])
def test_mcp_memo_rejects_invalid_content_without_persisting_scope(
    tmp_path, monkeypatch, title, content
):
    root = tmp_path / "memo-invalid-no-side-effect"
    project = Project.init(root)
    _, revision_id = project.import_text("memo-source.txt", "Frozen memo scope.")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with Project.open(root) as project:
        assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0
        assert project.db.execute("SELECT COUNT(*) FROM research_memos").fetchone()[0] == 0

    with pytest.raises(ProjectError, match="research memo requires a title and content"):
        create_research_memo([revision_id], title, content)

    with Project.open(root) as project:
        assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0
        assert project.db.execute("SELECT COUNT(*) FROM research_memos").fetchone()[0] == 0


@pytest.mark.parametrize(("offset", "limit"), [(-1, 1), (0, 0), (0, 101)])
def test_mcp_memo_page_rejects_out_of_range_bounds(offset, limit):
    with pytest.raises(ToolError):
        list_research_memos("explicit-scope", offset=offset, limit=limit)
