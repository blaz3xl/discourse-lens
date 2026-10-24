from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError


def _sdk_page_payload(scope_id: str, page: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"scope_id": scope_id}
    payload.update({key: value for key, value in page.items() if key != "rows"})
    payload["rows"] = [
        dict(row) for row in cast(list[Mapping[str, object]], page["rows"])
    ]
    return payload


def test_mcp_exact_search_matches_sdk_across_scope_pages_and_revisions(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "exact-scope-project"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt",
        "Scope Parity occurs in the first revision.\n\n"
        "Scope Parity occurs again in the first revision.",
    )
    _, second_revision = project.import_text(
        "second.txt",
        "Scope Parity occurs in the second revision.\n\n"
        "Scope Parity occurs again in the second revision.",
    )
    _, outside_revision = project.import_text(
        "outside.txt", "Scope Parity is outside this immutable scope."
    )
    scope_id = project.snapshot([first_revision, second_revision])
    expected_revisions = project.scope_revisions(scope_id)
    sdk_first = project.search_exact_page("Scope Parity", scope_id, offset=0, limit=2)
    sdk_next = project.search_exact_page("Scope Parity", scope_id, offset=2, limit=2)
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    monkeypatch.setattr(mcp_module, "_open_project", lambda: Project.open(root, read_only=True))
    mcp_first = mcp_module.search_exact_for_scope(
        scope_id, "Scope Parity", offset=0, limit=2
    )
    mcp_next = mcp_module.search_exact_for_scope(
        scope_id, "Scope Parity", offset=2, limit=2
    )

    assert mcp_first == _sdk_page_payload(scope_id, sdk_first)
    assert mcp_next == _sdk_page_payload(scope_id, sdk_next)
    assert mcp_first["scope_id"] == scope_id
    assert mcp_first["total_count"] == 4
    assert mcp_first["has_more"] is True
    assert mcp_next["offset"] == 2
    assert mcp_next["has_more"] is False
    first_rows = cast(list[dict[str, object]], mcp_first["rows"])
    next_rows = cast(list[dict[str, object]], mcp_next["rows"])
    rows = [*first_rows, *next_rows]
    assert {row["revision_id"] for row in rows} == set(expected_revisions)
    assert outside_revision not in {row["revision_id"] for row in rows}
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("scope_id", "phrase", "offset", "limit"),
    [
        ("", "phrase", 0, 100),
        (" \t", "phrase", 0, 100),
        ("s" * (mcp_module.MCP_EXACT_SEARCH_MAX_SCOPE_ID_CHARS + 1), "phrase", 0, 100),
        ("scope", "", 0, 100),
        ("scope", " \t ", 0, 100),
        ("scope", "x" * 2_001, 0, 100),
        ("scope", "!!!", 0, 100),
        (1, "phrase", 0, 100),
        ("scope", 42, 0, 100),
        ("scope", "phrase", True, 100),
        ("scope", "phrase", -1, 100),
        ("scope", "phrase", mcp_module.MCP_EXACT_SEARCH_MAX_OFFSET + 1, 100),
        ("scope", "phrase", 0, False),
        ("scope", "phrase", 0, 0),
        ("scope", "phrase", 0, mcp_module.MCP_EXACT_SEARCH_MAX_LIMIT + 1),
    ],
)
def test_mcp_exact_search_rejects_invalid_arguments_before_opening_project(
    monkeypatch, scope_id, phrase, offset, limit
) -> None:
    monkeypatch.setattr(
        mcp_module, "_open_project", lambda: pytest.fail("must validate before opening project")
    )

    with pytest.raises(ProjectError):
        mcp_module.search_exact_for_scope(scope_id, phrase, offset=offset, limit=limit)


def test_mcp_exact_search_rejects_unknown_scope_without_snapshotting(tmp_path, monkeypatch) -> None:
    root = tmp_path / "unknown-exact-scope-project"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "A searchable source.")
    project.snapshot([revision_id])
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    monkeypatch.setattr(mcp_module, "_open_project", lambda: Project.open(root, read_only=True))
    with pytest.raises(ScopeViolationError, match="unknown scope"):
        mcp_module.search_exact_for_scope("unknown-scope", "searchable")

    assert database.read_bytes() == before


def test_mcp_exact_search_schema_is_discoverable_and_read_only(monkeypatch) -> None:
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        tools = asyncio.run(mcp_module.server.list_tools())
        assert "search_exact_for_scope" in {tool.name for tool in tools}
        tool = next(item for item in tools if item.name == "search_exact_for_scope")
        schema = tool.model_dump(by_alias=True)["inputSchema"]
        assert schema["required"] == ["scope_id", "phrase"]
        assert schema["properties"]["scope_id"]["minLength"] == 1
        assert (
            schema["properties"]["scope_id"]["maxLength"]
            == mcp_module.MCP_EXACT_SEARCH_MAX_SCOPE_ID_CHARS
        )
        assert schema["properties"]["scope_id"]["pattern"] == r"\S"
        assert schema["properties"]["phrase"]["minLength"] == 1
        assert schema["properties"]["phrase"]["pattern"] == r"\S"
        assert "searchable word token" in schema["properties"]["phrase"]["description"]
        assert (
            schema["properties"]["phrase"]["maxLength"]
            == mcp_module.MCP_EXACT_SEARCH_MAX_PHRASE_CHARS
        )
        assert schema["properties"]["offset"]["minimum"] == 0
        assert (
            schema["properties"]["offset"]["maximum"]
            == mcp_module.MCP_EXACT_SEARCH_MAX_OFFSET
        )
        assert schema["properties"]["limit"]["minimum"] == 1
        assert (
            schema["properties"]["limit"]["maximum"]
            == mcp_module.MCP_EXACT_SEARCH_MAX_LIMIT
        )
        assert "search_exact_for_scope" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
