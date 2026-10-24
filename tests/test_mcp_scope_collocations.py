from __future__ import annotations

import asyncio

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError


def test_mcp_scope_collocations_matches_sdk_pages_and_excludes_later_imports(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-collocations"
    project = Project.init(root)
    _, revision_a = project.import_text("a.txt", "focus red focus blue focus red")
    _, revision_b = project.import_text("b.txt", "focus green focus blue")
    scope_id = project.snapshot([revision_a, revision_b])
    expected_members = project.scope_revisions(scope_id)
    sdk_pages = [
        project.corpus_collocations_page(scope_id, "focus", window=1, offset=0, limit=1),
        project.corpus_collocations_page(scope_id, "focus", window=1, offset=1, limit=1),
        project.corpus_collocations_page(scope_id, "focus", window=1, offset=2, limit=2),
    ]
    snapshot_count = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    _, later_revision = project.import_text("later.txt", "focus postonly")
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    def forbid_snapshot(*_args, **_kwargs):
        pytest.fail("scope collocations must not create a ScopeSnapshot")

    monkeypatch.setattr(Project, "snapshot", forbid_snapshot)
    mcp_pages = [
        mcp_module.corpus_collocations_for_scope(
            scope_id, "focus", window=1, offset=0, limit=1
        ),
        mcp_module.corpus_collocations_for_scope(
            scope_id, "focus", window=1, offset=1, limit=1
        ),
        mcp_module.corpus_collocations_for_scope(
            scope_id, "focus", window=1, offset=2, limit=2
        ),
    ]

    assert mcp_pages == sdk_pages
    assert [page["rows"][0]["term"] for page in mcp_pages] == ["blue", "red", "green"]
    assert [page["has_more"] for page in mcp_pages] == [True, True, False]
    assert all(page["scope_id"] == scope_id for page in mcp_pages)
    assert later_revision not in expected_members
    assert all(row["term"] != "postonly" for page in mcp_pages for row in page["rows"])
    assert database.read_bytes() == before
    with Project.open(root, read_only=True) as reopened:
        assert reopened.scope_revisions(scope_id) == expected_members
        assert reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            snapshot_count
        )


@pytest.mark.parametrize(
    ("scope_id", "query", "window", "offset", "limit"),
    [
        ("", "focus", 5, 0, 100),
        ("  ", "focus", 5, 0, 100),
        ("s" * 129, "focus", 5, 0, 100),
        ("scope", "", 5, 0, 100),
        ("scope", "  ", 5, 0, 100),
        ("scope", "focus other", 5, 0, 100),
        ("scope", "!!!", 5, 0, 100),
        ("scope", "x" * 2_001, 5, 0, 100),
        ("scope", "focus", 0, 0, 100),
        ("scope", "focus", 101, 0, 100),
        ("scope", "focus", True, 0, 100),
        ("scope", "focus", 5, -1, 100),
        ("scope", "focus", 5, True, 100),
        ("scope", "focus", 5, 0x7FFF_FFFF_FFFF_FFFF + 1, 100),
        ("scope", "focus", 5, 0, 0),
        ("scope", "focus", 5, 0, 1_001),
        ("scope", "focus", 5, 0, False),
    ],
)
def test_mcp_scope_collocations_rejects_invalid_arguments_before_open(
    monkeypatch, scope_id, query, window, offset, limit
) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("invalid scoped collocation arguments must be rejected first"),
    )
    with pytest.raises(ProjectError):
        mcp_module.corpus_collocations_for_scope(
            scope_id,
            query,
            window=window,
            offset=offset,
            limit=limit,
        )


def test_mcp_scope_collocations_rejects_unknown_scope_read_only(tmp_path, monkeypatch) -> None:
    root = tmp_path / "unknown-scope-collocations"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "focus word")
    project.snapshot([revision_id])
    project.close()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: Project.open(root, read_only=True))

    with pytest.raises(ScopeViolationError, match="scope"):
        mcp_module.corpus_collocations_for_scope("unknown-scope", "focus")


def test_mcp_scope_collocations_rejects_page_over_64kib_without_truncating_term(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "oversized-scope-collocations"
    project = Project.init(root)
    long_collocate = "x" * (65 * 1024)
    _, revision_id = project.import_text("large-token.txt", f"focus {long_collocate}")
    scope_id = project.snapshot([revision_id])
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    with pytest.raises(
        ProjectError,
        match="collocations page exceeds the 64 KiB UTF-8 JSON response budget",
    ):
        mcp_module.corpus_collocations_for_scope(
            scope_id,
            "focus",
            window=1,
            offset=0,
            limit=10,
        )

    assert database.read_bytes() == before


def test_mcp_scope_collocations_preflights_term_characters_before_json_serialization(
    monkeypatch,
) -> None:
    def unexpected_json_serialization(*_args, **_kwargs):
        pytest.fail("raw oversized collocation terms must be rejected before serialization")

    monkeypatch.setattr(mcp_module.json, "dumps", unexpected_json_serialization)
    page = {
        "scope_id": "scope",
        "query": "focus",
        "window": 1,
        "offset": 0,
        "limit": 2,
        "rows": [
            {"term": "a" * 33_000, "count": 1},
            {"term": "b" * 33_000, "count": 1},
        ],
        "total_values": 2,
        "has_more": False,
    }

    with pytest.raises(
        ProjectError,
        match="collocations page exceeds the 64 KiB UTF-8 JSON response budget",
    ):
        mcp_module._enforce_mcp_collocations_page_budget(page)


def test_mcp_scope_collocations_is_discoverable_with_read_only_schema(monkeypatch) -> None:
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        tools = asyncio.run(mcp_module.server.list_tools())
        assert "corpus_collocations_for_scope" in {tool.name for tool in tools}
        tool = next(item for item in tools if item.name == "corpus_collocations_for_scope")
        schema = tool.model_dump(by_alias=True)["inputSchema"]
        assert schema["required"] == ["scope_id", "query"]
        assert schema["properties"]["scope_id"]["minLength"] == 1
        assert schema["properties"]["scope_id"]["maxLength"] == 128
        assert schema["properties"]["query"]["minLength"] == 1
        assert schema["properties"]["query"]["maxLength"] == mcp_module.MCP_KWIC_MAX_QUERY_CHARS
        assert schema["properties"]["window"]["minimum"] == 1
        assert schema["properties"]["window"]["maximum"] == (
            mcp_module.MCP_SCOPE_CORPUS_MAX_WINDOW
        )
        assert schema["properties"]["offset"]["minimum"] == 0
        assert schema["properties"]["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
        assert schema["properties"]["limit"]["minimum"] == 1
        assert schema["properties"]["limit"]["maximum"] == (
            mcp_module.MCP_SCOPE_CORPUS_MAX_PAGE_SIZE
        )
        assert "corpus_collocations_for_scope" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
