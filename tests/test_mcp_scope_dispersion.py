from __future__ import annotations

import asyncio

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError


def test_mcp_scope_dispersion_matches_sdk_pages_and_preserves_frozen_membership(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-dispersion"
    project = Project.init(root)
    _, revision_a = project.import_text("a.txt", "policy policy alpha")
    _, revision_b = project.import_text("b.txt", "policy beta")
    _, revision_c = project.import_text("c.txt", "gamma policy gamma")
    scope_id = project.snapshot([revision_a, revision_b, revision_c])
    expected_members = project.scope_revisions(scope_id)
    sdk_full = project.corpus_dispersion_page(scope_id, "policy")
    sdk_pages = [
        project.corpus_dispersion_page(scope_id, "policy", offset=0, limit=1),
        project.corpus_dispersion_page(scope_id, "policy", offset=1, limit=1),
        project.corpus_dispersion_page(scope_id, "policy", offset=2, limit=1),
    ]
    snapshot_count = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    _, later_revision = project.import_text("later.txt", "policy lateronly")
    project.close()

    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    def forbid_snapshot(*_args, **_kwargs):
        pytest.fail("scope dispersion must not create a ScopeSnapshot")

    monkeypatch.setattr(Project, "snapshot", forbid_snapshot)
    mcp_pages = [
        mcp_module.corpus_dispersion_for_scope(scope_id, "policy", offset=0, limit=1),
        mcp_module.corpus_dispersion_for_scope(scope_id, "policy", offset=1, limit=1),
        mcp_module.corpus_dispersion_for_scope(scope_id, "policy", offset=2, limit=1),
    ]

    assert mcp_pages == sdk_pages
    assert [page["has_more"] for page in mcp_pages] == [True, True, False]
    documents = [row for page in mcp_pages for row in page["documents"]]
    assert documents == sdk_full["documents"]
    assert len({row["revision_id"] for row in documents}) == len(documents) == 3
    assert all(page["scope_id"] == scope_id for page in mcp_pages)
    assert all(page["total_rows"] == sdk_full["total_rows"] == 3 for page in mcp_pages)
    assert later_revision not in {str(row["revision_id"]) for row in documents}
    assert database.read_bytes() == before
    with Project.open(root, read_only=True) as reopened:
        assert reopened.scope_revisions(scope_id) == expected_members
        assert reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            snapshot_count
        )


@pytest.mark.parametrize(
    ("scope_id", "term", "offset", "limit"),
    [
        ("", "policy", 0, 100),
        (" padded ", "policy", 0, 100),
        ("s" * 129, "policy", 0, 100),
        ("scope", "", 0, 100),
        ("scope", "   ", 0, 100),
        ("scope", "two tokens", 0, 100),
        ("scope", "!!!", 0, 100),
        ("scope", "x" * (mcp_module.MCP_KWIC_MAX_QUERY_CHARS + 1), 0, 100),
        ("scope", True, 0, 100),
        ("scope", "policy", True, 100),
        ("scope", "policy", -1, 100),
        ("scope", "policy", 0x7FFF_FFFF_FFFF_FFFF + 1, 100),
        ("scope", "policy", 0, True),
        ("scope", "policy", 0, 0),
        ("scope", "policy", 0, mcp_module.MCP_SCOPE_CORPUS_MAX_PAGE_SIZE + 1),
    ],
)
def test_mcp_scope_dispersion_rejects_invalid_arguments_before_open(
    monkeypatch, scope_id, term, offset, limit
) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("invalid scoped dispersion arguments must be rejected first"),
    )
    with pytest.raises(ProjectError):
        mcp_module.corpus_dispersion_for_scope(
            scope_id, term, offset=offset, limit=limit
        )


def test_mcp_scope_dispersion_rejects_unknown_scope_read_only(tmp_path, monkeypatch) -> None:
    root = tmp_path / "unknown-scope-dispersion"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "policy word")
    project.snapshot([revision_id])
    project.close()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: Project.open(root, read_only=True))

    with pytest.raises(ScopeViolationError, match="scope"):
        mcp_module.corpus_dispersion_for_scope("unknown-scope", "policy")


def test_mcp_scope_dispersion_response_has_a_64_kib_json_budget() -> None:
    page = {
        "term": "policy",
        "scope_id": "scope",
        "calculation_version": "surface-document-dispersion-v2",
        "document_count": 1,
        "document_frequency": 1,
        "document_range": 1.0,
        "total_occurrences": 1,
        "population_tokens": 1,
        "offset": 0,
        "limit": 1,
        "total_rows": 1,
        "has_more": False,
        "documents": [
            {
                "revision_id": "x" * mcp_module.MAX_MCP_DISPERSION_PAGE_BYTES,
                "occurrences": 1,
                "token_count": 1,
                "per_1000_tokens": 1000.0,
            }
        ],
    }

    with pytest.raises(
        ProjectError,
        match="dispersion page exceeds the 64 KiB UTF-8 JSON response budget",
    ):
        mcp_module._enforce_mcp_dispersion_page_budget(page)


def test_mcp_scope_dispersion_is_discoverable_in_read_only_mode(monkeypatch) -> None:
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        tools = asyncio.run(mcp_module.server.list_tools())
        assert "corpus_dispersion_for_scope" in {tool.name for tool in tools}
        tool = next(item for item in tools if item.name == "corpus_dispersion_for_scope")
        schema = tool.model_dump(by_alias=True)["inputSchema"]
        assert schema["required"] == ["scope_id", "term"]
        properties = schema["properties"]
        assert properties["scope_id"]["minLength"] == 1
        assert properties["scope_id"]["maxLength"] == 128
        assert properties["term"]["minLength"] == 1
        assert properties["term"]["maxLength"] == mcp_module.MCP_KWIC_MAX_QUERY_CHARS
        assert properties["term"]["pattern"] == r"\S"
        assert properties["offset"]["minimum"] == 0
        assert properties["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
        assert properties["limit"]["minimum"] == 1
        assert properties["limit"]["maximum"] == mcp_module.MCP_SCOPE_CORPUS_MAX_PAGE_SIZE
        assert "corpus_dispersion_for_scope" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
