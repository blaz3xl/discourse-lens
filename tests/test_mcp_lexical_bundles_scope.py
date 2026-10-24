from __future__ import annotations

import asyncio

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError


def test_mcp_scoped_lexical_bundle_pages_match_core_and_preserve_frozen_scope(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-lexical-bundles"
    project = Project.init(root)
    _, revision_a = project.import_text("a.txt", "alpha beta gamma delta alpha beta")
    _, revision_b = project.import_text("b.txt", "alpha beta delta zeta")
    scope_id = project.snapshot([revision_a, revision_b])
    expected_members = project.scope_revisions(scope_id)
    arguments = {
        "size": 2,
        "min_frequency": 1,
        "min_document_frequency": 1,
    }
    sdk_full = project.corpus_lexical_bundles(scope_id, **arguments)
    sdk_pages = [
        project.corpus_lexical_bundles(scope_id, **arguments, offset=0, limit=2),
        project.corpus_lexical_bundles(scope_id, **arguments, offset=2, limit=2),
        project.corpus_lexical_bundles(scope_id, **arguments, offset=4, limit=2),
    ]
    snapshot_count = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    _, later_revision = project.import_text("later.txt", "outsideonly laterflag")
    project.close()

    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    def forbid_snapshot(*_args, **_kwargs):
        pytest.fail("scoped lexical bundles must not create a ScopeSnapshot")

    monkeypatch.setattr(Project, "snapshot", forbid_snapshot)
    mcp_pages = [
        mcp_module.corpus_lexical_bundles_for_scope(
            scope_id, **arguments, offset=0, limit=2
        ),
        mcp_module.corpus_lexical_bundles_for_scope(
            scope_id, **arguments, offset=2, limit=2
        ),
        mcp_module.corpus_lexical_bundles_for_scope(
            scope_id, **arguments, offset=4, limit=2
        ),
    ]

    assert mcp_pages == sdk_pages
    assert [page["has_more"] for page in mcp_pages] == [True, True, False]
    assert [page["truncated"] for page in mcp_pages] == [True, True, False]
    all_rows = [row for page in mcp_pages for row in page["rows"]]
    assert all_rows == sdk_full["rows"]
    assert len({row["bundle"] for row in all_rows}) == len(all_rows) == 6
    assert all(page["candidate_count"] == sdk_full["candidate_count"] for page in mcp_pages)
    assert all(page["scope_id"] == scope_id for page in mcp_pages)
    assert "outsideonly laterflag" not in {row["bundle"] for row in all_rows}
    assert later_revision not in expected_members
    assert database.read_bytes() == before
    with Project.open(root, read_only=True) as reopened:
        assert reopened.scope_revisions(scope_id) == expected_members
        assert reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            snapshot_count
        )


@pytest.mark.parametrize(
    ("scope_id", "size", "min_frequency", "min_document_frequency", "offset", "limit"),
    [
        ("", 3, 2, 1, 0, 100),
        (" padded ", 3, 2, 1, 0, 100),
        ("s" * 129, 3, 2, 1, 0, 100),
        ("scope", True, 2, 1, 0, 100),
        ("scope", 1, 2, 1, 0, 100),
        ("scope", 11, 2, 1, 0, 100),
        ("scope", "2", 2, 1, 0, 100),
        ("scope", 3, True, 1, 0, 100),
        ("scope", 3, 0, 1, 0, 100),
        ("scope", 3, "2", 1, 0, 100),
        ("scope", 3, 2, True, 0, 100),
        ("scope", 3, 2, 0, 0, 100),
        ("scope", 3, 2, "1", 0, 100),
        ("scope", 3, 2, 1, True, 100),
        ("scope", 3, 2, 1, -1, 100),
        ("scope", 3, 2, 1, 0x7FFF_FFFF_FFFF_FFFF + 1, 100),
        ("scope", 3, 2, 1, 0, True),
        ("scope", 3, 2, 1, 0, 0),
        ("scope", 3, 2, 1, 0, 1_001),
    ],
)
def test_mcp_scoped_lexical_bundles_rejects_invalid_values_before_open(
    monkeypatch, scope_id, size, min_frequency, min_document_frequency, offset, limit
) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("invalid scoped lexical-bundle inputs must be rejected first"),
    )
    with pytest.raises(ProjectError):
        mcp_module.corpus_lexical_bundles_for_scope(
            scope_id,
            size,
            min_frequency,
            min_document_frequency,
            offset,
            limit,
        )


def test_mcp_scoped_lexical_bundles_rejects_unknown_scope_read_only(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "unknown-scope-lexical-bundles"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "alpha beta gamma")
    project.snapshot([revision_id])
    project.close()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: Project.open(root, read_only=True))

    with pytest.raises(ScopeViolationError, match="scope"):
        mcp_module.corpus_lexical_bundles_for_scope("unknown-scope")


def test_mcp_scoped_lexical_bundles_preflights_oversized_real_bundle(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "oversized-real-lexical-bundle"
    project = Project.init(root)
    token = "x" * mcp_module.MAX_MCP_LEXICAL_BUNDLE_PAGE_BYTES
    _, revision_id = project.import_text("large.txt", f"left {token}")
    scope_id = project.snapshot([revision_id])
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    def unexpected_json_serialization(*_args, **_kwargs):
        pytest.fail("oversized raw bundle text must be rejected before JSON serialization")

    monkeypatch.setattr(mcp_module.json, "dumps", unexpected_json_serialization)
    with pytest.raises(
        ProjectError,
        match="lexical-bundle page exceeds the 64 KiB UTF-8 JSON response budget",
    ):
        mcp_module.corpus_lexical_bundles_for_scope(
            scope_id, size=2, min_frequency=1
        )


def test_mcp_lexical_bundle_response_budget_counts_unicode_json_escapes() -> None:
    page = {
        "scope_id": "scope",
        "calculation_version": "surface-lexical-bundles-v1",
        "size": 2,
        "min_frequency": 1,
        "min_document_frequency": 1,
        "candidate_count": 1,
        "offset": 0,
        "limit": 1,
        "has_more": False,
        "truncated": False,
        "rows": [{"bundle": "é" * 11_000, "frequency": 1, "document_frequency": 1}],
    }

    with pytest.raises(
        ProjectError,
        match="lexical-bundle page exceeds the 64 KiB UTF-8 JSON response budget",
    ):
        mcp_module._enforce_mcp_lexical_bundle_page_budget(page)


def test_mcp_scoped_lexical_bundles_is_discoverable_in_read_only_mode(monkeypatch) -> None:
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        tools = asyncio.run(mcp_module.server.list_tools())
        assert "corpus_lexical_bundles_for_scope" in {tool.name for tool in tools}
        tool = next(
            item for item in tools if item.name == "corpus_lexical_bundles_for_scope"
        )
        schema = tool.model_dump(by_alias=True)["inputSchema"]
        assert schema["required"] == ["scope_id"]
        properties = schema["properties"]
        assert properties["scope_id"]["minLength"] == 1
        assert properties["scope_id"]["maxLength"] == 128
        assert properties["size"]["default"] == 3
        assert properties["size"]["minimum"] == 2
        assert properties["size"]["maximum"] == 10
        assert properties["min_frequency"]["default"] == 2
        assert properties["min_frequency"]["minimum"] == 1
        assert properties["min_document_frequency"]["default"] == 1
        assert properties["min_document_frequency"]["minimum"] == 1
        assert properties["offset"]["default"] == 0
        assert properties["offset"]["minimum"] == 0
        assert properties["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
        assert properties["limit"]["default"] == 100
        assert properties["limit"]["minimum"] == 1
        assert properties["limit"]["maximum"] == mcp_module.MCP_SCOPE_CORPUS_MAX_PAGE_SIZE
        assert "corpus_lexical_bundles_for_scope" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
