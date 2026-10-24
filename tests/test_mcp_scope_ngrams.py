from __future__ import annotations

import asyncio

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError


def test_mcp_scope_ngram_pages_match_sdk_and_preserve_frozen_membership(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-ngrams"
    project = Project.init(root)
    _, revision_a = project.import_text("a.txt", "red one red two red")
    _, revision_b = project.import_text("b.txt", "blue one blue two blue")
    scope_id = project.snapshot([revision_a, revision_b])
    expected_members = project.scope_revisions(scope_id)
    sdk_full = project.corpus_ngram_page(scope_id, 2, offset=0, limit=100)
    sdk_pages = [
        project.corpus_ngram_page(scope_id, 2, offset=0, limit=3),
        project.corpus_ngram_page(scope_id, 2, offset=3, limit=3),
        project.corpus_ngram_page(scope_id, 2, offset=6, limit=3),
    ]
    snapshot_count = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    _, later_revision = project.import_text("later.txt", "later unique")
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    def forbid_snapshot(*_args, **_kwargs):
        pytest.fail("scope n-grams must not create a ScopeSnapshot")

    monkeypatch.setattr(Project, "snapshot", forbid_snapshot)
    mcp_pages = [
        mcp_module.corpus_ngrams_for_scope(scope_id, 2, offset=0, limit=3),
        mcp_module.corpus_ngrams_for_scope(scope_id, 2, offset=3, limit=3),
        mcp_module.corpus_ngrams_for_scope(scope_id, 2, offset=6, limit=3),
    ]

    assert mcp_pages == sdk_pages
    all_rows = [row for page in mcp_pages for row in page["rows"]]
    assert len(all_rows) == sdk_full["total_values"]
    assert len({row["ngram"] for row in all_rows}) == len(all_rows)
    assert all_rows == sdk_full["rows"]
    assert [page["has_more"] for page in mcp_pages] == [True, True, False]
    assert all(page["scope_id"] == scope_id and page["size"] == 2 for page in mcp_pages)
    assert "red blue" not in {row["ngram"] for row in all_rows}
    assert "later unique" not in {row["ngram"] for row in all_rows}
    assert later_revision not in expected_members
    assert database.read_bytes() == before
    with Project.open(root, read_only=True) as reopened:
        assert reopened.scope_revisions(scope_id) == expected_members
        assert reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            snapshot_count
        )


@pytest.mark.parametrize(
    ("scope_id", "size", "offset", "limit"),
    [
        ("", 1, 0, 100),
        ("  ", 1, 0, 100),
        ("s" * 129, 1, 0, 100),
        ("scope", True, 0, 100),
        ("scope", 0, 0, 100),
        ("scope", -1, 0, 100),
        ("scope", "2", 0, 100),
        ("scope", 1, -1, 100),
        ("scope", 1, True, 100),
        ("scope", 1, 0x7FFF_FFFF_FFFF_FFFF + 1, 100),
        ("scope", 1, 0, 0),
        ("scope", 1, 0, 1_001),
        ("scope", 1, 0, False),
    ],
)
def test_mcp_scope_ngrams_rejects_invalid_arguments_before_open(
    monkeypatch, scope_id, size, offset, limit
) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("invalid scoped n-gram arguments must be rejected first"),
    )
    with pytest.raises(ProjectError):
        mcp_module.corpus_ngrams_for_scope(scope_id, size, offset=offset, limit=limit)


def test_mcp_scope_ngrams_rejects_unknown_scope_read_only(tmp_path, monkeypatch) -> None:
    root = tmp_path / "unknown-scope-ngrams"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "focus word")
    project.snapshot([revision_id])
    project.close()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: Project.open(root, read_only=True))

    with pytest.raises(ScopeViolationError, match="scope"):
        mcp_module.corpus_ngrams_for_scope("unknown-scope", 1)


def test_mcp_scope_ngrams_rejects_oversized_token_before_serialization(tmp_path, monkeypatch):
    root = tmp_path / "oversized-scope-ngrams"
    project = Project.init(root)
    long_token = "x" * (65 * 1024)
    _, revision_id = project.import_text("large-token.txt", f"focus {long_token}")
    scope_id = project.snapshot([revision_id])
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    def unexpected_serialization(*_args, **_kwargs):
        pytest.fail("oversized raw n-gram text must be rejected before JSON serialization")

    monkeypatch.setattr(mcp_module.json, "dumps", unexpected_serialization)
    with pytest.raises(
        ProjectError,
        match="n-gram page exceeds the 64 KiB UTF-8 JSON response budget",
    ):
        mcp_module.corpus_ngrams_for_scope(scope_id, 1)

    assert database.read_bytes() == before


def test_mcp_scope_ngram_json_budget_rejects_escaped_unicode_expansion() -> None:
    page = {
        "scope_id": "scope",
        "size": 1,
        "offset": 0,
        "limit": 1,
        "rows": [{"ngram": "é" * 12_000, "count": 1}],
        "total_values": 1,
        "has_more": False,
    }

    with pytest.raises(
        ProjectError,
        match="n-gram page exceeds the 64 KiB UTF-8 JSON response budget",
    ):
        mcp_module._enforce_mcp_ngram_page_budget(page)


def test_mcp_scope_ngrams_is_discoverable_with_read_only_schema(monkeypatch) -> None:
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        tools = asyncio.run(mcp_module.server.list_tools())
        assert "corpus_ngrams_for_scope" in {tool.name for tool in tools}
        tool = next(item for item in tools if item.name == "corpus_ngrams_for_scope")
        schema = tool.model_dump(by_alias=True)["inputSchema"]
        assert schema["required"] == ["scope_id", "size"]
        assert schema["properties"]["scope_id"]["minLength"] == 1
        assert schema["properties"]["scope_id"]["maxLength"] == 128
        assert schema["properties"]["size"]["minimum"] == 1
        assert schema["properties"]["offset"]["minimum"] == 0
        assert schema["properties"]["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
        assert schema["properties"]["limit"]["minimum"] == 1
        assert schema["properties"]["limit"]["maximum"] == (
            mcp_module.MCP_SCOPE_CORPUS_MAX_PAGE_SIZE
        )
        assert "corpus_ngrams_for_scope" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
