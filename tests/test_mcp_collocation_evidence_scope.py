from __future__ import annotations

import asyncio
import math
from typing import cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError


def _persistence_counts(root) -> tuple[int, int, int]:
    with Project.open(root, read_only=True) as project:
        coverage_count = project.db.execute(
            "SELECT COUNT(*) FROM coverage_descriptors"
        ).fetchone()[0]
        evidence_count = project.db.execute(
            "SELECT COUNT(*) FROM evidence_artifacts"
        ).fetchone()[0]
        snapshot_count = project.db.execute(
            "SELECT COUNT(*) FROM scope_snapshots"
        ).fetchone()[0]
    return coverage_count, evidence_count, snapshot_count


def test_mcp_collocation_evidence_for_scope_persists_exhaustive_frozen_result(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-collocation-evidence"
    project = Project.init(root)
    _, revision_a = project.import_text("a.txt", "apple red apple red apple blue")
    _, revision_b = project.import_text("b.txt", "apple green")
    scope_id = project.snapshot([revision_a, revision_b])
    expected_members = project.scope_revisions(scope_id)
    snapshot_count = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    _, later_revision = project.import_text(
        "later.txt", "apple red apple red apple red"
    )
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)

    def forbid_snapshot(*_args, **_kwargs):
        pytest.fail("scoped collocation evidence must not create a ScopeSnapshot")

    monkeypatch.setattr(Project, "snapshot", forbid_snapshot)
    result = mcp_module.corpus_collocation_evidence_for_scope(
        scope_id, "apple", "red", window=1, min_frequency=2
    )

    assert set(result) == {
        "scope_id",
        "query",
        "collocate",
        "window",
        "focus_frequency",
        "pair_frequency",
        "token_frequency",
        "population_tokens",
        "pair_opportunities",
        "expected_frequency",
        "mutual_information",
        "t_score",
        "coverage_id",
        "evidence_id",
    }
    assert result["scope_id"] == scope_id
    assert result["query"] == "apple"
    assert result["collocate"] == "red"
    assert result["window"] == 1
    assert result["focus_frequency"] == 4
    assert result["pair_frequency"] == 4
    assert result["token_frequency"] == 2
    assert result["population_tokens"] == 8
    assert result["pair_opportunities"] == 6
    assert result["expected_frequency"] == pytest.approx(3.0)
    assert result["mutual_information"] == pytest.approx(math.log2(4 / 3))
    assert result["t_score"] == pytest.approx(0.5)

    with Project.open(root, read_only=True) as reopened:
        assert reopened.scope_revisions(scope_id) == expected_members
        assert later_revision not in expected_members
        assert reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            snapshot_count
        )
        coverage = reopened.coverage_descriptor(str(result["coverage_id"]))
        evidence = reopened.evidence_content(str(result["evidence_id"]))
        assert coverage.coverage_mode.value == "EXHAUSTIVE"
        assert coverage.scope_snapshot_id == scope_id
        assert coverage.population_count == result["pair_opportunities"] == 6
        assert evidence["scope_snapshot_id"] == scope_id
        assert evidence["value"] == pytest.approx(result["mutual_information"])
        provenance = cast(dict[str, object], evidence["provenance"])
        assert provenance["coverage_id"] == result["coverage_id"]
        assert provenance["pair_frequency"] == 4
        assert provenance["pair_opportunities"] == 6
        assert provenance["t_score"] == pytest.approx(0.5)
        assert "not significance tests" in str(provenance["interpretation_note"])


@pytest.mark.parametrize(
    ("scope_id", "query", "collocate", "window", "min_frequency"),
    [
        ("", "apple", "red", 5, 2),
        ("   ", "apple", "red", 5, 2),
        (" padded ", "apple", "red", 5, 2),
        ("s" * 129, "apple", "red", 5, 2),
        (True, "apple", "red", 5, 2),
        (
            "scope",
            "x" * (mcp_module.MCP_SCOPE_COLLOCATION_EVIDENCE_TOKEN_MAX_CHARS + 1),
            "red",
            5,
            2,
        ),
        ("scope", "", "red", 5, 2),
        ("scope", "   ", "red", 5, 2),
        ("scope", "two tokens", "red", 5, 2),
        ("scope", "!!!", "red", 5, 2),
        ("scope", True, "red", 5, 2),
        ("scope", "apple", "", 5, 2),
        ("scope", "apple", "  ", 5, 2),
        ("scope", "apple", "two tokens", 5, 2),
        ("scope", "apple", "!!!", 5, 2),
        (
            "scope",
            "apple",
            "x" * (mcp_module.MCP_SCOPE_COLLOCATION_EVIDENCE_TOKEN_MAX_CHARS + 1),
            5,
            2,
        ),
        ("scope", "apple", True, 5, 2),
        ("scope", "apple", "red", True, 2),
        ("scope", "apple", "red", 0, 2),
        ("scope", "apple", "red", 101, 2),
        ("scope", "apple", "red", "5", 2),
        ("scope", "apple", "red", 5, True),
        ("scope", "apple", "red", 5, 0),
        ("scope", "apple", "red", 5, "2"),
    ],
)
def test_mcp_scope_collocation_evidence_rejects_invalid_values_before_open(
    monkeypatch, scope_id, query, collocate, window, min_frequency
) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("invalid scoped collocation evidence opened the project"),
    )
    with pytest.raises(ProjectError):
        mcp_module.corpus_collocation_evidence_for_scope(
            scope_id, query, collocate, window=window, min_frequency=min_frequency
        )


@pytest.mark.parametrize(
    ("query", "collocate"),
    [("apple", "APPLE"), ("Café", "cafe\u0301")],
)
def test_mcp_scope_collocation_evidence_rejects_normalized_self_pair_before_open(
    monkeypatch, query, collocate
) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("self-collocation evidence must be rejected before opening"),
    )
    with pytest.raises(ProjectError, match="self-collocation"):
        mcp_module.corpus_collocation_evidence_for_scope("scope", query, collocate)


def test_mcp_scope_collocation_evidence_failures_create_no_orphan_records(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-collocation-evidence-failures"
    project = Project.init(root)
    _, normal_revision = project.import_text("normal.txt", "apple red apple")
    normal_scope = project.snapshot([normal_revision])
    _, no_pair_revision = project.import_text("no-pairs.txt", "singleton")
    no_pair_scope = project.snapshot([no_pair_revision])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    before = _persistence_counts(root)

    with pytest.raises(ScopeViolationError, match="scope"):
        mcp_module.corpus_collocation_evidence_for_scope(
            "unknown-scope", "apple", "red", min_frequency=1
        )
    with pytest.raises(ProjectError, match="self-collocation"):
        mcp_module.corpus_collocation_evidence_for_scope(
            normal_scope, "apple", "APPLE", min_frequency=1
        )
    with pytest.raises(ScopeViolationError):
        mcp_module.corpus_collocation_evidence_for_scope(
            normal_scope, "apple", "red", min_frequency=100
        )
    with pytest.raises(ScopeViolationError):
        mcp_module.corpus_collocation_evidence_for_scope(
            no_pair_scope, "apple", "red", min_frequency=1
        )

    assert _persistence_counts(root) == before


def test_mcp_scope_collocation_evidence_is_registered_but_hidden_read_only(
    monkeypatch,
) -> None:
    tools = asyncio.run(mcp_module.server.list_tools())
    names = {tool.name for tool in tools}
    tool_name = "corpus_collocation_evidence_for_scope"
    assert tool_name in names
    assert tool_name not in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    tool = next(item for item in tools if item.name == tool_name)
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert schema["required"] == ["scope_id", "query", "collocate"]
    properties = schema["properties"]
    assert properties["scope_id"]["minLength"] == 1
    assert properties["scope_id"]["maxLength"] == 128
    for token_name in ("query", "collocate"):
        assert properties[token_name]["minLength"] == 1
        assert properties[token_name]["maxLength"] == (
            mcp_module.MCP_SCOPE_COLLOCATION_EVIDENCE_TOKEN_MAX_CHARS
        )
    assert properties["window"]["default"] == 5
    assert properties["window"]["minimum"] == 1
    assert properties["window"]["maximum"] == mcp_module.MCP_SCOPE_CORPUS_MAX_WINDOW
    assert properties["min_frequency"]["default"] == 2
    assert properties["min_frequency"]["minimum"] == 1

    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("read-only collocation evidence must be rejected before opening"),
    )
    try:
        with pytest.raises(ProjectError, match="read-only"):
            mcp_module.corpus_collocation_evidence_for_scope("scope", "apple", "red")

        asyncio.run(mcp_module._configure_read_only_server())
        exposed_names = {item.name for item in asyncio.run(mcp_module.server.list_tools())}
        assert tool_name not in exposed_names
        with pytest.raises(Exception, match=f"Unknown tool: {tool_name}"):
            asyncio.run(
                mcp_module.server.call_tool(
                    tool_name,
                    {
                        "scope_id": "scope",
                        "query": "apple",
                        "collocate": "red",
                        "window": 5,
                        "min_frequency": 2,
                    },
                )
            )
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
