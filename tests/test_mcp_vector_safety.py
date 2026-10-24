from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from typing import Any, cast

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.vectors import VectorChunk, build_vector_index


def test_raw_vector_searches_bound_results_and_persist_scope_for_run_lineage(
    tmp_path, monkeypatch
):
    root = tmp_path / "vector-project"
    project = Project.init(root)
    text = "Apples are green."
    _, revision = project.import_text("selected.txt", text)
    paragraph = project.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (revision,)
    ).fetchone()[0]
    artifact_id = build_vector_index(
        project,
        revision,
        [VectorChunk(paragraph, 0, len(text), (0.9, 0.0))],
    )
    initial_scope_count = int(project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    vector_result = mcp_module.search_vector(revision, [artifact_id], [1.0, 0.0], limit=1)
    hybrid_result = mcp_module.search_hybrid(
        revision, [artifact_id], "Apples are green.", [1.0, 0.0], limit=1
    )

    assert vector_result["scope_id"]
    assert vector_result["scope_id"] != revision
    assert vector_result["retrieval_run_id"]
    assert vector_result["artifact_ids"] == [artifact_id]
    assert vector_result["limit"] == 1
    assert hybrid_result["scope_id"]
    assert hybrid_result["scope_id"] != revision
    assert hybrid_result["retrieval_run_id"]
    assert hybrid_result["artifact_ids"] == [artifact_id]
    assert hybrid_result["limit"] == 1
    vector_hits = vector_result["hits"]
    hybrid_hits = hybrid_result["hits"]
    assert len(vector_hits) == len(hybrid_hits) == 1
    assert vector_hits[0]["revision_id"] == revision
    assert hybrid_hits[0]["revision_id"] == revision
    assert set(vector_hits[0]) == {
        "artifact_id",
        "revision_id",
        "chunk_id",
        "start",
        "end",
        "distance",
    }
    assert set(hybrid_hits[0]) == {
        "revision_id",
        "unit_id",
        "artifact_id",
        "lexical_rank",
        "vector_rank",
        "fusion_score",
    }
    assert hybrid_hits[0]["artifact_id"] == artifact_id
    assert set(vector_result) == {
        "scope_id",
        "retrieval_run_id",
        "artifact_ids",
        "limit",
        "hits",
    }
    assert set(hybrid_result) == set(vector_result)

    with Project.open(root, read_only=True) as reopened:
        final_scope_count = int(
            reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
        )
        assert final_scope_count == initial_scope_count + 2
        persisted_runs = {}
        for result in (vector_result, hybrid_result):
            run = reopened.db.execute(
                "SELECT r.scope_id,r.returned_count,r.query_terms_json,c.operation,c.config_json "
                "FROM retrieval_runs r JOIN retrieval_run_context c "
                "ON c.retrieval_run_id=r.id WHERE r.id=?",
                (result["retrieval_run_id"],),
            ).fetchone()
            assert run is not None
            assert run["scope_id"] == result["scope_id"]
            assert reopened.scope_revisions(str(run["scope_id"])) == [revision]
            config = json.loads(run["config_json"])
            assert config["artifact_ids"] == [artifact_id]
            assert config["vector_provenance"] == "caller-supplied-raw-vector"
            assert run["returned_count"] == len(result["hits"])
            assert "Apples are green." not in str(run["query_terms_json"])
            hit_rows = reopened.db.execute(
                "SELECT h.rank,c.artifact_id,c.lexical_rank,c.vector_rank "
                "FROM retrieval_hits h LEFT JOIN retrieval_hit_context c "
                "ON c.retrieval_hit_id=h.id WHERE h.retrieval_run_id=? ORDER BY h.rank",
                (result["retrieval_run_id"],),
            ).fetchall()
            assert [row["rank"] for row in hit_rows] == list(
                range(1, len(result["hits"]) + 1)
            )
            assert all(row["artifact_id"] == artifact_id for row in hit_rows)
            persisted_runs[str(run["operation"])] = (config, hit_rows)

        vector_config, vector_rows = persisted_runs["search_vectors_with_provenance"]
        vector_bytes = struct.pack("<2f", 1.0, 0.0)
        assert vector_config["query_sha256"] == hashlib.sha256(vector_bytes).hexdigest()
        assert vector_config["query_dimension"] == 2
        assert "omitted" in vector_config["query_representation"].lower()
        assert "[1.0, 0.0]" not in json.dumps(vector_config)
        assert vector_rows[0]["lexical_rank"] is None
        assert vector_rows[0]["vector_rank"] == 1

        hybrid_config, hybrid_rows = persisted_runs["hybrid_search_with_provenance"]
        assert hybrid_config["query_sha256"] == hashlib.sha256(
            b"Apples are green."
        ).hexdigest()
        assert hybrid_config["query_vector_sha256"] == hashlib.sha256(
            vector_bytes
        ).hexdigest()
        assert hybrid_config["query_vector_dimension"] == 2
        assert "omitted" in hybrid_config["query_representation"].lower()
        assert "Apples are green." not in json.dumps(hybrid_config)
        assert "[1.0, 0.0]" not in json.dumps(hybrid_config)
        assert hybrid_rows[0]["lexical_rank"] == 1
        assert hybrid_rows[0]["vector_rank"] == 1


def test_raw_vector_search_error_leaves_scope_but_no_retrieval_run(tmp_path, monkeypatch):
    root = tmp_path / "invalid-vector-project"
    project = Project.init(root)
    _, revision = project.import_text("selected.txt", "Apples are green.")
    initial_scope_count = int(project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError):
        mcp_module.search_vector(revision, ["missing-artifact"], [1.0, 0.0])

    with Project.open(root, read_only=True) as reopened:
        final_scope_count = int(
            reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
        )
        assert final_scope_count == initial_scope_count + 1
        scope_row = reopened.db.execute(
            "SELECT id FROM scope_snapshots ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        assert scope_row is not None
        assert reopened.scope_revisions(str(scope_row["id"])) == [revision]
        assert reopened.db.execute(
            "SELECT COUNT(*) FROM retrieval_runs WHERE scope_id=?", (scope_row["id"],)
        ).fetchone()[0] == 0


def test_raw_vector_searches_return_existing_scope_and_bound_hits(tmp_path, monkeypatch):
    root = tmp_path / "multi-revision-vector-project"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Apples grow in the north.")
    _, second_revision = project.import_text("second.txt", "Apples grow in the south.")
    _, outside_revision = project.import_text("outside.txt", "Apples grow outside scope.")
    selected_artifacts: list[str] = []
    for revision_id in (first_revision, second_revision):
        paragraph_id = project.db.execute(
            "SELECT id FROM paragraphs WHERE revision_id=?", (revision_id,)
        ).fetchone()[0]
        selected_artifacts.append(
            build_vector_index(
                project,
                revision_id,
                [VectorChunk(paragraph_id, 0, 6, (0.9, 0.0))],
            )
        )
    outside_paragraph = project.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (outside_revision,)
    ).fetchone()[0]
    outside_artifact = build_vector_index(
        project,
        outside_revision,
        [VectorChunk(outside_paragraph, 0, 6, (0.9, 0.0))],
    )
    scope_id = project.snapshot([first_revision, second_revision])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    vector_result = mcp_module.search_vector(scope_id, selected_artifacts, [1.0, 0.0])
    hybrid_result = mcp_module.search_hybrid(
        scope_id, selected_artifacts, "Apples", [1.0, 0.0]
    )

    assert vector_result["scope_id"] == scope_id
    assert hybrid_result["scope_id"] == scope_id
    assert {hit["revision_id"] for hit in vector_result["hits"]} == {
        first_revision,
        second_revision,
    }
    assert {hit["revision_id"] for hit in hybrid_result["hits"]} == {
        first_revision,
        second_revision,
    }
    assert outside_revision not in {hit["revision_id"] for hit in vector_result["hits"]}
    assert outside_revision not in {hit["revision_id"] for hit in hybrid_result["hits"]}
    assert all(hit["artifact_id"] != outside_artifact for hit in vector_result["hits"])


def test_raw_vector_inputs_are_bounded_before_opening_project(monkeypatch):
    def unexpected_project_open():
        pytest.fail("invalid vector arguments must be rejected before opening a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    artifact_ids = [f"artifact-{index}" for index in range(101)]
    with pytest.raises(ToolError, match="1..100 unique"):
        mcp_module.search_vector("revision", artifact_ids, [1.0, 0.0])
    with pytest.raises(ToolError, match="1..4000 characters"):
        mcp_module.search_hybrid("revision", ["artifact"], "x" * 4_001, [1.0, 0.0])


def test_mcp_vector_tool_schemas_are_typed_bounded_and_read_only_excluded():
    tools = asyncio.run(mcp_module.server.list_tools())
    by_name = {tool.name: tool.model_dump(by_alias=True) for tool in tools}
    for name in (
        "search_vector",
        "search_hybrid",
        "search_semantic_with_preset",
        "search_hybrid_with_preset",
    ):
        schema = by_name[name]["inputSchema"]
        assert {"artifact_ids", "limit"} <= set(schema["properties"])
        assert schema["properties"]["artifact_ids"]["maxItems"] == 100
        assert schema["properties"]["limit"]["maximum"] == 100
        assert name not in mcp_module.READ_ONLY_MCP_TOOL_NAMES

    raw_hybrid_properties = by_name["search_hybrid"]["inputSchema"]["properties"]
    assert raw_hybrid_properties["lexical_query"]["maxLength"] == 4_000
    assert "not sent to a model provider" in raw_hybrid_properties["lexical_query"]["description"]
    for name in ("search_vector", "search_hybrid"):
        vector_schema = by_name[name]["inputSchema"]["properties"]["vector_query"]
        assert vector_schema["minItems"] == 1
        assert vector_schema["maxItems"] == mcp_module.MCP_VECTOR_QUERY_MAX_DIMENSION

    for name in ("search_vector", "search_hybrid"):
        output_properties = by_name[name]["outputSchema"]["properties"]
        assert set(output_properties) == {
            "scope_id",
            "retrieval_run_id",
            "artifact_ids",
            "limit",
            "hits",
        }

    hybrid_output = by_name["search_hybrid_with_preset"]["outputSchema"]
    assert set(hybrid_output["properties"]) == {
        "scope_id",
        "retrieval_run_id",
        "artifact_ids",
        "preset_name",
        "query_characters",
        "limit",
        "hits",
    }

    async def read_only_surface() -> set[str]:
        await mcp_module._configure_read_only_server()
        return {tool.name for tool in await mcp_module.server.list_tools()}

    try:
        exposed = asyncio.run(read_only_surface())
    finally:
        mcp_module._restore_full_server_registry()
    assert not {
        "search_vector",
        "search_hybrid",
        "search_semantic_with_preset",
        "search_hybrid_with_preset",
    }.intersection(exposed)


def test_raw_vector_dimensions_and_components_are_bounded_before_project_open(monkeypatch):
    def unexpected_project_open():
        pytest.fail("invalid raw vector must be rejected before opening a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    oversized = [0.0] * (mcp_module.MCP_VECTOR_QUERY_MAX_DIMENSION + 1)
    with pytest.raises(ToolError, match="1..16384"):
        mcp_module.search_vector("revision", ["artifact"], oversized)
    with pytest.raises(ToolError, match="1..16384"):
        mcp_module.search_hybrid("revision", ["artifact"], "query", oversized)

    for invalid in ([], [float("nan")], [float("inf")], [True], ["1.0"], [10**400]):
        with pytest.raises(ToolError, match="1..16384"):
            mcp_module.search_vector("revision", ["artifact"], cast(Any, invalid))
