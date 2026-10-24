from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, cast

import pytest
from mcp_types._types import CallToolResult, TextContent
from typer.testing import CliRunner

import discourselens.cli as cli_module
import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.cli import app
from discourselens.errors import ScopeViolationError
from discourselens.models import CredentialReference, ModelPreset
from discourselens.vectors import (
    RetrievalSearchResult,
    VectorChunk,
    build_paragraph_embedding_index,
    build_vector_index,
    hybrid_search,
)


class _LocalEmbeddingFixture:
    provider_id = "cohere-embeddings"
    endpoint = "https://api.cohere.com"

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, list[str]]] = []
        self.query_calls: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.document_calls.append((model, texts))
        return [(0.2, 0.8) for _ in texts]

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.query_calls.append((model, texts))
        return [(0.2, 0.8) for _ in texts]


def test_sdk_cli_and_mcp_hybrid_preset_paths_match_through_core(tmp_path, monkeypatch):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    try:
        _, selected_revision = project.import_text("selected.txt", "Apples are green.")
        _, outside_revision = project.import_text("outside.txt", "Apples are not local.")
        project.save_credential_reference(
            CredentialReference("cohere-embeddings", "parity", "DL_VECTOR_PARITY_SECRET")
        )
        project.save_model_preset(
            ModelPreset("parity-preset", "cohere-embeddings", "parity", "embedding-model")
        )
        provider = _LocalEmbeddingFixture()
        selected_artifact = build_paragraph_embedding_index(
            project, "parity-preset", selected_revision, provider
        )
        outside_artifact = build_paragraph_embedding_index(
            project, "parity-preset", outside_revision, provider
        )
        explicit_scope = project.snapshot([selected_revision])
    finally:
        project.close()

    query = "Apples"
    query_file = tmp_path / "query.txt"
    query_file.write_text(query, encoding="utf-8")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DL_VECTOR_PARITY_SECRET", "fixture-secret-must-not-be-read-or-returned")

    def forbidden_boundary(*args, **kwargs):
        raise AssertionError("parity test must not resolve secrets or access the network")

    monkeypatch.setattr("discourselens.secrets.ReferenceSecretStore.get", forbidden_boundary)
    monkeypatch.setattr("discourselens.providers._open_provider_request", forbidden_boundary)

    with Project.open(project_root) as sdk_project:
        sdk_result = sdk_project.hybrid_search_with_preset(
            "parity-preset",
            explicit_scope,
            [selected_artifact],
            query,
            embedding_provider=provider,
            limit=5,
        )
        assert isinstance(sdk_result, list)
        assert sdk_result.hits == list(sdk_result)
        assert sdk_result.scope_id == explicit_scope
        assert sdk_result.retrieval_run_id
        sdk_hits = []
        for hit in sdk_result.hits:
            payload = asdict(hit)
            payload.pop("retrieval_run_id", None)
            sdk_hits.append(payload)

    cli_core_entrypoint = cli_module.hybrid_search_with_preset

    def cli_with_fixture(project, preset, scope_id, artifact_ids, query_text, **kwargs):
        return cli_core_entrypoint(
            project,
            preset,
            scope_id,
            artifact_ids,
            query_text,
            embedding_provider=provider,
            **kwargs,
        )

    with monkeypatch.context() as cli_patch:
        cli_patch.setattr(cli_module, "hybrid_search_with_preset", cli_with_fixture)
        cli_result = CliRunner().invoke(
            app,
            [
                "vectors",
                "hybrid-search-preset",
                str(project_root),
                "parity-preset",
                selected_revision,
                selected_artifact,
                "--query-file",
                str(query_file),
                "--limit",
                "5",
            ],
        )
    assert cli_result.exit_code == 0, cli_result.output
    cli_envelope = json.loads(cli_result.output)
    assert isinstance(cli_envelope, dict)
    assert isinstance(cli_envelope["scope_id"], str)
    assert cli_envelope["scope_id"] != explicit_scope
    assert isinstance(cli_envelope["retrieval_run_id"], str)
    cli_hits = cli_envelope["hits"]
    assert isinstance(cli_hits, list)
    with Project.open(project_root, read_only=True) as reopened:
        assert reopened.scope_revisions(cli_envelope["scope_id"]) == [selected_revision]

    project_method = Project.hybrid_search_with_preset

    def mcp_with_fixture(
        project,
        preset_name,
        scope_id,
        artifact_ids,
        query_text,
        *,
        limit=10,
        rrf_constant=60,
    ):
        return project_method(
            project,
            preset_name,
            scope_id,
            artifact_ids,
            query_text,
            embedding_provider=provider,
            limit=limit,
            rrf_constant=rrf_constant,
        )

    with monkeypatch.context() as mcp_patch:
        mcp_patch.setattr(Project, "hybrid_search_with_preset", mcp_with_fixture)
        mcp_module._restore_full_server_registry()

        async def invoke_mcp():
            return await mcp_module.server.call_tool(
                "search_hybrid_with_preset",
                {
                    "scope_id": explicit_scope,
                    "artifact_ids": [selected_artifact],
                    "preset_name": "parity-preset",
                    "query_text": query,
                    "limit": 5,
                },
            )

        try:
            mcp_result = cast(CallToolResult, asyncio.run(invoke_mcp()))
        finally:
            mcp_module._restore_full_server_registry()
    assert not mcp_result.is_error
    assert len(mcp_result.content) == 1
    mcp_content = mcp_result.content[0]
    assert isinstance(mcp_content, TextContent)
    mcp_text = mcp_content.text
    mcp_envelope = json.loads(mcp_text)
    assert isinstance(mcp_envelope, dict)
    assert mcp_envelope["scope_id"] == explicit_scope
    assert isinstance(mcp_envelope["retrieval_run_id"], str)
    assert mcp_envelope["artifact_ids"] == [selected_artifact]
    mcp_hits = mcp_envelope["hits"]
    assert isinstance(mcp_hits, list)

    expected_hit_fields = {
        "revision_id",
        "unit_id",
        "artifact_id",
        "lexical_rank",
        "vector_rank",
        "fusion_score",
    }
    assert sdk_hits and cli_hits and mcp_hits
    assert all(set(row) == expected_hit_fields for rows in (sdk_hits, cli_hits, mcp_hits) for row in rows)
    assert sdk_hits == cli_hits == mcp_hits
    assert {row["revision_id"] for row in sdk_hits} == {selected_revision}
    assert outside_revision not in {row["revision_id"] for row in sdk_hits}
    assert provider.query_calls == [("embedding-model", [query])] * 3

    # Independently preserve the Core scope gate: the outside index cannot be
    # queried through the SDK wrapper, and is rejected before provider embed.
    with Project.open(project_root) as sdk_project, pytest.raises(ScopeViolationError):
        sdk_project.hybrid_search_with_preset(
            "parity-preset",
            explicit_scope,
            [outside_artifact],
            query,
            embedding_provider=provider,
            limit=5,
        )
    assert provider.query_calls == [("embedding-model", [query])] * 3

    outputs = [cli_result.output, mcp_text, json.dumps(sdk_hits)]
    for output in outputs:
        assert query not in output
        assert "DL_VECTOR_PARITY_SECRET" not in output
        assert "fixture-secret-must-not-be-read-or-returned" not in output


def test_sdk_cli_and_mcp_semantic_preset_paths_match_and_keep_hit_provenance(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "semantic-research"
    project = Project.init(project_root)
    try:
        _, selected_revision = project.import_text("selected.txt", "Apples are green.")
        _, outside_revision = project.import_text("outside.txt", "Apples are not local.")
        project.save_credential_reference(
            CredentialReference("cohere-embeddings", "semantic-parity", "DL_SEMANTIC_PARITY_SECRET")
        )
        project.save_model_preset(
            ModelPreset("semantic-preset", "cohere-embeddings", "semantic-parity", "embedding-model")
        )
        provider = _LocalEmbeddingFixture()
        selected_artifact = build_paragraph_embedding_index(
            project, "semantic-preset", selected_revision, provider
        )
        outside_artifact = build_paragraph_embedding_index(
            project, "semantic-preset", outside_revision, provider
        )
        scope_id = project.snapshot([selected_revision])
    finally:
        project.close()

    query = "Apples semantic query"
    query_file = tmp_path / "semantic-query.txt"
    query_file.write_text(query, encoding="utf-8")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DL_SEMANTIC_PARITY_SECRET", "never-return-this-secret")

    def forbidden_boundary(*args, **kwargs):
        raise AssertionError("mock semantic parity must not resolve secrets or access the network")

    monkeypatch.setattr("discourselens.secrets.ReferenceSecretStore.get", forbidden_boundary)
    monkeypatch.setattr("discourselens.providers._open_provider_request", forbidden_boundary)

    with Project.open(project_root) as sdk_project:
        sdk_result = sdk_project.semantic_search_with_preset(
            "semantic-preset",
            scope_id,
            [selected_artifact],
            query,
            embedding_provider=provider,
            limit=5,
        )
        assert isinstance(sdk_result, list)
        assert sdk_result.hits == list(sdk_result)
        assert sdk_result.scope_id == scope_id
        assert sdk_result.retrieval_run_id
        sdk_hits = []
        for hit in sdk_result.hits:
            payload = asdict(hit)
            payload.pop("retrieval_run_id", None)
            sdk_hits.append(payload)

    cli_core_entrypoint = cli_module.semantic_search_with_preset

    def cli_with_fixture(project, preset, selected_scope, artifact_ids, query_text, **kwargs):
        return cli_core_entrypoint(
            project,
            preset,
            selected_scope,
            artifact_ids,
            query_text,
            embedding_provider=provider,
            **kwargs,
        )

    with monkeypatch.context() as cli_patch:
        cli_patch.setattr(cli_module, "semantic_search_with_preset", cli_with_fixture)
        cli_result = CliRunner().invoke(
            app,
            [
                "vectors",
                "search",
                str(project_root),
                "semantic-preset",
                selected_revision,
                selected_artifact,
                "--query-file",
                str(query_file),
                "--limit",
                "5",
            ],
        )
    assert cli_result.exit_code == 0, cli_result.output
    cli_envelope = json.loads(cli_result.output)
    assert isinstance(cli_envelope, dict)
    assert isinstance(cli_envelope["scope_id"], str)
    assert isinstance(cli_envelope["retrieval_run_id"], str)
    cli_hits = cli_envelope["hits"]

    project_method = Project.semantic_search_with_preset

    def mcp_with_fixture(
        project,
        preset_name,
        selected_scope,
        artifact_ids,
        query_text,
        *,
        embedding_provider=None,
        limit=10,
    ):
        return project_method(
            project,
            preset_name,
            selected_scope,
            artifact_ids,
            query_text,
            embedding_provider=provider,
            limit=limit,
        )

    with monkeypatch.context() as mcp_patch:
        mcp_patch.setattr(Project, "semantic_search_with_preset", mcp_with_fixture)
        mcp_module._restore_full_server_registry()

        async def invoke_mcp():
            return await mcp_module.server.call_tool(
                "search_semantic_with_preset",
                {
                    "scope_id": scope_id,
                    "artifact_ids": [selected_artifact],
                    "preset_name": "semantic-preset",
                    "query_text": query,
                    "limit": 5,
                },
            )

        try:
            mcp_result = cast(CallToolResult, asyncio.run(invoke_mcp()))
        finally:
            mcp_module._restore_full_server_registry()

        with pytest.raises(ScopeViolationError):
            mcp_module.search_semantic_with_preset(
                scope_id,
                [outside_artifact],
                "semantic-preset",
                query,
                limit=5,
            )

        mcp_module._MCP_READ_ONLY = True
        try:
            async def invoke_read_only_tool():
                await mcp_module._configure_read_only_server()
                exposed = {tool.name for tool in await mcp_module.server.list_tools()}
                assert "search_semantic_with_preset" not in exposed
                with pytest.raises(Exception, match="Unknown tool: search_semantic_with_preset"):
                    await mcp_module.server.call_tool("search_semantic_with_preset", {})

            asyncio.run(invoke_read_only_tool())
        finally:
            mcp_module._MCP_READ_ONLY = False
            mcp_module._restore_full_server_registry()

    assert not mcp_result.is_error
    assert isinstance(mcp_result.structured_content, dict)
    mcp_payload = mcp_result.structured_content
    assert isinstance(mcp_payload["retrieval_run_id"], str)
    mcp_hits = mcp_payload["hits"]

    assert sdk_hits and cli_hits and mcp_hits
    assert sdk_hits == cli_hits == mcp_hits
    assert {row["revision_id"] for row in sdk_hits} == {selected_revision}
    assert {row["artifact_id"] for row in sdk_hits} == {selected_artifact}
    assert outside_revision not in {row["revision_id"] for row in sdk_hits}
    assert mcp_payload["scope_id"] == scope_id
    assert mcp_payload["artifact_ids"] == [selected_artifact]
    assert mcp_payload["preset_name"] == "semantic-preset"
    assert mcp_payload["query_characters"] == len(query)
    assert provider.query_calls == [("embedding-model", [query])] * 3

    for output in (cli_result.output, json.dumps(mcp_payload), json.dumps(sdk_hits)):
        assert query not in output
        assert "DL_SEMANTIC_PARITY_SECRET" not in output
        assert "never-return-this-secret" not in output


@pytest.mark.parametrize(
    ("operation", "cli_command", "mcp_tool"),
    [
        ("semantic_search_with_preset", "search", "search_semantic_with_preset"),
        (
            "hybrid_search_with_preset",
            "hybrid-search-preset",
            "search_hybrid_with_preset",
        ),
    ],
)
def test_cli_preset_search_accepts_the_same_multirevision_scope_as_core_and_mcp(
    tmp_path, monkeypatch, operation, cli_command, mcp_tool
):
    project_root = tmp_path / "multi-revision-scope"
    project = Project.init(project_root)
    try:
        _, first_revision = project.import_text("first.txt", "Apples grow in the north.")
        _, second_revision = project.import_text("second.txt", "Apples grow in the south.")
        _, outside_revision = project.import_text("outside.txt", "Apples grow outside scope.")
        project.save_credential_reference(
            CredentialReference("cohere-embeddings", "multi-scope", "DL_MULTI_SCOPE_SECRET")
        )
        project.save_model_preset(
            ModelPreset("multi-scope-preset", "cohere-embeddings", "multi-scope", "embedding-model")
        )
        provider = _LocalEmbeddingFixture()
        first_artifact = build_paragraph_embedding_index(
            project, "multi-scope-preset", first_revision, provider
        )
        second_artifact = build_paragraph_embedding_index(
            project, "multi-scope-preset", second_revision, provider
        )
        outside_artifact = build_paragraph_embedding_index(
            project, "multi-scope-preset", outside_revision, provider
        )
        scope_id = project.snapshot([first_revision, second_revision])
    finally:
        project.close()

    query = "Apples"
    query_file = tmp_path / "multi-scope-query.txt"
    query_file.write_text(query, encoding="utf-8")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DL_MULTI_SCOPE_SECRET", "must-not-be-read")

    def forbidden_boundary(*args, **kwargs):
        raise AssertionError("fixture scope parity must not resolve secrets or use the network")

    monkeypatch.setattr("discourselens.secrets.ReferenceSecretStore.get", forbidden_boundary)
    monkeypatch.setattr("discourselens.providers._open_provider_request", forbidden_boundary)

    core_method = getattr(Project, operation)
    cli_entrypoint = getattr(cli_module, operation)

    with Project.open(project_root) as sdk_project:
        sdk_result = cast(
            RetrievalSearchResult[Any],
            getattr(sdk_project, operation)(
                "multi-scope-preset",
                scope_id,
                [first_artifact, second_artifact],
                query,
                embedding_provider=provider,
                limit=5,
            ),
        )
        assert isinstance(sdk_result, list)
        assert sdk_result.hits == list(sdk_result)
        assert sdk_result.scope_id == scope_id
        assert sdk_result.retrieval_run_id
        sdk_hits = []
        for hit in sdk_result.hits:
            payload = asdict(hit)
            payload.pop("retrieval_run_id", None)
            sdk_hits.append(payload)

    def cli_with_fixture(project, preset, selected_scope, artifact_ids, query_text, **kwargs):
        return cli_entrypoint(
            project,
            preset,
            selected_scope,
            artifact_ids,
            query_text,
            embedding_provider=provider,
            **kwargs,
        )

    with monkeypatch.context() as cli_patch:
        cli_patch.setattr(cli_module, operation, cli_with_fixture)
        cli_result = CliRunner().invoke(
            app,
            [
                "vectors",
                cli_command,
                str(project_root),
                "multi-scope-preset",
                scope_id,
                f"{first_artifact},{second_artifact}",
                "--query-file",
                str(query_file),
                "--limit",
                "5",
            ],
        )
    assert cli_result.exit_code == 0, cli_result.output
    cli_envelope = json.loads(cli_result.output)
    assert isinstance(cli_envelope, dict)
    assert cli_envelope["scope_id"] == scope_id
    assert isinstance(cli_envelope["retrieval_run_id"], str)
    cli_hits = cli_envelope["hits"]

    with Project.open(project_root) as sdk_project:
        raw_sdk_hits = [
            asdict(hit)
            for hit in hybrid_search(
                sdk_project,
                scope_id,
                [first_artifact, second_artifact],
                query,
                (0.2, 0.8),
                limit=5,
            )
        ]
    for hit in raw_sdk_hits:
        hit.pop("retrieval_run_id", None)
    raw_cli_result = CliRunner().invoke(
        app,
        [
            "vectors",
            "hybrid-search",
            str(project_root),
            scope_id,
            f"{first_artifact},{second_artifact}",
            query,
            "--vector-json",
            "[0.2,0.8]",
            "--limit",
            "5",
        ],
    )
    assert raw_cli_result.exit_code == 0, raw_cli_result.output
    raw_cli_payload = json.loads(raw_cli_result.output)
    assert raw_cli_payload["scope_id"] == scope_id
    assert isinstance(raw_cli_payload["retrieval_run_id"], str)
    assert raw_cli_payload["retrieval_run_id"]
    assert raw_cli_payload["hits"] == raw_sdk_hits
    assert {row["revision_id"] for row in raw_sdk_hits} == {first_revision, second_revision}
    with Project.open(project_root, read_only=True) as reopened:
        retrieval_run = reopened.db.execute(
            "SELECT scope_id,returned_count FROM retrieval_runs WHERE id=?",
            (raw_cli_payload["retrieval_run_id"],),
        ).fetchone()
        assert retrieval_run is not None
        assert retrieval_run["scope_id"] == scope_id
        assert retrieval_run["returned_count"] == len(raw_cli_payload["hits"])
        assert set(reopened.scope_revisions(raw_cli_payload["scope_id"])) == {
            first_revision,
            second_revision,
        }

    def mcp_with_fixture(project, preset_name, selected_scope, artifact_ids, query_text, **kwargs):
        return core_method(
            project,
            preset_name,
            selected_scope,
            artifact_ids,
            query_text,
            embedding_provider=provider,
            **kwargs,
        )

    with monkeypatch.context() as mcp_patch:
        mcp_patch.setattr(Project, operation, mcp_with_fixture)
        mcp_module._restore_full_server_registry()
        try:
            mcp_payload = getattr(mcp_module, mcp_tool)(
                scope_id,
                [first_artifact, second_artifact],
                "multi-scope-preset",
                query,
                limit=5,
            )
        finally:
            mcp_module._restore_full_server_registry()

    mcp_hits = mcp_payload["hits"]
    assert sdk_hits and cli_hits and mcp_hits
    assert isinstance(mcp_payload["retrieval_run_id"], str)
    assert mcp_payload["scope_id"] == scope_id
    assert sdk_hits == cli_hits == mcp_hits
    assert {row["revision_id"] for row in sdk_hits} == {first_revision, second_revision}
    assert outside_revision not in {row["revision_id"] for row in sdk_hits}
    assert mcp_payload["scope_id"] == scope_id
    assert mcp_payload["artifact_ids"] == [first_artifact, second_artifact]
    assert provider.query_calls == [("embedding-model", [query])] * 3

    with Project.open(project_root) as sdk_project, pytest.raises(ScopeViolationError):
        getattr(sdk_project, operation)(
            "multi-scope-preset",
            scope_id,
            [outside_artifact],
            query,
            embedding_provider=provider,
            limit=5,
        )
    assert provider.query_calls == [("embedding-model", [query])] * 3


def test_cli_raw_hybrid_revision_shorthand_returns_created_scope(tmp_path):
    project_root = tmp_path / "raw-revision-scope"
    project = Project.init(project_root)
    _, revision = project.import_text("selected.txt", "Apples are green.")
    paragraph_id = str(
        project.db.execute(
            "SELECT id FROM paragraphs WHERE revision_id=?", (revision,)
        ).fetchone()[0]
    )
    artifact_id = build_vector_index(
        project,
        revision,
        [VectorChunk(paragraph_id, 0, len("Apples are green."), (1.0, 0.0))],
    )
    initial_scope_count = int(
        project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    )
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "vectors",
            "hybrid-search",
            str(project_root),
            revision,
            artifact_id,
            "Apples",
            "--vector-json",
            "[1.0,0.0]",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {"scope_id", "retrieval_run_id", "hits"}
    assert isinstance(payload["scope_id"], str)
    assert isinstance(payload["retrieval_run_id"], str)
    assert payload["retrieval_run_id"]
    assert payload["hits"]
    assert {hit["revision_id"] for hit in payload["hits"]} == {revision}

    with Project.open(project_root) as reopened:
        assert reopened.scope_revisions(payload["scope_id"]) == [revision]
        retrieval_run = reopened.db.execute(
            "SELECT scope_id,returned_count FROM retrieval_runs WHERE id=?",
            (payload["retrieval_run_id"],),
        ).fetchone()
        assert retrieval_run is not None
        assert retrieval_run["scope_id"] == payload["scope_id"]
        assert retrieval_run["returned_count"] == len(payload["hits"])
        scope_count = int(
            reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
        )
        assert scope_count == initial_scope_count + 1
