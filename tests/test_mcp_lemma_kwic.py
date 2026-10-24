from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import (
    corpus_lemma_kwic,
    corpus_lemma_kwic_evidence,
    server,
)
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact


def test_mcp_lemma_kwic_schema_requires_explicit_scope_and_artifacts() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    tool = tools["corpus_lemma_kwic"]
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    properties = schema["properties"]

    assert set(schema["required"]) == {"scope_id", "artifact_ids", "query"}
    assert properties["artifact_ids"]["minItems"] == 1
    assert properties["artifact_ids"]["maxItems"] == mcp_module.MCP_LEMMA_KWIC_MAX_ARTIFACTS
    assert properties["artifact_ids"]["uniqueItems"] is True
    assert properties["window"]["minimum"] == 0
    assert properties["window"]["maximum"] == mcp_module.MCP_LEMMA_KWIC_MAX_WINDOW
    assert properties["offset"]["maximum"] == mcp_module.MCP_LEMMA_KWIC_MAX_OFFSET
    assert properties["limit"]["maximum"] == mcp_module.MCP_LEMMA_KWIC_MAX_PAGE_SIZE
    assert properties["query"]["maxLength"] == 256
    assert "corpus_lemma_kwic" in mcp_module.READ_ONLY_MCP_TOOL_NAMES


@pytest.mark.parametrize(
    "arguments",
    [
        {"artifact_ids": ["parsed-1"], "query": "apple"},
        {"scope_id": "scope-1", "artifact_ids": [], "query": "apple"},
        {
            "scope_id": "scope-1",
            "artifact_ids": ["parsed-1"],
            "query": "apple",
            "window": 101,
        },
        {
            "scope_id": "scope-1",
            "artifact_ids": ["parsed-1"],
            "query": "apple",
            "limit": 101,
        },
        {
            "scope_id": "scope-1",
            "artifact_ids": ["parsed-1"],
            "query": "a" * 257,
        },
    ],
)
def test_mcp_lemma_kwic_tool_schema_rejects_invalid_arguments_before_project_open(
    monkeypatch, arguments: dict[str, object]
) -> None:
    def unexpected_project_open():
        pytest.fail("schema-invalid lemma KWIC arguments must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    async def invoke() -> None:
        with pytest.raises(ToolError, match="invalid MCP tool arguments"):
            await server.call_tool("corpus_lemma_kwic", arguments)

    asyncio.run(invoke())


def test_mcp_lemma_kwic_forwards_the_exact_explicit_scope_and_artifacts(monkeypatch) -> None:
    observed: dict[str, Any] = {}

    class StubProject:
        def corpus_lemma_kwic(
            self,
            scope_id: str,
            artifact_ids: list[str],
            query: str,
            *,
            window: int,
            offset: int,
            limit: int,
        ) -> dict[str, object]:
            observed.update(
                scope_id=scope_id,
                artifact_ids=list(artifact_ids),
                query=query,
                window=window,
                offset=offset,
                limit=limit,
            )
            return {
                "scope_id": scope_id,
                "artifact_ids": list(artifact_ids),
                "lemma": query,
                "window": window,
                "offset": offset,
                "limit": limit,
                "total_count": 1,
                "has_more": False,
                "rows": [
                    {
                        "revision_id": "revision-1",
                        "paragraph_id": "paragraph-1",
                        "start_offset": 0,
                        "end_offset": 5,
                        "left": "",
                        "keyword": "Apple",
                        "lemma": query,
                        "right": "",
                    }
                ],
            }

        def close(self) -> None:
            observed["closed"] = True

    stub = StubProject()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)

    result = corpus_lemma_kwic(
        "scope-explicit",
        ["parsed-artifact-explicit"],
        "apple",
        window=3,
        offset=2,
        limit=7,
    )

    assert observed == {
        "scope_id": "scope-explicit",
        "artifact_ids": ["parsed-artifact-explicit"],
        "query": "apple",
        "window": 3,
        "offset": 2,
        "limit": 7,
        "closed": True,
    }
    assert result == {
        "scope_id": "scope-explicit",
        "artifact_ids": ["parsed-artifact-explicit"],
        "lemma": "apple",
        "window": 3,
        "offset": 2,
        "limit": 7,
        "total_count": 1,
        "has_more": False,
        "text_truncated": False,
        "rows": [
            {
                "revision_id": "revision-1",
                "paragraph_id": "paragraph-1",
                "start_offset": 0,
                "end_offset": 5,
                "left": "",
                "keyword": "Apple",
                "lemma": "apple",
                "right": "",
                "text_truncated": False,
            }
        ],
    }


@pytest.mark.parametrize(
    ("scope_id", "artifact_ids", "query", "window", "offset", "limit"),
    [
        ("scope", [], "apple", 5, 0, 20),
        ("scope", ["artifact", "artifact"], "apple", 5, 0, 20),
        ("scope", ["artifact"], " ", 5, 0, 20),
        ("scope", ["artifact"], "apple", 101, 0, 20),
        ("scope", ["artifact"], "apple", 5, -1, 20),
        ("scope", ["artifact"], "apple", 5, 0, 101),
    ],
)
def test_mcp_lemma_kwic_rejects_invalid_bounds_before_opening_project(
    monkeypatch, scope_id, artifact_ids, query, window, offset, limit
) -> None:
    def unexpected_project_open():
        pytest.fail("invalid lemma KWIC arguments must be rejected before project open")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    with pytest.raises(ProjectError):
        corpus_lemma_kwic(scope_id, artifact_ids, query, window, offset, limit)


def test_mcp_lemma_kwic_empty_result_is_json_serializable(tmp_path, monkeypatch) -> None:
    root = tmp_path / "lemma-kwic-empty"
    project = Project.init(root)
    _, revision_id = project.import_text("sample.txt", "Apples grow beside trees.")
    artifact = build_parsed_token_artifact(project, revision_id, _ParsedFixture())
    scope_id = project.snapshot([revision_id])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    result = corpus_lemma_kwic(scope_id, [artifact.artifact_id], "unseen", limit=5)

    assert result["scope_id"] == scope_id
    assert result["artifact_ids"] == [artifact.artifact_id]
    assert result["total_count"] == 0
    assert result["has_more"] is False
    assert result["rows"] == []
    assert json.loads(json.dumps(result)) == result


def test_mcp_lemma_kwic_caps_response_text_in_read_only_mode(tmp_path, monkeypatch) -> None:
    root = tmp_path / "lemma-kwic-bounded-text"
    source_text = "x" * (mcp_module.MCP_KWIC_PAGE_TEXT_MAX_BYTES + 1_024)
    project = Project.init(root)
    _, revision_id = project.import_text("long-token.txt", source_text)
    artifact = build_parsed_token_artifact(project, revision_id, _LongSurfaceFixture())
    scope_id = project.snapshot([revision_id])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    result = corpus_lemma_kwic(scope_id, [artifact.artifact_id], "needle", window=0, limit=1)

    rows = cast(list[dict[str, object]], result["rows"])
    assert result["text_truncated"] is True
    assert len(str(rows[0]["keyword"]).encode("utf-8")) == mcp_module.MCP_KWIC_PAGE_TEXT_MAX_BYTES
    assert rows[0]["text_truncated"] is True
    assert rows[0]["keyword"] == source_text[: mcp_module.MCP_KWIC_PAGE_TEXT_MAX_BYTES]
    assert rows[0]["start_offset"] == 0
    assert rows[0]["end_offset"] == len(source_text)
    assert database.read_bytes() == before


def test_mcp_lemma_kwic_evidence_schema_is_explicit_and_write_only() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    tool = tools["corpus_lemma_kwic_evidence"]
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    properties = schema["properties"]

    assert set(schema["required"]) == {
        "scope_id",
        "artifact_ids",
        "revision_id",
        "lemma",
        "start_offset",
        "end_offset",
    }
    assert properties["artifact_ids"]["minItems"] == 1
    assert properties["artifact_ids"]["maxItems"] == mcp_module.MCP_LEMMA_KWIC_MAX_ARTIFACTS
    assert properties["artifact_ids"]["uniqueItems"] is True
    assert properties["start_offset"]["minimum"] == 0
    assert properties["end_offset"]["minimum"] == 0
    assert properties["lemma"]["maxLength"] == 256
    assert (
        properties["end_offset"]["maximum"]
        == mcp_module.MCP_LEMMA_KWIC_EVIDENCE_MAX_OFFSET
    )
    assert "page" not in properties
    assert "hit_index" not in properties
    assert "corpus_lemma_kwic_evidence" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES


def test_mcp_lemma_kwic_evidence_forwards_exact_occurrence_and_serializes(monkeypatch) -> None:
    observed: dict[str, Any] = {}

    class StubProject:
        def corpus_lemma_kwic_evidence(
            self,
            scope_id: str,
            artifact_ids: list[str],
            revision_id: str,
            lemma: str,
            start_offset: int,
            end_offset: int,
        ) -> dict[str, object]:
            observed.update(
                scope_id=scope_id,
                artifact_ids=list(artifact_ids),
                revision_id=revision_id,
                lemma=lemma,
                start_offset=start_offset,
                end_offset=end_offset,
            )
            return {
                "scope_id": scope_id,
                "revision_id": revision_id,
                "lemma": lemma,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "exact_text": "Apples",
                "artifact_ids": list(artifact_ids),
                "coverage_id": "coverage-1",
                "evidence_id": "evidence-1",
                "citation": "[sample.txt · ¶1]",
            }

        def close(self) -> None:
            observed["closed"] = True

    stub = StubProject()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)

    result = corpus_lemma_kwic_evidence(
        "scope-explicit",
        ["parsed-artifact-explicit"],
        "revision-explicit",
        "apple",
        0,
        6,
    )

    assert observed == {
        "scope_id": "scope-explicit",
        "artifact_ids": ["parsed-artifact-explicit"],
        "revision_id": "revision-explicit",
        "lemma": "apple",
        "start_offset": 0,
        "end_offset": 6,
        "closed": True,
    }
    assert result["evidence_id"] == "evidence-1"
    assert result["citation"] == "[sample.txt · ¶1]"
    assert json.loads(json.dumps(result)) == result


def test_mcp_lemma_kwic_evidence_persists_the_selected_occurrence(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "lemma-kwic-evidence"
    project = Project.init(root)
    _, revision_id = project.import_text("sample.txt", "Apples grow beside trees.")
    artifact = build_parsed_token_artifact(project, revision_id, _ParsedFixture())
    scope_id = project.snapshot([revision_id])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    result = corpus_lemma_kwic_evidence(
        scope_id,
        [artifact.artifact_id],
        revision_id,
        "apple",
        0,
        6,
    )

    assert result["scope_id"] == scope_id
    assert result["artifact_ids"] == [artifact.artifact_id]
    assert result["revision_id"] == revision_id
    assert result["lemma"] == "apple"
    assert result["exact_text"] == "Apples"
    assert result["coverage_mode"] == "EXAMPLE_BASED"
    assert result["evidence_id"]
    assert result["citation"]
    assert json.loads(json.dumps(result)) == result
    with Project.open(root, read_only=True) as reopened:
        evidence = reopened.db.execute(
            "SELECT kind FROM evidence_artifacts WHERE id=?", (result["evidence_id"],)
        ).fetchone()
        assert evidence is not None and evidence["kind"] == "TEXTUAL"


@pytest.mark.parametrize(
    ("scope_id", "artifact_ids", "revision_id", "lemma", "start_offset", "end_offset"),
    [
        ("", ["artifact"], "revision", "apple", 0, 6),
        ("scope", [], "revision", "apple", 0, 6),
        ("scope", ["artifact", "artifact"], "revision", "apple", 0, 6),
        ("scope", ["artifact"], "", "apple", 0, 6),
        ("scope", ["artifact"], "revision", " ", 0, 6),
        ("scope", ["artifact"], "revision", "apple", -1, 6),
        ("scope", ["artifact"], "revision", "apple", 6, 6),
        ("scope", ["artifact"], "revision", "apple", 0, 50_001),
        ("scope", ["artifact"], "revision", "a" * 257, 0, 1),
    ],
)
def test_mcp_lemma_kwic_evidence_rejects_invalid_arguments_before_opening_project(
    monkeypatch, scope_id, artifact_ids, revision_id, lemma, start_offset, end_offset
) -> None:
    def unexpected_project_open():
        pytest.fail("invalid lemma evidence arguments must be rejected before project open")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    with pytest.raises(ProjectError):
        corpus_lemma_kwic_evidence(
            scope_id,
            artifact_ids,
            revision_id,
            lemma,
            start_offset,
            end_offset,
        )


def test_mcp_lemma_kwic_evidence_is_hidden_and_rejected_in_read_only_mode(monkeypatch) -> None:
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    with pytest.raises(ProjectError, match="read-only mode"):
        corpus_lemma_kwic_evidence(
            "scope",
            ["artifact"],
            "revision",
            "apple",
            0,
            6,
        )

    async def invoke() -> None:
        await mcp_module._configure_read_only_server()
        names = {tool.name for tool in await server.list_tools()}
        assert "corpus_lemma_kwic" in names
        assert "corpus_lemma_kwic_evidence" not in names
        with pytest.raises(Exception, match="Unknown tool: corpus_lemma_kwic_evidence"):
            await server.call_tool("corpus_lemma_kwic_evidence", {})

    try:
        asyncio.run(invoke())
    finally:
        mcp_module._restore_full_server_registry()


def test_mcp_lemma_kwic_evidence_schema_rejects_overlong_lemma_before_project_open(
    monkeypatch,
) -> None:
    def unexpected_project_open():
        pytest.fail("schema-invalid lemma evidence must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    async def invoke() -> None:
        with pytest.raises(ToolError, match="invalid MCP tool arguments"):
            await server.call_tool(
                "corpus_lemma_kwic_evidence",
                {
                    "scope_id": "scope",
                    "artifact_ids": ["artifact"],
                    "revision_id": "revision",
                    "lemma": "a" * 257,
                    "start_offset": 0,
                    "end_offset": 1,
                },
            )

    asyncio.run(invoke())


class _ParsedFixture:
    language = "en"
    backend_id = "fixture"
    model_id = "fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        return [ParsedToken("Apples", "apple", "NOUN", None, None, "ROOT", 0, 6)]


class _LongSurfaceFixture:
    language = "en"
    backend_id = "fixture"
    model_id = "fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        return [ParsedToken(text, "needle", "NOUN", None, None, "ROOT", 0, len(text))]
