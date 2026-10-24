from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from discourselens import Project, mcp_server
from discourselens.errors import ProjectError
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import PARSED_TOKEN_SCHEMA_VERSION


class _ParsedFixture:
    def __init__(
        self,
        language: str,
        model_id: str = "fixture-en-v1",
        backend_id: str = "spacy",
    ) -> None:
        self.language = language
        self.backend_id = backend_id
        self.model_id = model_id

    def parse(self, text: str) -> list[ParsedToken]:
        if text != "Apples matter.":
            raise AssertionError("unexpected local fixture input")
        return [
            ParsedToken("Apples", "apple", "NOUN", None, None, "ROOT", 0, 6),
            ParsedToken("matter", "matter", "VERB", None, 0, "dep", 7, 13),
            ParsedToken(".", ".", "PUNCT", None, 1, "punct", 13, 14),
        ]


def _call_parsed_tokens(revision_id: str, **route: object):
    async def invoke():
        return await mcp_server.server.call_tool(
            "build_parsed_token_artifact", {"revision_id": revision_id, **route}
        )

    return asyncio.run(invoke())


def _project_with_revision(root, *, language: str | None = "en"):
    project = Project.init(root)
    document_id, revision_id = project.import_text("source.txt", "Apples matter.")
    if language is not None:
        project.set_document_metadata(document_id, "language", language)
    return project, revision_id


def test_mcp_builds_one_explicit_local_parsed_artifact_while_offline(tmp_path, monkeypatch):
    root = tmp_path / "parsed-token-project"
    project, revision_id = _project_with_revision(root)
    project.set_offline_mode(True)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    loads: list[tuple[str, str]] = []

    def load_local_spacy(language: str, model: str):
        loads.append((language, model))
        return _ParsedFixture(language, model)

    monkeypatch.setattr(mcp_server, "load_spacy", load_local_spacy)
    tool = next(
        item
        for item in asyncio.run(mcp_server.server.list_tools())
        if item.name == "build_parsed_token_artifact"
    )
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert schema["required"] == ["revision_id", "backend", "model"]
    assert set(schema["properties"]) == {"revision_id", "backend", "model"}
    output_schema = tool.model_dump(by_alias=True)["outputSchema"]
    assert "coverage" in output_schema["properties"]
    coverage_schema = output_schema["$defs"]["ParsedArtifactCoverage"]
    assert set(coverage_schema["properties"]) == {
        "format_version",
        "tokenization",
        "lemma",
        "pos",
        "dependency_parse",
    }
    assert tool.description is not None
    assert "already-installed local model" in tool.description
    assert "No downloads" in tool.description

    response = _call_parsed_tokens(
        revision_id, backend="spacy", model="fixture-en-v1"
    )
    assert not response.is_error
    artifact = json.loads(response.content[0].text)
    assert set(artifact) == {
        "artifact_id",
        "revision_id",
        "artifact_type",
        "backend",
        "model",
        "schema_version",
        "derivation_hash",
        "content_hash",
        "coverage",
    }
    assert artifact["revision_id"] == revision_id
    assert artifact["artifact_type"] == "parsed_tokens"
    assert artifact["backend"] == "spacy"
    assert artifact["model"] == "fixture-en-v1"
    assert artifact["schema_version"] == PARSED_TOKEN_SCHEMA_VERSION
    assert artifact["coverage"]["tokenization"]["status"] == "SUPPORTED"
    assert artifact["coverage"]["lemma"]["status"] == "SUPPORTED"
    assert artifact["coverage"]["pos"]["status"] == "SUPPORTED"
    assert artifact["coverage"]["dependency_parse"]["status"] == "SUPPORTED"
    assert "relative_path" not in artifact
    assert "Apples matter." not in response.content[0].text
    assert loads == [("en", "fixture-en-v1")]

    with Project.open(root, read_only=True) as stored:
        assert stored.offline_mode
        row = stored.db.execute(
            "SELECT revision_id,artifact_type,backend,model,schema_version,content_hash,file_hash "
            "FROM artifact_registry WHERE id=?",
            (artifact["artifact_id"],),
        ).fetchone()
        assert tuple(row) == (
            revision_id,
            "parsed_tokens",
            "spacy",
            "fixture-en-v1",
            PARSED_TOKEN_SCHEMA_VERSION,
            artifact["derivation_hash"],
            artifact["content_hash"],
        )


def test_mcp_parsed_build_exposes_partial_fields_without_tokens_or_paths(tmp_path, monkeypatch):
    root = tmp_path / "partial-parsed-token-project"
    project, revision_id = _project_with_revision(root)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    class PartialBackend(_ParsedFixture):
        def parse(self, text: str) -> list[ParsedToken]:
            if text != "Apples matter.":
                raise AssertionError("unexpected local fixture input")
            return [
                ParsedToken("Apples", "apple", "", None, None, "ROOT", 0, 6),
                ParsedToken("matter", "matter", "VERB", None, None, None, 7, 13),
                ParsedToken(".", ".", "PUNCT", None, None, None, 13, 14),
            ]

    monkeypatch.setattr(
        mcp_server,
        "load_spacy",
        lambda language, model: PartialBackend(language, model),
    )

    response = _call_parsed_tokens(
        revision_id, backend="spacy", model="fixture-en-v1"
    )

    assert not response.is_error
    artifact = json.loads(response.content[0].text)
    assert artifact["coverage"]["tokenization"]["status"] == "SUPPORTED"
    assert artifact["coverage"]["lemma"]["status"] == "SUPPORTED"
    assert artifact["coverage"]["pos"]["status"] == "LIMITED"
    assert artifact["coverage"]["pos"]["annotated_tokens"] == 2
    assert artifact["coverage"]["pos"]["total_tokens"] == 3
    assert artifact["coverage"]["dependency_parse"]["status"] == "LIMITED"
    assert artifact["coverage"]["dependency_parse"]["annotated_tokens"] == 1
    assert not {"tokens", "token", "token_id", "token_ids", "relative_path", "path"} & set(artifact)
    assert "Apples matter." not in response.content[0].text


def test_mcp_stanza_uses_only_the_fixed_server_model_directory_setting(tmp_path, monkeypatch):
    root = tmp_path / "stanza-project"
    project, revision_id = _project_with_revision(root)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    configured_resources = tmp_path / "private-stanza-resources"
    monkeypatch.setenv(mcp_server.MCP_STANZA_MODEL_DIR_ENV, str(configured_resources))
    loads: list[tuple[str, str, str]] = []

    def load_local_stanza(language: str, model: str, model_directory: str):
        loads.append((language, model, model_directory))
        return _ParsedFixture(language, model, backend_id="stanza")

    monkeypatch.setattr(mcp_server, "load_stanza", load_local_stanza)
    response = _call_parsed_tokens(revision_id, backend="stanza", model="en-fixture")

    assert not response.is_error
    result = json.loads(response.content[0].text)
    assert result["backend"] == "stanza"
    assert result["model"] == "en-fixture"
    assert str(configured_resources) not in response.content[0].text
    assert loads == [("en", "en-fixture", str(configured_resources))]


@pytest.mark.parametrize(
    ("backend", "model"),
    [
        ("spacy", "../../private-model"),
        ("spacy", "."),
        ("spacy", ".."),
        ("spacy", "A" * 129),
        ("stanza", ""),
    ],
)
def test_mcp_rejects_nonportable_model_routes_before_loading(
    tmp_path, monkeypatch, backend, model
):
    root = tmp_path / "invalid-route-project"
    project, revision_id = _project_with_revision(root)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    loads: list[tuple[str, str]] = []
    monkeypatch.setattr(
        mcp_server,
        "load_spacy",
        lambda language, model: loads.append((language, model)),
    )
    monkeypatch.setattr(
        mcp_server,
        "load_stanza",
        lambda language, model, model_dir: loads.append((language, model)),
    )

    with pytest.raises(ToolError):
        mcp_server.build_parsed_token_artifact(revision_id, backend, model)

    assert loads == []
    with Project.open(root, read_only=True) as stored:
        assert stored.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='parsed_tokens'"
        ).fetchone()[0] == 0


def test_mcp_requires_declared_language_before_loading_or_writing(tmp_path, monkeypatch):
    root = tmp_path / "missing-language-project"
    project, revision_id = _project_with_revision(root, language=None)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    loads: list[tuple[str, str]] = []
    monkeypatch.setattr(
        mcp_server,
        "load_spacy",
        lambda language, model: loads.append((language, model)),
    )

    with pytest.raises(ToolError, match="declared document language metadata"):
        mcp_server.build_parsed_token_artifact(
            revision_id, "spacy", "fixture-en-v1"
        )

    assert loads == []
    with Project.open(root, read_only=True) as stored:
        assert stored.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='parsed_tokens'"
        ).fetchone()[0] == 0


def test_mcp_parsed_artifact_tool_is_not_discoverable_or_callable_read_only(
    tmp_path, monkeypatch
):
    root = tmp_path / "parsed-readonly-project"
    project, revision_id = _project_with_revision(root)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", True)
    loads: list[tuple[str, str]] = []
    monkeypatch.setattr(
        mcp_server,
        "load_spacy",
        lambda language, model: loads.append((language, model)),
    )

    try:
        asyncio.run(mcp_server._configure_read_only_server())
        names = {tool.name for tool in asyncio.run(mcp_server.server.list_tools())}
        assert "build_parsed_token_artifact" not in names
        with pytest.raises(ProjectError, match="read-only mode"):
            mcp_server.build_parsed_token_artifact(
                revision_id, "spacy", "fixture-en-v1"
            )
    finally:
        monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", False)
        mcp_server._restore_full_server_registry()

    assert loads == []
    with Project.open(root, read_only=True) as stored:
        assert stored.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='parsed_tokens'"
        ).fetchone()[0] == 0


@pytest.mark.parametrize("failure_stage", ["loader", "parser"])
def test_mcp_errors_withhold_paths_and_text_and_close_project(
    tmp_path, monkeypatch, failure_stage
):
    root = tmp_path / f"parsed-error-project-{failure_stage}"
    project, revision_id = _project_with_revision(root)
    close_calls: list[bool] = []
    original_close = project.close

    def close_spy():
        close_calls.append(True)
        original_close()

    monkeypatch.setattr(project, "close", close_spy)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_open_project", lambda: project)

    if failure_stage == "loader":
        def fail_loader(language: str, model: str):
            raise RuntimeError("secret /private/model directory; source=Apples matter.")

        monkeypatch.setattr(mcp_server, "load_spacy", fail_loader)
    else:
        def load_bad_parser(language: str, model: str):
            class BadParser(_ParsedFixture):
                def parse(self, text: str):
                    raise RuntimeError(f"secret /private/model directory; source={text}")

            return BadParser(language, model)

        monkeypatch.setattr(mcp_server, "load_spacy", load_bad_parser)

    with pytest.raises(ToolError) as error:
        mcp_server.build_parsed_token_artifact(
            revision_id, "spacy", "fixture-en-v1"
        )
    assert "/private/model" not in str(error.value)
    assert "Apples matter." not in str(error.value)
    assert close_calls == [True]
