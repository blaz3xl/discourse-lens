from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import TypedDict, cast

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, InputRequiredResult

import discourselens.mcp_server as mcp_module

TOOL_NAME = "build_parsed_token_artifact"


class _BackendSchema(TypedDict):
    enum: list[str]


class _ModelSchema(TypedDict):
    minLength: int
    maxLength: int
    pattern: str


class _ToolSchemaProperties(TypedDict):
    revision_id: dict[str, object]
    backend: _BackendSchema
    model: _ModelSchema


class _ToolSchema(TypedDict):
    properties: _ToolSchemaProperties
    required: list[str]


def _tool_schema() -> _ToolSchema:
    tool = next(
        item for item in asyncio.run(mcp_module.server.list_tools()) if item.name == TOOL_NAME
    )
    return cast(_ToolSchema, tool.model_dump(by_alias=True)["inputSchema"])


def _call_tool(arguments: dict[str, object]) -> CallToolResult | InputRequiredResult:
    async def invoke() -> CallToolResult | InputRequiredResult:
        return await mcp_module.server.call_tool(TOOL_NAME, arguments)

    return asyncio.run(invoke())


def test_parsed_token_tool_schema_bounds_local_model_route() -> None:
    schema = _tool_schema()
    properties = schema["properties"]

    assert set(properties) == {"revision_id", "backend", "model"}
    assert schema["required"] == ["revision_id", "backend", "model"]
    assert properties["backend"]["enum"] == ["spacy", "stanza"]

    model_schema = properties["model"]
    assert model_schema["minLength"] == 1
    assert model_schema["maxLength"] == 128
    assert re.fullmatch(model_schema["pattern"], "en_core_web_sm")
    assert not re.fullmatch(model_schema["pattern"], "../private-model")
    assert not re.fullmatch(model_schema["pattern"], r"models\private-model")
    assert not re.fullmatch(model_schema["pattern"], ".")
    assert not re.fullmatch(model_schema["pattern"], "..")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "unknown-backend"),
        ("model", "m" * 129),
        ("model", "/private/corpus/CREDENTIAL_SENTINEL"),
        ("model", "../private-model"),
        ("model", r"models\private-model"),
        ("model", "."),
        ("model", ".."),
    ],
)
def test_parsed_token_tool_rejects_invalid_route_before_project_open(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    def unexpected_project_open() -> None:
        pytest.fail("schema-invalid parsed-token route must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    arguments: dict[str, object] = {
        "revision_id": "revision-safe",
        "backend": "spacy",
        "model": "en_core_web_sm",
    }
    arguments[field] = value

    with pytest.raises(ToolError, match="invalid MCP tool arguments") as captured:
        _call_tool(arguments)

    assert value not in str(captured.value)
    assert "input_value" not in str(captured.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "unknown-backend"),
        ("model", "m" * 129),
        ("model", "../private-model"),
        ("model", r"models\private-model"),
        ("model", "."),
        ("model", ".."),
    ],
)
def test_direct_parsed_token_call_rejects_invalid_route_before_project_or_loader(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    project_open_calls: list[bool] = []
    loader_calls: list[bool] = []

    def unexpected_project_open() -> None:
        project_open_calls.append(True)
        pytest.fail("invalid direct parsed-token route must not open a project")

    def unexpected_loader(*_args: object) -> None:
        loader_calls.append(True)
        pytest.fail("invalid direct parsed-token route must not call an NLP loader")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    monkeypatch.setattr(mcp_module, "load_spacy", unexpected_loader)
    monkeypatch.setattr(mcp_module, "load_stanza", unexpected_loader)

    backend = cast(
        mcp_module.McpNlpBackend,
        value if field == "backend" else "spacy",
    )
    model = value if field == "model" else "en_core_web_sm"
    with pytest.raises(ToolError):
        mcp_module.build_parsed_token_artifact("revision-safe", backend, model)

    assert project_open_calls == []
    assert loader_calls == []


def test_direct_parsed_token_call_cannot_select_a_model_directory_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_open_calls: list[bool] = []
    loader_calls: list[bool] = []

    def unexpected_project_open() -> None:
        project_open_calls.append(True)
        pytest.fail("direct callers must not select a model-directory environment")

    def unexpected_loader(*_args: object) -> None:
        loader_calls.append(True)
        pytest.fail("direct callers must not select a model-directory environment")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    monkeypatch.setattr(mcp_module, "load_spacy", unexpected_loader)
    monkeypatch.setattr(mcp_module, "load_stanza", unexpected_loader)

    # The public type signature rejects this removed keyword, while the test
    # deliberately checks Python's runtime rejection at the call boundary.
    untyped_build_route = cast(Callable[..., object], mcp_module.build_parsed_token_artifact)
    with pytest.raises(TypeError):
        untyped_build_route(
            "revision-safe",
            "stanza",
            "fixture-en-v1",
            model_dir_env="EVIL_ENV",
        )

    assert project_open_calls == []
    assert loader_calls == []


def test_mcp_extra_model_directory_environment_does_not_override_server_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_model_directory = "/server/configured/stanza-models"
    evil_model_directory = "/caller/selected/stanza-models"
    monkeypatch.setenv(mcp_module.MCP_STANZA_MODEL_DIR_ENV, fixed_model_directory)
    monkeypatch.setenv("EVIL_ENV", evil_model_directory)
    opened: list[bool] = []
    stanza_loads: list[tuple[str, str, str]] = []

    class StubProject:
        def document_revision_summary(self, revision_id: str) -> dict[str, str]:
            return {"document_id": "document-safe"}

        def document_metadata(self, document_id: str) -> dict[str, str]:
            return {"language": "en"}

        def build_parsed_token_artifact(self, revision_id: str, backend: object):
            return SimpleNamespace(
                artifact_id="artifact-safe",
                revision_id=revision_id,
                derivation_hash="derivation-safe",
                content_hash="content-safe",
                coverage={
                    "format_version": 1,
                    "tokenization": {
                        "status": "SUPPORTED",
                        "covered_non_whitespace_characters": 0,
                        "source_non_whitespace_characters": 0,
                        "total_tokens": 0,
                        "limitation": None,
                    },
                    "lemma": {
                        "status": "SUPPORTED",
                        "annotated_tokens": 0,
                        "total_tokens": 0,
                        "limitation": None,
                    },
                    "pos": {
                        "status": "SUPPORTED",
                        "annotated_tokens": 0,
                        "total_tokens": 0,
                        "limitation": None,
                    },
                    "dependency_parse": {
                        "status": "SUPPORTED",
                        "annotated_tokens": 0,
                        "total_tokens": 0,
                        "limitation": None,
                    },
                },
            )

        def close(self) -> None:
            return None

    def open_project() -> StubProject:
        opened.append(True)
        return StubProject()

    def load_stanza(language: str, model: str, model_directory: str):
        stanza_loads.append((language, model, model_directory))
        return SimpleNamespace(language=language, backend_id="stanza", model_id=model)

    monkeypatch.setattr(mcp_module, "_open_project", open_project)
    monkeypatch.setattr(mcp_module, "load_stanza", load_stanza)

    result = _call_tool(
        {
            "revision_id": "revision-safe",
            "backend": "stanza",
            "model": "fixture-en-v1",
            "model_dir_env": "EVIL_ENV",
        }
    )

    assert isinstance(result, CallToolResult)
    assert not result.is_error
    assert opened == [True]
    assert stanza_loads == [("en", "fixture-en-v1", fixed_model_directory)]
