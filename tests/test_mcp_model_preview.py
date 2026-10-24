from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.models import CredentialReference, ModelPreset


def _preview_project(root) -> None:
    project = Project.init(root)
    try:
        project.save_credential_reference(
            CredentialReference(
                "openai-compatible",
                "mcp-preview-secret-alias",
                "DL_MCP_PREVIEW_SECRET",
                "https://preview.example.test/v1",
            )
        )
        project.save_model_preset(
            ModelPreset(
                "preview-fixture",
                "openai-compatible",
                "mcp-preview-secret-alias",
                "fixture-model",
                parameters={"max_tokens": 64},
            )
        )
    finally:
        project.close()


def _call_preview(arguments: dict[str, object]):
    async def invoke():
        return await mcp_module.server.call_tool("preview_model_request", arguments)

    return asyncio.run(invoke())


def _result_json(result) -> dict[str, object]:
    assert not result.is_error
    assert len(result.content) == 1
    text = result.content[0].text
    value = json.loads(text)
    assert isinstance(value, dict)
    return value


def test_preview_model_request_schema_has_bounded_text_arguments():
    tools = asyncio.run(mcp_module.server.list_tools())
    tool = next(item for item in tools if item.name == "preview_model_request")
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    properties = schema["properties"]

    assert {"preset_name", "prompt", "system_prompt"} <= set(properties)
    assert properties["prompt"]["minLength"] == 1
    assert properties["prompt"]["maxLength"] == mcp_module.MCP_MODEL_PREVIEW_MAX_PROMPT_CHARS

    system_schema = properties["system_prompt"]
    string_variants = [
        system_schema,
        *(variant for variant in system_schema.get("anyOf", []) if isinstance(variant, dict)),
    ]
    bounded_variants = [variant for variant in string_variants if "maxLength" in variant]
    assert bounded_variants
    assert bounded_variants[0]["minLength"] == 1
    assert bounded_variants[0]["maxLength"] == mcp_module.MCP_MODEL_PREVIEW_MAX_SYSTEM_PROMPT_CHARS


@pytest.mark.parametrize(
    ("arguments", "sensitive_marker"),
    [
        (
            {
                "preset_name": "preview-fixture",
                "prompt": "private-prompt-marker "
                + "x" * mcp_module.MCP_MODEL_PREVIEW_MAX_PROMPT_CHARS,
            },
            "private-prompt-marker",
        ),
        (
            {
                "preset_name": "preview-fixture",
                "prompt": "valid request",
                "system_prompt": "private-system-marker "
                + "x" * mcp_module.MCP_MODEL_PREVIEW_MAX_SYSTEM_PROMPT_CHARS,
            },
            "private-system-marker",
        ),
    ],
)
def test_preview_model_request_rejects_oversized_text_without_echo(
    monkeypatch, arguments: dict[str, object], sensitive_marker: str
):
    monkeypatch.delenv("DISCOURSELENS_PROJECT", raising=False)
    with pytest.raises(ToolError) as error:
        _call_preview(arguments)

    assert sensitive_marker not in str(error.value)
    assert "configured length limit" in str(error.value)


def test_preview_model_request_is_secret_free_and_available_read_only(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    _preview_project(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DL_MCP_PREVIEW_SECRET", "never-return-this-secret-value")
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    def reject_network(*args, **kwargs):
        raise AssertionError("request preview must never access the network")

    def reject_secret_resolution(*args, **kwargs):
        raise AssertionError("request preview must never resolve secret values")

    monkeypatch.setattr("discourselens.providers._open_provider_request", reject_network)
    monkeypatch.setattr(
        "discourselens.secrets.EnvironmentSecretStore.get", reject_secret_resolution
    )
    monkeypatch.setattr("discourselens.secrets.KeyringSecretStore.get", reject_secret_resolution)

    database = root / ".discourselens" / "project.sqlite"
    original_database_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    prompt = "private prompt text from corpus"
    system_prompt = "private system instructions"

    async def invoke_read_only():
        await mcp_module._configure_read_only_server()
        names = {tool.name for tool in await mcp_module.server.list_tools()}
        assert "preview_model_request" in names
        assert names == mcp_module.READ_ONLY_MCP_TOOL_NAMES
        return await mcp_module.server.call_tool(
            "preview_model_request",
            {
                "preset_name": "preview-fixture",
                "prompt": prompt,
                "system_prompt": system_prompt,
            },
        )

    try:
        result = _result_json(asyncio.run(invoke_read_only()))
    finally:
        mcp_module._restore_full_server_registry()

    assert result["provider_id"] == "openai-compatible"
    assert result["endpoint"] == "https://preview.example.test/v1"
    assert result["model"] == "fixture-model"
    assert result["allowed"] is True
    assert result["prompt_hash"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert result["prompt_characters"] == len(prompt)
    assert result["system_prompt_hash"] == hashlib.sha256(system_prompt.encode()).hexdigest()
    assert result["system_prompt_characters"] == len(system_prompt)
    assert "credential_alias" not in result
    assert "parameters" not in result
    serialized = json.dumps(result)
    for sensitive_value in (
        prompt,
        system_prompt,
        "never-return-this-secret-value",
        "mcp-preview-secret-alias",
        "DL_MCP_PREVIEW_SECRET",
    ):
        assert sensitive_value not in serialized
    assert hashlib.sha256(database.read_bytes()).hexdigest() == original_database_hash
