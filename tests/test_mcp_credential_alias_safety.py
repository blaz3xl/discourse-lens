from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.models import CredentialReference, ModelPreset
from discourselens.secrets import is_credential_like_alias
from discourselens.types import EvidenceRole


def _project_with_unregistered_secret_like_preset_alias(root) -> tuple[str, str]:
    project = Project.init(root)
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible",
            "university",
            "DL_MCP_UNIVERSITY_KEY",
            "https://fixture.example/v1",
        )
    )
    project.save_model_preset(
        ModelPreset("review-fixture", "openai-compatible", "university", "fixture-model")
    )
    _, revision_id = project.import_text("speech.txt", "Evidence supports the account.")
    scope_id = project.snapshot([revision_id])
    run_id = project.create_run(scope_id, {"profile": "fixture"})
    finding_id = project.create_finding(
        run_id,
        "fixture.method",
        "1.0.0",
        "fixture-phenomenon",
        "The account is supported by this wording.",
    )
    span_id = project.create_span(revision_id, 0, len("Evidence supports the account."))
    evidence_id = project.textual_evidence(span_id)
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)

    suspicious_alias = "sk-ant-api03-mcp-unregistered-fixture-secret"
    assert is_credential_like_alias(suspicious_alias)
    project.db.execute(
        "UPDATE model_presets SET credential_alias=? WHERE name=?",
        (suspicious_alias, "review-fixture"),
    )
    project.db.commit()
    project.close()
    return finding_id, suspicious_alias


@pytest.mark.parametrize("tool_name", ["preview_model_request", "review_finding_with_model"])
def test_mcp_preset_alias_errors_do_not_echo_unregistered_secret_like_alias(
    tmp_path, monkeypatch, tool_name: str
):
    root = tmp_path / tool_name
    finding_id, suspicious_alias = _project_with_unregistered_secret_like_preset_alias(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    arguments: dict[str, object]
    if tool_name == "preview_model_request":
        arguments = {"preset_name": "review-fixture", "prompt": "private prompt"}
    else:
        arguments = {"finding_id": finding_id, "preset_name": "review-fixture"}

    async def invoke() -> None:
        with pytest.raises(ToolError) as error:
            await mcp_module.server.call_tool(tool_name, arguments)
        assert suspicious_alias not in str(error.value)

    asyncio.run(invoke())


def test_mcp_analysis_run_metadata_does_not_surface_legacy_credential_like_endpoint_host(
    tmp_path, monkeypatch
):
    root = tmp_path / "legacy-run-endpoint-host"
    project = Project.init(root)
    _, revision_id = project.import_text("speech.txt", "Evidence stays local.")
    scope_id = project.snapshot([revision_id])
    run_id = project.create_run(scope_id, {"profile": "fixture"})
    sentinel = "sk-proj-mcp-run-host-fixture-secret"
    config = {
        "framework_preset_snapshots": {
            "review": {
                "provider_id": "openai-compatible",
                "endpoint": f"https://{sentinel}.example/v1",
            }
        }
    }
    config_json = json.dumps(config, sort_keys=True)
    project.db.execute("DROP TRIGGER analysis_runs_provenance_immutable")
    project.db.execute(
        "UPDATE analysis_runs SET config_json=?,config_hash=? WHERE id=?",
        (config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest(), run_id),
    )
    project.db.commit()
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    async def invoke() -> None:
        with pytest.raises(ToolError) as error:
            await mcp_module.server.call_tool("get_analysis_run", {"analysis_run_id": run_id})
        assert sentinel not in str(error.value)

    asyncio.run(invoke())


def test_mcp_preview_does_not_surface_legacy_credential_like_endpoint_hostname(
    tmp_path, monkeypatch
):
    root = tmp_path / "credential-hostname"
    project = Project.init(root)
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible",
            "university",
            "DL_MCP_UNIVERSITY_KEY",
            "https://provider.example/v1",
        )
    )
    project.save_model_preset(
        ModelPreset("review-fixture", "openai-compatible", "university", "fixture-model")
    )
    sentinel = "sk-proj-mcp-preview-host-fixture-secret"
    project.db.execute(
        "UPDATE provider_credentials SET endpoint=? WHERE provider_id=? AND credential_alias=?",
        (f"https://{sentinel}.example/v1", "openai-compatible", "university"),
    )
    project.db.commit()
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    async def invoke() -> None:
        with pytest.raises(ToolError) as error:
            await mcp_module.server.call_tool(
                "preview_model_request",
                {"preset_name": "review-fixture", "prompt": "private prompt"},
            )
        assert sentinel not in str(error.value)

    asyncio.run(invoke())
