from __future__ import annotations

import hashlib
import json
from email.message import Message
from typing import Self, cast
from urllib.request import Request

import pytest

from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.execution import execute_model_preset
from discourselens.models import (
    CredentialReference,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
)
from discourselens.secrets import ReferenceSecretStore


class _JSONResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode("utf-8")
        self.headers = Message()
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def _save_anthropic_project(project: Project) -> None:
    project.save_credential_reference(
        CredentialReference("anthropic", "selected", "DL_ANTHROPIC_SELECTED_KEY")
    )
    project.save_credential_reference(
        CredentialReference("anthropic", "other", "DL_ANTHROPIC_OTHER_KEY")
    )
    project.save_model_capability_profile(
        ModelCapabilityProfile(
            provider_id="anthropic",
            model="claude-fixture-model",
            capabilities=ModelCapabilities(
                system_prompt=True,
                reasoning_efforts=("high",),
                reasoning_modes=("adaptive",),
                generation_parameters=("max_tokens",),
            ),
            assurance="TESTED",
            source_reference=(
                "https://platform.claude.com/docs/en/build-with-claude/extended-thinking"
            ),
        )
    )
    project.save_model_preset(
        ModelPreset(
            name="anthropic-analysis",
            provider_id="anthropic",
            credential_alias="selected",
            model="claude-fixture-model",
            reasoning="high",
            parameters={"max_tokens": 8192},
            reasoning_mode="adaptive",
        )
    )


def _anthropic_response(completion: str) -> dict[str, object]:
    return {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "model": "claude-fixture-model",
        "content": [
            {"type": "thinking", "thinking": "private reasoning block"},
            {"type": "text", "text": completion},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 21, "output_tokens": 13},
    }


def test_saved_anthropic_preset_routes_reasoning_and_records_safe_ledger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "anthropic-integration-prompt-marker-62bf"
    system_prompt = "anthropic-integration-system-marker-c9da"
    completion = "anthropic-integration-completion-marker-116a"
    selected_secret = "selected-anthropic-secret-marker-a941"
    other_secret = "other-anthropic-secret-must-not-be-used"
    monkeypatch.setenv("DL_ANTHROPIC_SELECTED_KEY", selected_secret)
    monkeypatch.setenv("DL_ANTHROPIC_OTHER_KEY", other_secret)
    project = Project.init(tmp_path / "anthropic-saved-execution")
    _save_anthropic_project(project)

    resolved_references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_secret_lookup(store: ReferenceSecretStore, reference: str) -> str:
        resolved_references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_secret_lookup)
    received: list[tuple[Request, dict[str, object]]] = []
    fixture_response = _JSONResponse(_anthropic_response(completion))

    def open_request(request: Request, *, timeout: float) -> _JSONResponse:
        assert timeout > 0
        request_body = json.loads(cast(bytes, request.data).decode("utf-8"))
        received.append((request, request_body))
        return fixture_response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    try:
        result = execute_model_preset(
            project,
            "anthropic-analysis",
            prompt,
            system_prompt=system_prompt,
        )

        assert result.response.text == completion
        assert result.response.provider == "anthropic"
        assert result.response.model == "claude-fixture-model"
        assert result.response.usage is not None
        assert result.response.usage.input_tokens == 21
        assert result.response.usage.output_tokens == 13
        assert resolved_references == ["DL_ANTHROPIC_SELECTED_KEY"]
        assert len(received) == 1

        request, body = received[0]
        headers = {name.casefold(): value for name, value in request.header_items()}
        assert request.full_url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == selected_secret
        assert headers["anthropic-version"] == "2023-06-01"
        assert body == {
            "model": "claude-fixture-model",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
            "system": system_prompt,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        assert fixture_response.closed

        ledger = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert ledger is not None
        assert ledger["id"] == result.execution_id
        assert ledger["provider_id"] == "anthropic"
        assert ledger["credential_alias"] == "selected"
        assert ledger["model"] == "claude-fixture-model"
        assert ledger["reasoning"] == "high"
        assert ledger["reasoning_mode"] == "adaptive"
        assert ledger["reasoning_budget_tokens"] is None
        assert ledger["endpoint"] == "https://api.anthropic.com/v1"
        assert ledger["input_tokens"] == 21
        assert ledger["output_tokens"] == 13
        assert ledger["error_type"] is None
        assert ledger["request_hash"] != prompt
        assert ledger["response_hash"] == hashlib.sha256(completion.encode()).hexdigest()

        serialized_ledger = json.dumps(dict(ledger), sort_keys=True)
        for private_value in (
            prompt,
            system_prompt,
            completion,
            selected_secret,
            other_secret,
            "DL_ANTHROPIC_SELECTED_KEY",
            "DL_ANTHROPIC_OTHER_KEY",
        ):
            assert private_value not in serialized_ledger
    finally:
        project.close()


def test_offline_saved_anthropic_preset_blocks_before_secret_or_transport(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.init(tmp_path / "anthropic-offline")
    _save_anthropic_project(project)
    project.set_offline_mode(True)
    secret_lookups: list[str] = []
    transport_calls: list[bool] = []

    def unexpected_secret_lookup(
        _store: ReferenceSecretStore, reference: str
    ) -> str:
        secret_lookups.append(reference)
        pytest.fail("offline Anthropic execution must not resolve a secret")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("offline Anthropic execution must not open provider transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    try:
        with pytest.raises(ProjectError, match="offline mode blocks remote model execution"):
            execute_model_preset(project, "anthropic-analysis", "offline prompt")

        assert secret_lookups == []
        assert transport_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()
