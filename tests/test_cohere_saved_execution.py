from __future__ import annotations

import hashlib
import json
from email.message import Message
from typing import Self, cast
from urllib.request import Request

import pytest

from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.execution import execute_model_preset, preview_model_preset_request
from discourselens.models import (
    CredentialReference,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
)
from discourselens.reasoning_policy import COHERE_MINIMUM_FINAL_RESPONSE_TOKENS
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


def _save_cohere_project(
    project: Project,
    *,
    reasoning_mode: str = "manual",
    profile_modes: tuple[str, ...] | None = None,
    reasoning_budget_tokens: int | None = 1024,
    max_tokens: int = 4096,
) -> None:
    project.save_credential_reference(
        CredentialReference("cohere", "selected", "DL_COHERE_SELECTED_KEY")
    )
    project.save_credential_reference(
        CredentialReference("cohere", "other", "DL_COHERE_OTHER_KEY")
    )
    project.save_model_capability_profile(
        ModelCapabilityProfile(
            provider_id="cohere",
            model="fixture-command-model",
            capabilities=ModelCapabilities(
                reasoning_modes=profile_modes or (reasoning_mode,),
                generation_parameters=("max_tokens",),
            ),
            assurance="TESTED",
            source_reference="https://docs.cohere.com/docs/reasoning",
        )
    )
    project.save_model_preset(
        ModelPreset(
            name="cohere-analysis",
            provider_id="cohere",
            credential_alias="selected",
            model="fixture-command-model",
            parameters={"max_tokens": max_tokens},
            reasoning_mode=reasoning_mode,
            reasoning_budget_tokens=reasoning_budget_tokens,
        )
    )


def _cohere_response(completion: str) -> dict[str, object]:
    return {
        "finish_reason": "COMPLETE",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": completion}],
        },
        "usage": {
            "tokens": {"input_tokens": 23, "output_tokens": 9},
            "billed_units": {"input_tokens": 6, "output_tokens": 5},
        },
    }


def test_saved_cohere_preset_executes_with_selected_alias_and_safe_ledger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "cohere-integration-prompt-marker-91f0"
    completion = "cohere-integration-completion-marker-d832"
    selected_secret = "selected-cohere-secret-marker-5cc1"
    other_secret = "other-cohere-secret-must-not-be-used"
    monkeypatch.setenv("DL_COHERE_SELECTED_KEY", selected_secret)
    monkeypatch.setenv("DL_COHERE_OTHER_KEY", other_secret)
    project = Project.init(tmp_path / "cohere-saved-execution")
    _save_cohere_project(project)

    resolved_references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_secret_lookup(store: ReferenceSecretStore, reference: str) -> str:
        resolved_references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_secret_lookup)
    received: list[tuple[Request, dict[str, object]]] = []
    fixture_response = _JSONResponse(_cohere_response(completion))

    def open_request(request: Request, *, timeout: float) -> _JSONResponse:
        assert timeout > 0
        request_body = json.loads(cast(bytes, request.data).decode("utf-8"))
        received.append((request, request_body))
        return fixture_response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    try:
        result = execute_model_preset(project, "cohere-analysis", prompt)

        assert result.response.text == completion
        assert result.response.provider == "cohere"
        assert result.response.model == "fixture-command-model"
        assert result.response.usage is not None
        assert result.response.usage.input_tokens == 23
        assert result.response.usage.output_tokens == 9
        assert resolved_references == ["DL_COHERE_SELECTED_KEY"]
        assert len(received) == 1

        request, body = received[0]
        headers = {name.casefold(): value for name, value in request.header_items()}
        assert request.full_url == "https://api.cohere.com/v2/chat"
        assert headers["authorization"] == f"Bearer {selected_secret}"
        assert body == {
            "model": "fixture-command-model",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "token_budget": 1024},
        }
        assert fixture_response.closed

        ledger = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert ledger is not None
        assert ledger["id"] == result.execution_id
        assert ledger["provider_id"] == "cohere"
        assert ledger["credential_alias"] == "selected"
        assert ledger["model"] == "fixture-command-model"
        assert ledger["reasoning_mode"] == "manual"
        assert ledger["reasoning_budget_tokens"] == 1024
        assert ledger["endpoint"] == "https://api.cohere.com"
        assert ledger["input_tokens"] == 23
        assert ledger["output_tokens"] == 9
        assert ledger["error_type"] is None
        assert ledger["request_hash"] != prompt
        assert ledger["response_hash"] == hashlib.sha256(completion.encode()).hexdigest()

        serialized_ledger = json.dumps(dict(ledger), sort_keys=True)
        for private_value in (
            prompt,
            completion,
            selected_secret,
            other_secret,
            "DL_COHERE_SELECTED_KEY",
            "DL_COHERE_OTHER_KEY",
        ):
            assert private_value not in serialized_ledger
    finally:
        project.close()


@pytest.mark.parametrize("reasoning_mode", ["enabled", "disabled"])
def test_saved_cohere_reasoning_toggles_map_and_require_exact_profile(
    tmp_path, monkeypatch: pytest.MonkeyPatch, reasoning_mode: str
) -> None:
    secret = "selected-cohere-toggle-secret"
    monkeypatch.setenv("DL_COHERE_SELECTED_KEY", secret)
    project = Project.init(tmp_path / f"cohere-{reasoning_mode}-execution")
    _save_cohere_project(
        project,
        reasoning_mode=reasoning_mode,
        reasoning_budget_tokens=None,
    )
    received: list[tuple[Request, dict[str, object]]] = []

    def open_request(request: Request, *, timeout: float) -> _JSONResponse:
        assert timeout > 0
        body = json.loads(cast(bytes, request.data).decode("utf-8"))
        received.append((request, body))
        return _JSONResponse(_cohere_response("toggle answer"))

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    try:
        execution = execute_model_preset(project, "cohere-analysis", "reason carefully")

        assert execution.response.text == "toggle answer"
        assert len(received) == 1
        request, body = received[0]
        assert request.get_header("Authorization") == f"Bearer {secret}"
        assert body["thinking"] == {"type": reasoning_mode}
        row = project.db.execute(
            "SELECT reasoning_mode,reasoning_budget_tokens FROM model_executions WHERE id=?",
            (execution.execution_id,),
        ).fetchone()
        assert row is not None
        assert row["reasoning_mode"] == reasoning_mode
        assert row["reasoning_budget_tokens"] is None
    finally:
        project.close()


def test_saved_cohere_toggle_is_blocked_when_exact_profile_does_not_declare_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.init(tmp_path / "cohere-undeclared-toggle")
    _save_cohere_project(
        project,
        reasoning_mode="enabled",
        profile_modes=("disabled",),
        reasoning_budget_tokens=None,
    )
    transport_calls: list[bool] = []

    def forbidden_request(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("undeclared Cohere reasoning mode must fail before transport")

    monkeypatch.setattr("discourselens.providers._open_provider_request", forbidden_request)
    try:
        with pytest.raises(ProjectError, match="unsupported reasoning mode"):
            execute_model_preset(project, "cohere-analysis", "reason carefully")

        assert transport_calls == []
    finally:
        project.close()


def test_cohere_reasoning_mode_validators_remain_provider_specific(tmp_path) -> None:
    project = Project.init(tmp_path / "cohere-reasoning-validation")
    project.save_credential_reference(
        CredentialReference("cohere", "selected", "DL_COHERE_SELECTED_KEY")
    )
    try:
        with pytest.raises(ProjectError, match="unknown reasoning mode capability"):
            project.save_model_capability_profile(
                ModelCapabilityProfile(
                    "cohere",
                    "fixture-command-model",
                    ModelCapabilities(reasoning_modes=("automatic",)),
                )
            )
        with pytest.raises(ProjectError, match="unsupported for this provider"):
            project.save_model_preset(
                ModelPreset(
                    "cohere-invalid-mode",
                    "cohere",
                    "selected",
                    "fixture-command-model",
                    reasoning_mode="adaptive",
                )
            )
    finally:
        project.close()


def test_saved_cohere_manual_budget_accepts_documented_small_budget(tmp_path) -> None:
    project = Project.init(tmp_path / "cohere-small-budget")
    _save_cohere_project(project, reasoning_budget_tokens=500)
    try:
        preset = project.model_preset("cohere-analysis")
        assert preset["reasoning_budget_tokens"] == 500
    finally:
        project.close()


@pytest.mark.parametrize(
    ("budget", "max_tokens", "message"),
    [
        (3097, 4096, "leave at least 1000"),
        (500, 0, "explicit positive max_tokens"),
    ],
)
def test_saved_cohere_manual_budget_rejects_insufficient_response_room(
    tmp_path, budget: int, max_tokens: int, message: str
) -> None:
    project = Project.init(tmp_path / f"cohere-invalid-budget-{budget}-{max_tokens}")
    project.save_credential_reference(
        CredentialReference("cohere", "selected", "DL_COHERE_SELECTED_KEY")
    )
    try:
        with pytest.raises(ProjectError, match=message):
            project.save_model_preset(
                ModelPreset(
                    name="cohere-analysis",
                    provider_id="cohere",
                    credential_alias="selected",
                    model="fixture-command-model",
                    parameters={"max_tokens": max_tokens},
                    reasoning_mode="manual",
                    reasoning_budget_tokens=budget,
                )
            )
    finally:
        project.close()


def test_tampered_cohere_manual_budget_is_rejected_before_transport_or_secret_lookup(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.init(tmp_path / "cohere-tampered-budget")
    _save_cohere_project(project, reasoning_budget_tokens=500)
    project.db.execute(
        "UPDATE model_presets SET reasoning_budget_tokens=? WHERE name=?",
        (4096 - COHERE_MINIMUM_FINAL_RESPONSE_TOKENS + 1, "cohere-analysis"),
    )
    project.db.commit()
    secret_lookups: list[str] = []
    transport_calls: list[bool] = []

    def unexpected_secret_lookup(_store: ReferenceSecretStore, reference: str) -> str:
        secret_lookups.append(reference)
        pytest.fail("invalid Cohere budget must fail before secret lookup")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("invalid Cohere budget must fail before provider transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    try:
        preview = preview_model_preset_request(project, "cohere-analysis", "reason carefully")
        assert not preview.allowed
        assert "1000 response tokens" in (preview.block_reason or "")
        with pytest.raises(ProjectError, match="leave at least 1000"):
            execute_model_preset(project, "cohere-analysis", "reason carefully")
        assert secret_lookups == []
        assert transport_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


def test_offline_saved_cohere_preset_blocks_before_secret_lookup_or_transport(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.init(tmp_path / "cohere-offline")
    _save_cohere_project(project)
    project.set_offline_mode(True)
    secret_lookups: list[str] = []
    transport_calls: list[bool] = []

    def unexpected_secret_lookup(
        _store: ReferenceSecretStore, reference: str
    ) -> str:
        secret_lookups.append(reference)
        pytest.fail("offline Cohere execution must not resolve a secret")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("offline Cohere execution must not open provider transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    try:
        with pytest.raises(ProjectError, match="offline mode blocks remote model execution"):
            execute_model_preset(project, "cohere-analysis", "offline prompt")

        assert secret_lookups == []
        assert transport_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()
