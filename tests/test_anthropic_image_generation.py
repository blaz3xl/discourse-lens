from __future__ import annotations

import base64
import hashlib
import json
from email.message import Message
from typing import Self, cast
from urllib.request import Request

import pytest

from discourselens import Project
from discourselens.errors import ModelProviderError, ProjectError
from discourselens.execution import execute_model_preset
from discourselens.models import (
    CredentialReference,
    GenerationImage,
    GenerationRequest,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
)
from discourselens.providers import AnthropicCompatibleProvider, AnthropicProvider
from discourselens.secrets import ReferenceSecretStore

_PNG_BYTES = b"\x89PNG\r\n\x1a\nprivate-anthropic-image-fixture"
_MODEL = "claude-vision-fixture-model"
_PROMPT = "Describe the selected figure."


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


def _response(text: str = "A figure.") -> dict[str, object]:
    return {
        "id": "msg_image_fixture",
        "type": "message",
        "role": "assistant",
        "model": _MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 31, "output_tokens": 7},
    }


def _save_project(project: Project, provider_id: str, *, vision: bool | None) -> str:
    secret_reference = f"DL_{provider_id.upper().replace('-', '_')}_VISION_KEY"
    endpoint = "https://anthropic-gateway.example/v1" if provider_id == "anthropic-compatible" else None
    project.save_credential_reference(
        CredentialReference(provider_id, "selected", secret_reference, endpoint)
    )
    if vision is not None:
        project.save_model_capability_profile(
            ModelCapabilityProfile(
                provider_id=provider_id,
                model=_MODEL,
                capabilities=ModelCapabilities(vision=vision),
                assurance="TESTED",
                source_reference="https://docs.anthropic.com/en/docs/build-with-claude/vision",
            )
        )
    project.save_model_preset(
        ModelPreset(
            name="vision-review",
            provider_id=provider_id,
            credential_alias="selected",
            model=_MODEL,
        )
    )
    return secret_reference


@pytest.mark.parametrize("provider_id", ["anthropic", "anthropic-compatible"])
def test_saved_anthropic_adapter_sends_base64_image_and_records_safe_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch, provider_id: str
) -> None:
    image_path = tmp_path / "private-anthropic-image.png"
    image_path.write_bytes(_PNG_BYTES)
    image = GenerationImage.from_file(image_path)
    project = Project.init(tmp_path / f"anthropic-image-{provider_id}")
    secret_reference = _save_project(project, provider_id, vision=True)
    secret = "anthropic-image-secret-fixture"
    monkeypatch.setenv(secret_reference, secret)
    received: list[tuple[Request, dict[str, object]]] = []
    fixture_response = _JSONResponse(_response())

    def open_request(request: Request, *, timeout: float) -> _JSONResponse:
        assert timeout > 0
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        received.append((request, payload))
        return fixture_response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    try:
        result = execute_model_preset(project, "vision-review", _PROMPT, image=image)

        assert result.response.text == "A figure."
        assert result.response.provider == provider_id
        assert len(received) == 1
        request, payload = received[0]
        assert request.full_url == (
            "https://api.anthropic.com/v1/messages"
            if provider_id == "anthropic"
            else "https://anthropic-gateway.example/v1/messages"
        )
        assert request.get_header("X-api-key") == secret
        assert request.get_header("Anthropic-version") == "2023-06-01"
        assert payload["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(_PNG_BYTES).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ]
        assert fixture_response.closed

        row = project.db.execute(
            "SELECT request_hash,capabilities_json FROM model_executions WHERE id=?",
            (result.execution_id,),
        ).fetchone()
        assert row is not None
        metadata = image.metadata()
        request_material = json.dumps(
            {"user_prompt": _PROMPT, "image_input": metadata},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert row["request_hash"] == hashlib.sha256(request_material).hexdigest()
        stored_capabilities = json.loads(row["capabilities_json"])
        assert stored_capabilities["effective"]["vision"] is True
        assert stored_capabilities["request_image"] == metadata
        stored_row = json.dumps(dict(row), sort_keys=True)
        assert str(image_path) not in stored_row
        assert _PNG_BYTES.decode("latin-1") not in stored_row
        assert base64.b64encode(_PNG_BYTES).decode("ascii") not in stored_row
        assert secret not in stored_row
    finally:
        project.close()


@pytest.mark.parametrize("provider_id", ["anthropic", "anthropic-compatible"])
@pytest.mark.parametrize("declared_vision", [None, False], ids=["absent", "false"])
def test_anthropic_image_requires_exact_vision_profile_before_secret_or_transport(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
    declared_vision: bool | None,
) -> None:
    project = Project.init(tmp_path / f"anthropic-no-vision-{provider_id}-{declared_vision}")
    _save_project(project, provider_id, vision=declared_vision)

    def unexpected_secret_lookup(
        _store: ReferenceSecretStore, _reference: str
    ) -> str:
        pytest.fail("denied Anthropic image input must not resolve a secret")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        pytest.fail("denied Anthropic image input must not open transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    try:
        with pytest.raises(ModelProviderError, match="exact-model capability profile"):
            execute_model_preset(
                project,
                "vision-review",
                _PROMPT,
                image=GenerationImage("image/png", _PNG_BYTES),
            )

        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


@pytest.mark.parametrize("provider_id", ["anthropic", "anthropic-compatible"])
def test_anthropic_encoded_image_request_limit_precedes_secret_lookup(
    monkeypatch: pytest.MonkeyPatch, provider_id: str
) -> None:
    from discourselens import providers as provider_module

    monkeypatch.setattr(provider_module, "MAX_ANTHROPIC_MESSAGES_IMAGE_REQUEST_BYTES", 1024)
    monkeypatch.setattr(
        ReferenceSecretStore,
        "get",
        lambda *_args: pytest.fail("oversized image request must fail before secret lookup"),
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: pytest.fail("oversized image request must not open transport"),
    )
    capabilities = {_MODEL: ModelCapabilities(vision=True)}
    secrets = ReferenceSecretStore()
    if provider_id == "anthropic":
        provider = AnthropicProvider("KEY", secrets, capabilities_by_model=capabilities)
    else:
        provider = AnthropicCompatibleProvider(
            "https://anthropic-gateway.example/v1",
            "KEY",
            secrets,
            capabilities_by_model=capabilities,
        )

    with pytest.raises(ModelProviderError, match="image request exceeds"):
        provider.generate(
            GenerationRequest(
                "p" * 2048,
                _MODEL,
                image=GenerationImage("image/png", _PNG_BYTES),
            )
        )


@pytest.mark.parametrize("provider_id", ["anthropic", "anthropic-compatible"])
def test_anthropic_image_streaming_fails_before_secret_or_transport(
    monkeypatch: pytest.MonkeyPatch, provider_id: str
) -> None:
    monkeypatch.setattr(
        ReferenceSecretStore,
        "get",
        lambda *_args: pytest.fail("streaming image input must fail before secret lookup"),
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: pytest.fail("streaming image input must not open transport"),
    )
    capabilities = {_MODEL: ModelCapabilities(vision=True, streaming=True)}
    secrets = ReferenceSecretStore()
    if provider_id == "anthropic":
        provider = AnthropicProvider("KEY", secrets, capabilities_by_model=capabilities)
    else:
        provider = AnthropicCompatibleProvider(
            "https://anthropic-gateway.example/v1",
            "KEY",
            secrets,
            capabilities_by_model=capabilities,
        )
    request = GenerationRequest(_PROMPT, _MODEL, image=GenerationImage("image/png", _PNG_BYTES))

    with pytest.raises(ModelProviderError, match="not supported with streaming"):
        list(provider.stream_generate(request))


def test_anthropic_image_keeps_offline_policy_before_secret_or_transport(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.init(tmp_path / "anthropic-image-offline")
    _save_project(project, "anthropic", vision=True)
    project.set_offline_mode(True)
    monkeypatch.setattr(
        ReferenceSecretStore,
        "get",
        lambda *_args: pytest.fail("offline image request must not resolve a secret"),
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: pytest.fail("offline image request must not open transport"),
    )
    try:
        with pytest.raises(ProjectError, match="offline mode blocks remote model execution"):
            execute_model_preset(
                project,
                "vision-review",
                _PROMPT,
                image=GenerationImage("image/png", _PNG_BYTES),
            )
    finally:
        project.close()


def test_anthropic_unsupported_image_mime_is_rejected() -> None:
    with pytest.raises(ModelProviderError, match="MIME type is unsupported"):
        GenerationImage("image/gif", b"GIF89a-private-image-data")
