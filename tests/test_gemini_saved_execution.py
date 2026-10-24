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
from discourselens.execution import execute_model_preset, execute_model_preset_stream
from discourselens.models import (
    MAX_GENERATION_IMAGE_BYTES,
    CredentialReference,
    GenerationImage,
    GenerationRequest,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelGateway,
    ModelPreset,
)
from discourselens.providers import GeminiProvider
from discourselens.secrets import ReferenceSecretStore

_PNG_BYTES = b"\x89PNG\r\n\x1a\nimage-fixture-bytes"


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


def _save_gemini_project(project: Project, *, vision: bool | None = False) -> None:
    project.save_credential_reference(
        CredentialReference("gemini", "selected", "DL_GEMINI_SELECTED_KEY")
    )
    project.save_credential_reference(
        CredentialReference("gemini", "other", "DL_GEMINI_OTHER_KEY")
    )
    if vision is not None:
        project.save_model_capability_profile(
            ModelCapabilityProfile(
                provider_id="gemini",
                model="gemini-fixture-model",
                capabilities=ModelCapabilities(
                    system_prompt=True,
                    vision=vision,
                    structured_output=True,
                    generation_parameters=("temperature", "max_output_tokens"),
                ),
                assurance="TESTED",
                source_reference="https://ai.google.dev/api/generate-content",
            )
        )
    project.save_model_preset(
        ModelPreset(
            name="gemini-analysis",
            provider_id="gemini",
            credential_alias="selected",
            model="gemini-fixture-model",
            parameters={"temperature": 0.2, "max_output_tokens": 512},
        )
    )


def _gemini_response(completion: str) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": completion}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 18,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": 0,
            "totalTokenCount": 25,
        },
    }


def test_saved_gemini_preset_routes_json_request_and_records_safe_ledger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "gemini-integration-prompt-marker-3904"
    system_prompt = "gemini-integration-system-marker-d432"
    completion = '{"answer":"gemini-integration-completion-marker-8c12"}'
    selected_secret = "selected-gemini-secret-marker-7e0b"
    other_secret = "other-gemini-secret-must-not-be-used"
    monkeypatch.setenv("DL_GEMINI_SELECTED_KEY", selected_secret)
    monkeypatch.setenv("DL_GEMINI_OTHER_KEY", other_secret)
    project = Project.init(tmp_path / "gemini-saved-execution")
    _save_gemini_project(project)

    resolved_references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_secret_lookup(store: ReferenceSecretStore, reference: str) -> str:
        resolved_references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_secret_lookup)
    received: list[tuple[Request, dict[str, object]]] = []
    fixture_response = _JSONResponse(_gemini_response(completion))

    def open_request(request: Request, *, timeout: float) -> _JSONResponse:
        assert timeout > 0
        request_body = json.loads(cast(bytes, request.data).decode("utf-8"))
        received.append((request, request_body))
        return fixture_response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    try:
        result = execute_model_preset(
            project,
            "gemini-analysis",
            prompt,
            system_prompt=system_prompt,
            response_schema_name="answer-output",
            response_schema=schema,
        )

        assert result.response.text == completion
        assert result.response.provider == "gemini"
        assert result.response.model == "gemini-fixture-model"
        assert result.response.usage is not None
        assert result.response.usage.input_tokens == 18
        assert result.response.usage.output_tokens == 7
        assert resolved_references == ["DL_GEMINI_SELECTED_KEY"]
        assert len(received) == 1

        request, body = received[0]
        headers = {name.casefold(): value for name, value in request.header_items()}
        assert request.full_url == (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-fixture-model:generateContent"
        )
        assert headers["x-goog-api-key"] == selected_secret
        assert body == {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
                "responseFormat": {
                    "text": {"mimeType": "APPLICATION_JSON", "schema": schema}
                },
            },
        }
        assert fixture_response.closed

        ledger = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert ledger is not None
        assert ledger["id"] == result.execution_id
        assert ledger["provider_id"] == "gemini"
        assert ledger["credential_alias"] == "selected"
        assert ledger["model"] == "gemini-fixture-model"
        assert ledger["endpoint"] == "https://generativelanguage.googleapis.com/v1beta"
        assert ledger["input_tokens"] == 18
        assert ledger["output_tokens"] == 7
        assert ledger["error_type"] is None
        assert ledger["request_hash"] != prompt
        assert ledger["response_hash"] == hashlib.sha256(completion.encode()).hexdigest()
        capabilities = json.loads(ledger["capabilities_json"])
        assert capabilities["effective"]["structured_output"] is True

        serialized_ledger = json.dumps(dict(ledger), sort_keys=True)
        for private_value in (
            prompt,
            system_prompt,
            completion,
            selected_secret,
            other_secret,
            "DL_GEMINI_SELECTED_KEY",
            "DL_GEMINI_OTHER_KEY",
        ):
            assert private_value not in serialized_ledger
    finally:
        project.close()


def test_offline_saved_gemini_preset_blocks_before_secret_transport_or_ledger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.init(tmp_path / "gemini-offline")
    _save_gemini_project(project)
    project.set_offline_mode(True)
    secret_lookups: list[str] = []
    transport_calls: list[bool] = []

    def unexpected_secret_lookup(
        _store: ReferenceSecretStore, reference: str
    ) -> str:
        secret_lookups.append(reference)
        pytest.fail("offline Gemini execution must not resolve a secret")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("offline Gemini execution must not open provider transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    try:
        with pytest.raises(ProjectError, match="offline mode blocks remote model execution"):
            execute_model_preset(project, "gemini-analysis", "offline prompt")

        assert secret_lookups == []
        assert transport_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


def test_saved_gemini_preset_sends_bounded_image_and_records_safe_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "describe this local image"
    image = GenerationImage("image/png", _PNG_BYTES)
    path_marker = "never-persist-this-local-image-path.png"
    monkeypatch.setenv("DL_GEMINI_SELECTED_KEY", "vision-gemini-secret")
    project = Project.init(tmp_path / "gemini-vision-allowed")
    _save_gemini_project(project, vision=True)
    received: list[tuple[Request, dict[str, object]]] = []

    def open_request(request: Request, *, timeout: float) -> _JSONResponse:
        assert timeout > 0
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        received.append((request, payload))
        return _JSONResponse(_gemini_response("image described"))

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    try:
        result = execute_model_preset(project, "gemini-analysis", prompt, image=image)

        assert result.response.text == "image described"
        assert len(received) == 1
        request, payload = received[0]
        assert payload["contents"] == [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(_PNG_BYTES).decode("ascii"),
                        }
                    },
                ],
            }
        ]
        assert request.full_url.endswith("/models/gemini-fixture-model:generateContent")

        row = project.db.execute(
            "SELECT request_hash,capabilities_json FROM model_executions WHERE id=?",
            (result.execution_id,),
        ).fetchone()
        assert row is not None
        metadata = image.metadata()
        expected_material = json.dumps(
            {"user_prompt": prompt, "image_input": metadata},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert row["request_hash"] == hashlib.sha256(expected_material).hexdigest()
        capabilities = json.loads(row["capabilities_json"])
        assert capabilities["request_image"] == metadata
        serialized = json.dumps(dict(row), sort_keys=True)
        assert path_marker not in serialized
        assert _PNG_BYTES.decode("latin-1") not in serialized
        assert base64.b64encode(_PNG_BYTES).decode("ascii") not in serialized
    finally:
        project.close()


@pytest.mark.parametrize("declared_vision", [None, False], ids=["absent", "false"])
def test_saved_gemini_image_requires_exact_vision_profile_before_secret_or_transport(
    tmp_path, monkeypatch: pytest.MonkeyPatch, declared_vision: bool | None
) -> None:
    project = Project.init(tmp_path / f"gemini-vision-denied-{declared_vision}")
    _save_gemini_project(project, vision=declared_vision)
    secret_lookups: list[str] = []
    transport_calls: list[bool] = []

    def unexpected_secret_lookup(
        _store: ReferenceSecretStore, reference: str
    ) -> str:
        secret_lookups.append(reference)
        pytest.fail("denied image input must not resolve a secret")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("denied image input must not open provider transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    try:
        with pytest.raises(ModelProviderError, match="exact-model capability profile"):
            execute_model_preset(
                project,
                "gemini-analysis",
                "describe this image",
                image=GenerationImage("image/png", _PNG_BYTES),
            )

        assert secret_lookups == []
        assert transport_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


@pytest.mark.parametrize(
    ("mime_type", "data"),
    [
        ("image/png", b"not-a-png"),
        ("image/jpeg", _PNG_BYTES),
        ("image/gif", b"GIF89a" + b"fixture"),
    ],
    ids=["bad-signature", "mismatched-mime", "unsupported-mime"],
)
def test_generation_image_rejects_bad_or_ambiguous_mime(
    mime_type: str, data: bytes
) -> None:
    with pytest.raises(ModelProviderError, match="image"):
        GenerationImage(mime_type=mime_type, data=data)


def test_generation_image_rejects_oversize_url_and_unsupported_local_path(tmp_path) -> None:
    with pytest.raises(ModelProviderError, match="byte limit"):
        GenerationImage("image/png", b"\x89PNG\r\n\x1a\n" + b"x" * MAX_GENERATION_IMAGE_BYTES)

    with pytest.raises(ModelProviderError, match="URL"):
        GenerationImage.from_file("https://example.test/image.png")

    oversized = tmp_path / "oversized.png"
    oversized.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * MAX_GENERATION_IMAGE_BYTES)
    with pytest.raises(ModelProviderError, match="byte limit"):
        GenerationImage.from_file(oversized)

    unsupported = tmp_path / "image.gif"
    unsupported.write_bytes(b"GIF89a")
    with pytest.raises(ModelProviderError, match="extension"):
        GenerationImage.from_file(unsupported)

    mismatch = tmp_path / "image.jpg"
    mismatch.write_bytes(_PNG_BYTES)
    with pytest.raises(ModelProviderError, match="signature"):
        GenerationImage.from_file(mismatch)


def test_gemini_aggregate_request_limit_blocks_before_secret_and_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from discourselens import providers as provider_module

    monkeypatch.setattr(provider_module, "MAX_GEMINI_INLINE_REQUEST_BYTES", 1024)
    secret_lookups: list[str] = []
    transport_calls: list[bool] = []

    def unexpected_secret_lookup(
        _store: ReferenceSecretStore, reference: str
    ) -> str:
        secret_lookups.append(reference)
        pytest.fail("oversized image request must not resolve a secret")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("oversized image request must not open provider transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    provider = GeminiProvider(
        "fixture-secret",
        ReferenceSecretStore(),
        capabilities_by_model={"fixture-model": ModelCapabilities(vision=True)},
    )

    with pytest.raises(ModelProviderError, match="request exceeds"):
        provider.generate(
            GenerationRequest(
                "p" * 2048,
                "fixture-model",
                image=GenerationImage("image/png", _PNG_BYTES),
            )
        )

    assert secret_lookups == []
    assert transport_calls == []


def test_image_streaming_is_rejected_before_provider_access() -> None:
    image = GenerationImage("image/png", _PNG_BYTES)
    provider = GeminiProvider("fixture-secret", ReferenceSecretStore())
    request = GenerationRequest("describe it", "fixture-model", image=image)

    with pytest.raises(ModelProviderError, match="not supported with streaming"):
        list(ModelGateway({"gemini": provider}).stream("gemini", request))

    with pytest.raises(ModelProviderError, match="not supported with streaming"):
        execute_model_preset_stream(None, "not-opened", "describe it", image=image)
