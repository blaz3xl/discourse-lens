from __future__ import annotations

import base64
import hashlib
import json
from email.message import Message
from typing import Self, cast
from urllib.request import Request

import pytest

from discourselens import Project
from discourselens.errors import ModelProviderError
from discourselens.execution import execute_model_preset
from discourselens.models import (
    CredentialReference,
    GenerationImage,
    GenerationRequest,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelGateway,
    ModelPreset,
)
from discourselens.providers import OpenAICompatibleProvider, OpenAIProvider
from discourselens.secrets import ReferenceSecretStore

_PNG_BYTES = b"\x89PNG\r\n\x1a\nopenai-image-fixture"
_MODEL = "openai-vision-fixture-model"


class _JSONResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode("utf-8")
        self.headers = Message()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def _completion(text: str = "described") -> dict[str, object]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 23, "completion_tokens": 5},
    }


def _save_project(
    project: Project, provider_id: str, *, vision: bool | None
) -> tuple[str, str, str | None]:
    secret_reference = f"DL_{provider_id.upper().replace('-', '_')}_VISION_KEY"
    endpoint = "https://compatible.example/v1" if provider_id != "openai" else None
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
                source_reference="https://platform.openai.com/docs/guides/images-vision",
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
    return secret_reference, "openai-vision-secret", endpoint


def _provider(provider_id: str, *, vision: bool) -> OpenAIProvider | OpenAICompatibleProvider:
    capabilities = {_MODEL: ModelCapabilities(vision=vision)}
    store = ReferenceSecretStore()
    if provider_id == "openai":
        return OpenAIProvider("DL_OPENAI_KEY", store, capabilities_by_model=capabilities)
    return OpenAICompatibleProvider(
        "https://compatible.example/v1",
        "DL_COMPATIBLE_KEY",
        store,
        provider_id=provider_id,
        capabilities_by_model=capabilities,
    )


@pytest.mark.parametrize("provider_id", ["openai", "openai-compatible"])
def test_saved_openai_adapter_sends_local_image_and_records_only_safe_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch, provider_id: str
) -> None:
    image_path = tmp_path / "private-openai-image.png"
    image_path.write_bytes(_PNG_BYTES)
    image = GenerationImage.from_file(image_path)
    prompt = "describe this picture"
    project = Project.init(tmp_path / f"openai-image-{provider_id}")
    secret_reference, secret, endpoint = _save_project(project, provider_id, vision=True)
    monkeypatch.setenv(secret_reference, secret)
    received: list[tuple[Request, dict[str, object]]] = []

    def open_request(request: Request, *, timeout: float) -> _JSONResponse:
        assert timeout > 0
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        received.append((request, payload))
        return _JSONResponse(_completion())

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    try:
        result = execute_model_preset(project, "vision-review", prompt, image=image)

        assert result.response.text == "described"
        assert len(received) == 1
        request, payload = received[0]
        expected_data_uri = f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode('ascii')}"
        assert payload["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": expected_data_uri}},
                ],
            }
        ]
        assert request.full_url == f"{endpoint or 'https://api.openai.com/v1'}/chat/completions"
        assert request.get_header("Authorization") == f"Bearer {secret}"

        row = project.db.execute(
            "SELECT request_hash,capabilities_json FROM model_executions WHERE id=?",
            (result.execution_id,),
        ).fetchone()
        assert row is not None
        metadata = image.metadata()
        request_material = json.dumps(
            {"user_prompt": prompt, "image_input": metadata},
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
    finally:
        project.close()


@pytest.mark.parametrize("provider_id", ["openai", "openai-compatible"])
@pytest.mark.parametrize("declared_vision", [None, False], ids=["absent", "false"])
def test_openai_image_requires_exact_vision_profile_before_secret_or_transport(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
    declared_vision: bool | None,
) -> None:
    root = tmp_path / f"openai-no-vision-{provider_id}-{declared_vision}"
    project = Project.init(root)
    _secret_reference, _secret, _endpoint = _save_project(project, provider_id, vision=declared_vision)
    secret_lookups: list[str] = []
    transport_calls: list[bool] = []

    def unexpected_secret_lookup(
        _store: ReferenceSecretStore, reference: str
    ) -> str:
        secret_lookups.append(reference)
        pytest.fail("denied OpenAI image input must not resolve a secret")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("denied OpenAI image input must not open transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    try:
        with pytest.raises(ModelProviderError, match="exact-model capability profile"):
            execute_model_preset(
                project,
                "vision-review",
                "describe image",
                image=GenerationImage("image/png", _PNG_BYTES),
            )

        assert secret_lookups == []
        assert transport_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


@pytest.mark.parametrize("provider_id", ["openai", "openai-compatible"])
def test_openai_adapter_rejects_unprofiled_image_before_secret_lookup(
    monkeypatch: pytest.MonkeyPatch, provider_id: str
) -> None:
    provider = _provider(provider_id, vision=False)
    monkeypatch.setattr(
        ReferenceSecretStore,
        "get",
        lambda *_args: pytest.fail("missing vision capability must fail before secret lookup"),
    )

    with pytest.raises(ModelProviderError, match="requires vision support"):
        provider.generate(
            GenerationRequest("describe", _MODEL, image=GenerationImage("image/png", _PNG_BYTES))
        )


@pytest.mark.parametrize("provider_id", ["openai", "openai-compatible"])
def test_openai_adapter_rejects_oversized_encoded_image_request_before_secret_or_transport(
    monkeypatch: pytest.MonkeyPatch, provider_id: str
) -> None:
    from discourselens import providers as provider_module

    monkeypatch.setattr(provider_module, "MAX_OPENAI_CHAT_IMAGE_REQUEST_BYTES", 1024)
    monkeypatch.setattr(
        ReferenceSecretStore,
        "get",
        lambda *_args: pytest.fail("oversized image request must fail before secret lookup"),
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: pytest.fail("oversized image request must not open transport"),
    )
    provider = _provider(provider_id, vision=True)

    with pytest.raises(ModelProviderError, match="image request exceeds"):
        provider.generate(
            GenerationRequest(
                "p" * 2048,
                _MODEL,
                image=GenerationImage("image/png", _PNG_BYTES),
            )
        )


@pytest.mark.parametrize("provider_id", ["openai", "openai-compatible"])
def test_openai_image_streaming_is_rejected_before_secret_or_transport(
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
    provider = _provider(provider_id, vision=True)
    request = GenerationRequest("describe", _MODEL, image=GenerationImage("image/png", _PNG_BYTES))

    with pytest.raises(ModelProviderError, match="not supported with streaming"):
        list(provider.stream_generate(request))


def test_openai_image_wire_format_rejects_non_openai_provider_ids() -> None:
    request = GenerationRequest("describe", _MODEL, image=GenerationImage("image/png", _PNG_BYTES))
    provider = OpenAICompatibleProvider(
        "http://localhost/v1",
        None,
        ReferenceSecretStore(),
        provider_id="anthropic",
        capabilities_by_model={_MODEL: ModelCapabilities(vision=True)},
    )

    with pytest.raises(
        ModelProviderError,
        match="unsupported by this provider's Chat Completions mapping",
    ):
        ModelGateway({"anthropic": provider}).generate("anthropic", request)

    with pytest.raises(ModelProviderError, match="unsupported by this provider"):
        provider.generate(request)
