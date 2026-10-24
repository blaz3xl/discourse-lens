from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from typing import Self
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from discourselens import Project
from discourselens.errors import ModelProviderError, ModelResponseError
from discourselens.execution import execute_model_preset_stream
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelGateway,
    ModelPreset,
)
from discourselens.providers import AnthropicCompatibleProvider, AnthropicProvider


class _Secrets:
    def get(self, reference: str) -> str:
        assert reference == "fixture"
        return "anthropic-stream-secret"


class _PresetSecrets:
    def get(self, reference: str) -> str:
        assert reference == "DL_ANTHROPIC_STREAM_FIXTURE_KEY"
        return "preset-secret"


class _Response:
    def __init__(self, body: bytes, *, max_chunk: int = 4096) -> None:
        self.body = body
        self.offset = 0
        self.max_chunk = max_chunk
        self.closed = False
        self.headers = Message()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def read1(self, limit: int = -1) -> bytes:
        if self.offset >= len(self.body):
            return b""
        size = self.max_chunk if limit < 0 else min(limit, self.max_chunk)
        result = self.body[self.offset : self.offset + size]
        self.offset += len(result)
        return result

    def read(self, limit: int = -1) -> bytes:
        if self.offset >= len(self.body):
            return b""
        size = len(self.body) - self.offset if limit < 0 else limit
        result = self.body[self.offset : self.offset + size]
        self.offset += len(result)
        return result


def _event(payload: object, *, name: str | None = None) -> bytes:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    prefix = f"event: {name}\n".encode("ascii") if name is not None else b""
    return prefix + b"data: " + encoded + b"\n\n"


def _message_start() -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": "fixture-model",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 0},
        },
    }


def _delta(delta: dict[str, object]) -> dict[str, object]:
    return {"type": "content_block_delta", "index": 0, "delta": delta}


def _message_stop() -> dict[str, object]:
    return {"type": "message_stop"}


def _provider(
    provider_kind: str,
    *,
    streaming: bool = True,
    model_capabilities: ModelCapabilities | None = None,
):
    capabilities = {
        "fixture-model": model_capabilities
        or ModelCapabilities(streaming=streaming, system_prompt=True)
    }
    secrets = _Secrets()
    if provider_kind == "anthropic":
        return AnthropicProvider("fixture", secrets, capabilities_by_model=capabilities)
    if provider_kind == "anthropic-compatible":
        return AnthropicCompatibleProvider(
            "https://gateway.example/v1",
            "fixture",
            secrets,
            capabilities_by_model=capabilities,
        )
    raise AssertionError(f"unknown provider kind: {provider_kind}")


_PROVIDER_KINDS = ("anthropic", "anthropic-compatible")


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_anthropic_stream_yields_only_text_deltas_and_sends_messages_stream_request(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    response = _Response(
        b"".join(
            [
                b": keepalive\n\n",
                _event(_message_start(), name="message_start"),
                _event({"type": "ping"}, name="ping"),
                _event(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                ),
                _event(_delta({"type": "thinking_delta", "thinking": "private-reasoning"})),
                _event(_delta({"type": "signature_delta", "signature": "private-signature"})),
                _event(
                    _delta(
                        {
                            "type": "input_json_delta",
                            "partial_json": '{"tool":"not-user-text"}',
                        }
                    )
                ),
                _event({"type": "future_event", "opaque": "ignored"}),
                _event(_delta({"type": "text_delta", "text": "Hola "})),
                _event(_delta({"type": "text_delta", "text": "mundo."})),
                _event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
                _event(_message_stop(), name="message_stop"),
            ]
        ),
        max_chunk=7,
    )
    captured: list[tuple[Request, float]] = []

    def open_request(request: Request, *, timeout: float):
        captured.append((request, timeout))
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    provider = _provider(provider_kind)
    gateway = ModelGateway({provider.provider_id: provider})
    request = GenerationRequest("prompt", "fixture-model", system_prompt="system")

    assert list(gateway.stream(provider.provider_id, request)) == ["Hola ", "mundo."]
    assert response.closed

    http_request, _timeout = captured[0]
    expected_endpoint = (
        "https://api.anthropic.com/v1/messages"
        if provider_kind == "anthropic"
        else "https://gateway.example/v1/messages"
    )
    assert http_request.full_url == expected_endpoint
    assert http_request.get_method() == "POST"
    headers = {name.lower(): value for name, value in http_request.header_items()}
    assert headers["accept"] == "text/event-stream"
    assert headers["content-type"] == "application/json"
    assert headers["x-api-key"] == "anthropic-stream-secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert isinstance(http_request.data, (bytes, bytearray, str))
    payload = json.loads(http_request.data)
    assert payload["stream"] is True
    assert payload["max_tokens"] == 4096
    assert payload["system"] == "system"
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
@pytest.mark.parametrize("reasoning_mode", ["adaptive", "manual"])
def test_anthropic_reasoning_stream_maps_config_and_yields_only_text(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str, reasoning_mode: str
) -> None:
    response = _Response(
        _event(_message_start(), name="message_start")
        + _event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }
        )
        + _event(_delta({"type": "thinking_delta", "thinking": "private reasoning"}))
        + _event(_delta({"type": "signature_delta", "signature": "private signature"}))
        + _event({"type": "content_block_stop", "index": 0})
        + _event(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            }
        )
        + _event(_delta({"type": "text_delta", "text": "public answer"}))
        + _event({"type": "content_block_stop", "index": 1})
        + _event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
        + _event(_message_stop(), name="message_stop")
    )
    captured: list[Request] = []

    def open_request(request: Request, *, timeout: float):
        assert timeout > 0
        captured.append(request)
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    capabilities = ModelCapabilities(
        streaming=True,
        system_prompt=True,
        reasoning_efforts=("high",),
        reasoning_modes=("adaptive", "manual"),
        generation_parameters=("max_tokens", "temperature"),
    )
    provider = _provider(provider_kind, model_capabilities=capabilities)
    request = GenerationRequest(
        "prompt",
        "fixture-model",
        reasoning="high",
        parameters={"max_tokens": 4096, "temperature": 1},
        reasoning_mode=reasoning_mode,
        reasoning_budget_tokens=2048 if reasoning_mode == "manual" else None,
    )

    assert list(ModelGateway({provider.provider_id: provider}).stream(provider.provider_id, request)) == [
        "public answer"
    ]
    assert response.closed
    assert len(captured) == 1
    data = captured[0].data
    assert isinstance(data, (bytes, bytearray, str))
    payload = json.loads(data)
    expected_thinking = (
        {"type": "adaptive"}
        if reasoning_mode == "adaptive"
        else {"type": "enabled", "budget_tokens": 2048}
    )
    assert payload["thinking"] == expected_thinking
    assert payload["output_config"] == {"effort": "high"}
    assert payload["stream"] is True
    assert payload["max_tokens"] == 4096
    assert payload["temperature"] == 1


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_anthropic_stream_rejects_undeclared_reasoning_before_secret_or_transport(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reasoning preflight must happen before credential or transport access")

    monkeypatch.setattr("discourselens.providers._resolve_provider_credential", fail_if_called)
    monkeypatch.setattr("discourselens.providers._open_provider_request", fail_if_called)
    provider = _provider(provider_kind)
    request = GenerationRequest(
        "prompt", "fixture-model", reasoning_mode="adaptive"
    )

    with pytest.raises(ModelProviderError, match="reasoning mode"):
        list(ModelGateway({provider.provider_id: provider}).stream(provider.provider_id, request))


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
@pytest.mark.parametrize(
    ("reasoning_budget_tokens", "temperature", "max_tokens", "message"),
    [
        (1023, 1, 4096, "at least 1024 tokens"),
        (2048, 0, 4096, "temperature 1"),
        (2048, 1, 2048, "less than max_tokens"),
    ],
)
def test_anthropic_stream_rejects_invalid_manual_reasoning_before_secret_or_transport(
    monkeypatch: pytest.MonkeyPatch,
    provider_kind: str,
    reasoning_budget_tokens: int,
    temperature: int,
    max_tokens: int,
    message: str,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reasoning preflight must happen before credential or transport access")

    monkeypatch.setattr("discourselens.providers._resolve_provider_credential", fail_if_called)
    monkeypatch.setattr("discourselens.providers._open_provider_request", fail_if_called)
    capabilities = ModelCapabilities(
        streaming=True,
        system_prompt=True,
        reasoning_efforts=("high",),
        reasoning_modes=("manual",),
        generation_parameters=("max_tokens", "temperature"),
    )
    provider = _provider(provider_kind, model_capabilities=capabilities)
    request = GenerationRequest(
        "prompt",
        "fixture-model",
        reasoning="high",
        parameters={"max_tokens": max_tokens, "temperature": temperature},
        reasoning_mode="manual",
        reasoning_budget_tokens=reasoning_budget_tokens,
    )

    with pytest.raises(ModelProviderError, match=message):
        list(ModelGateway({provider.provider_id: provider}).stream(provider.provider_id, request))


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
@pytest.mark.parametrize(
    ("body", "message", "remote_text"),
    [
        (b"data: {malformed remote-secret}\n\n", "malformed JSON", "remote-secret"),
        (
            _event(
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "remote-secret"},
                }
            ),
            "streaming error event",
            "remote-secret",
        ),
        (_event(_message_stop()), "before message_start", None),
        (_event(_message_start()), "without message_stop", None),
        (
            _event(_delta({"type": "text_delta", "text": "remote-secret"})),
            "before message_start",
            "remote-secret",
        ),
        (
            _event({"type": "message_start", "message": "remote-secret"}),
            "malformed",
            "remote-secret",
        ),
    ],
    ids=[
        "malformed-json",
        "remote-error",
        "stop-without-start",
        "missing-stop",
        "delta-before-start",
        "malformed-start",
    ],
)
def test_anthropic_stream_rejects_malformed_or_incomplete_events_safely(
    monkeypatch: pytest.MonkeyPatch,
    provider_kind: str,
    body: bytes,
    message: str,
    remote_text: str | None,
) -> None:
    response = _Response(body)
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    provider = _provider(provider_kind)
    stream = ModelGateway({provider.provider_id: provider}).stream(
        provider.provider_id, GenerationRequest("prompt", "fixture-model")
    )

    with pytest.raises(ModelResponseError, match=message) as captured:
        list(stream)
    if remote_text is not None:
        assert remote_text not in str(captured.value)
    assert response.closed


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_anthropic_max_tokens_stop_rejects_partial_text_and_closes_response(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    response = _Response(
        _event(_message_start(), name="message_start")
        + _event(_delta({"type": "text_delta", "text": "partial-secret"}))
        + _event({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}})
        + _event(_message_stop(), name="message_stop")
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    provider = _provider(provider_kind)
    stream = ModelGateway({provider.provider_id: provider}).stream(
        provider.provider_id, GenerationRequest("prompt", "fixture-model")
    )

    assert next(stream) == "partial-secret"
    with pytest.raises(ModelResponseError, match="without a complete text response") as caught:
        next(stream)
    assert "partial-secret" not in str(caught.value)
    assert response.closed


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_closing_anthropic_stream_closes_http_response(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    response = _Response(
        _event(_message_start())
        + _event(_delta({"type": "text_delta", "text": "first"}))
        + _event(_delta({"type": "text_delta", "text": "second"}))
        + _event(_message_stop())
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    provider = _provider(provider_kind)
    stream = ModelGateway({provider.provider_id: provider}).stream(
        provider.provider_id, GenerationRequest("prompt", "fixture-model")
    )

    assert next(stream) == "first"
    stream.close()
    assert response.closed


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_anthropic_buffered_generation_remains_non_streaming(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    response = _Response(
        json.dumps(
            {"content": [{"type": "text", "text": "buffered"}], "stop_reason": "end_turn"}
        ).encode()
    )
    captured: list[Request] = []

    def open_request(request: Request, *, timeout: float):
        captured.append(request)
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    provider = _provider(provider_kind)

    result = provider.generate(GenerationRequest("prompt", "fixture-model"))

    assert isinstance(result, GenerationResponse)
    assert result.text == "buffered"
    assert captured
    data = captured[0].data
    assert isinstance(data, (bytes, bytearray, str))
    payload = json.loads(data)
    assert "stream" not in payload
    assert {name.lower(): value for name, value in captured[0].header_items()}[
        "accept"
    ] == "application/json"
    assert response.closed


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_anthropic_http_error_does_not_echo_body_and_closes_error_response(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    remote_secret = "anthropic-http-stream-secret"
    error = HTTPError(
        "https://provider.example/messages",
        500,
        "Server Error",
        Message(),
        BytesIO(json.dumps({"error": {"message": remote_secret}}).encode()),
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("discourselens.providers._open_provider_request", fail)
    provider = _provider(provider_kind)
    stream = ModelGateway({provider.provider_id: provider}).stream(
        provider.provider_id, GenerationRequest("prompt", "fixture-model")
    )

    with pytest.raises(ModelProviderError) as captured:
        next(stream)
    assert remote_secret not in str(captured.value)
    assert error.fp is not None and error.fp.closed


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_anthropic_stream_requires_exact_model_streaming_capability(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("transport must not be opened")

    monkeypatch.setattr("discourselens.providers._open_provider_request", fail_if_called)
    provider = _provider(provider_kind, streaming=False)

    with pytest.raises(ModelProviderError, match="does not declare streaming support"):
        list(
            ModelGateway({provider.provider_id: provider}).stream(
                provider.provider_id, GenerationRequest("prompt", "fixture-model")
            )
        )
    assert not called


def _preset_project(path, *, declared_streaming: bool | None) -> Project:
    project = Project.init(path)
    project.save_credential_reference(
        CredentialReference(
            "anthropic-compatible",
            "research",
            "DL_ANTHROPIC_STREAM_FIXTURE_KEY",
            "https://gateway.example/v1",
        )
    )
    project.save_model_preset(
        ModelPreset("generate", "anthropic-compatible", "research", "fixture-model")
    )
    if declared_streaming is not None:
        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "anthropic-compatible",
                "fixture-model",
                ModelCapabilities(streaming=declared_streaming, system_prompt=True),
            )
        )
    return project


def test_anthropic_preset_stream_passes_exact_model_capability_to_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("DL_ANTHROPIC_STREAM_FIXTURE_KEY", "preset-secret")
    response = _Response(
        _event(_message_start())
        + _event(_delta({"type": "text_delta", "text": "preset-text"}))
        + _event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
        + _event(_message_stop())
    )
    captured: list[Request] = []

    def open_request(request: Request, *, timeout: float):
        captured.append(request)
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    project = _preset_project(tmp_path / "anthropic-preset-stream", declared_streaming=True)

    def provider_factory(endpoint: str | None, reference: str | None, provider_id: str):
        assert endpoint == "https://gateway.example/v1"
        assert reference == "DL_ANTHROPIC_STREAM_FIXTURE_KEY"
        assert provider_id == "anthropic-compatible"
        return AnthropicCompatibleProvider(
            endpoint,
            reference,
            _PresetSecrets(),
            provider_id=provider_id,
        )

    try:
        stream = execute_model_preset_stream(
            project, "generate", "prompt", provider_factory=provider_factory
        )
        assert list(stream) == ["preset-text"]
        assert captured
        assert response.closed
    finally:
        project.close()


@pytest.mark.parametrize("declared_streaming", [None, False], ids=["missing", "false"])
def test_anthropic_preset_stream_rejects_missing_or_false_exact_capability_before_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path, declared_streaming: bool | None
) -> None:
    project = _preset_project(
        tmp_path / f"anthropic-preset-no-stream-{declared_streaming}",
        declared_streaming=declared_streaming,
    )
    opened = False
    constructed = False

    def open_request(*_args: object, **_kwargs: object):
        nonlocal opened
        opened = True
        raise AssertionError("stream transport must not be opened")

    def provider_factory(*_args: object, **_kwargs: object):
        nonlocal constructed
        constructed = True
        raise AssertionError("adapter must not be constructed")

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    try:
        stream = execute_model_preset_stream(
            project, "generate", "prompt", provider_factory=provider_factory
        )
        with pytest.raises(ModelProviderError, match="exact-model capability profile"):
            next(stream)
        assert not constructed
        assert not opened
    finally:
        project.close()
