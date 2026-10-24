from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from typing import Self, cast
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from discourselens.errors import AuthenticationError, ModelProviderError, ModelResponseError
from discourselens.execution import _apply_project_capabilities, _default_provider
from discourselens.models import (
    GenerationRequest,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelGateway,
    TokenUsage,
)
from discourselens.providers import CohereProvider


class _Secrets:
    def __init__(self, secret: str = "cohere-secret") -> None:
        self.secret = secret
        self.references: list[str] = []

    def get(self, reference: str) -> str:
        self.references.append(reference)
        return self.secret


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


def _provider(
    *,
    capabilities: dict[str, ModelCapabilities] | None = None,
    endpoint: str = "https://api.cohere.com",
    secrets: _Secrets | None = None,
) -> CohereProvider:
    return CohereProvider(
        "DL_COHERE_FIXTURE_KEY",
        secrets or _Secrets(),
        endpoint,
        capabilities_by_model=capabilities or {},
    )


def _success_payload(
    *, finish_reason: str = "COMPLETE", text: str = "Cohere answer"
) -> dict[str, object]:
    return {
        "finish_reason": finish_reason,
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private chain"},
                {"type": "text", "text": text},
            ],
        },
        "usage": {
            "billed_units": {"input_tokens": 3, "output_tokens": 2},
            "tokens": {"input_tokens": 11, "output_tokens": 6},
        },
    }


def _event(payload: object) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _message_start() -> dict[str, object]:
    return {
        "type": "message-start",
        "delta": {"message": {"role": "assistant"}},
    }


def _content_start(index: int, content_type: str) -> dict[str, object]:
    return {
        "type": "content-start",
        "index": index,
        "delta": {"message": {"content": {"type": content_type, "text": ""}}},
    }


def _content_delta(index: int, text: str) -> dict[str, object]:
    return {
        "type": "content-delta",
        "index": index,
        "delta": {"message": {"content": {"text": text}}},
    }


def _successful_stream() -> bytes:
    return b"".join(
        [
            _message_start_event(),
            _event(_content_start(0, "thinking")),
            _event(_content_delta(0, "private reasoning")),
            _event({"type": "content-end", "index": 0}),
            _event(_content_start(1, "text")),
            _event(_content_delta(1, "Hello ")),
            _event(_content_delta(1, "world")),
            _event({"type": "content-end", "index": 1}),
            _event(
                {
                    "type": "message-end",
                    "delta": {
                        "finish_reason": "COMPLETE",
                        "usage": {"tokens": {"input_tokens": 7, "output_tokens": 2}},
                    },
                }
            ),
        ]
    )


def _message_start_event() -> bytes:
    return _event(_message_start())


def test_cohere_chat_v2_request_response_usage_and_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = _Secrets()
    provider = _provider(
        secrets=secrets,
        capabilities={
            "fixture-model": ModelCapabilities(
                structured_output=True,
                system_prompt=True,
                generation_parameters=(
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "stop_sequences",
                    "frequency_penalty",
                    "presence_penalty",
                    "seed",
                ),
            )
        },
    )
    response_body = json.dumps(_success_payload()).encode("utf-8")
    response = _Response(response_body)
    captured: list[tuple[Request, float]] = []

    def open_request(request: Request, *, timeout: float) -> _Response:
        captured.append((request, timeout))
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    result = provider.generate(
        GenerationRequest(
            "Generate JSON about this prompt",
            "fixture-model",
            system_prompt="Be concise",
            parameters={
                "max_tokens": 256,
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 20,
                "stop_sequences": ["END"],
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "seed": 12,
            },
            response_schema_name="answer",
            response_schema=schema,
        )
    )

    request, timeout = captured[0]
    assert request.full_url == "https://api.cohere.com/v2/chat"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer cohere-secret"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 60.0
    assert json.loads(cast(bytes, request.data).decode("utf-8")) == {
        "model": "fixture-model",
        "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Generate JSON about this prompt"},
        ],
        "stream": False,
        "max_tokens": 256,
        "temperature": 0.2,
        "p": 0.8,
        "k": 20,
        "stop_sequences": ["END"],
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "seed": 12,
        "response_format": {"type": "json_object", "schema": schema},
    }
    assert result.text == "Cohere answer"
    assert result.usage == TokenUsage(input_tokens=11, output_tokens=6)
    assert secrets.references == ["DL_COHERE_FIXTURE_KEY"]
    assert response.closed


def test_cohere_capabilities_fail_closed_without_exact_model_profile() -> None:
    capabilities = _provider().capabilities("unprofiled-model")

    assert capabilities.system_prompt
    assert capabilities.streaming
    assert not capabilities.structured_output
    assert capabilities.reasoning_modes == ()
    assert capabilities.reasoning_efforts == ()
    assert capabilities.generation_parameters == (
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_success_payload(finish_reason="MAX_TOKENS"), "complete text response"),
        (
            {
                **_success_payload(),
                "message": {"role": "user", "content": [{"type": "text", "text": "x"}]},
            },
            "assistant message",
        ),
        (
            {
                **_success_payload(),
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": 4},
                        {"type": "text", "text": "x"},
                    ],
                },
            },
            "malformed thinking content",
        ),
        (
            {
                **_success_payload(),
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                },
            },
            "user-facing text",
        ),
    ],
)
def test_cohere_rejects_incomplete_or_malformed_chat_responses(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object], message: str
) -> None:
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: _Response(json.dumps(payload).encode("utf-8")),
    )

    with pytest.raises(ModelResponseError, match=message):
        _provider().generate(GenerationRequest("prompt", "fixture-model"))


def test_cohere_reasoning_toggle_and_budget_are_profile_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    profile = ModelCapabilityProfile(
        provider_id="cohere",
        model="reasoning-model",
        capabilities=ModelCapabilities(
            reasoning_modes=("manual",),
            generation_parameters=("max_tokens",),
        ),
        source_reference="https://docs.cohere.com/docs/reasoning",
    )
    provider, capabilities, _snapshot = _apply_project_capabilities(
        None, "cohere", "reasoning-model", provider, profile
    )
    assert capabilities.reasoning_modes == ("manual",)

    captured_payload: dict[str, object] | None = None

    def open_request(request: Request, *, timeout: float) -> _Response:
        nonlocal captured_payload
        captured_payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        return _Response(json.dumps(_success_payload()).encode("utf-8"))

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    gateway = ModelGateway({"cohere": provider})
    with pytest.raises(ModelProviderError, match="manual reasoning requires a token budget"):
        gateway.generate(
            "cohere",
            GenerationRequest("reason carefully", "reasoning-model", reasoning_mode="manual"),
        )
    assert captured_payload is None

    gateway.generate(
        "cohere",
        GenerationRequest(
            "reason carefully",
            "reasoning-model",
            reasoning_mode="manual",
            reasoning_budget_tokens=256,
            parameters={"max_tokens": 2048},
        )
    )
    assert captured_payload is not None
    assert captured_payload["thinking"] == {"type": "enabled", "token_budget": 256}

    accepted_payload = captured_payload.copy()
    with pytest.raises(ModelProviderError, match="leave at least 1000"):
        gateway.generate(
            "cohere",
            GenerationRequest(
                "reason carefully",
                "reasoning-model",
                reasoning_mode="manual",
                reasoning_budget_tokens=1025,
                parameters={"max_tokens": 2024},
            ),
        )
    assert captured_payload == accepted_payload


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"max_tokens": 0}, "max_tokens must be a positive integer"),
        ({"max_tokens": True}, "max_tokens must be a positive integer"),
        ({"max_tokens": None}, "must not be null"),
        ({"temperature": -0.1}, "temperature must be non-negative"),
        ({"temperature": float("nan")}, "temperature must be a finite number"),
        ({"top_p": 1.0}, "top_p must be between 0.01 and 0.99"),
        ({"top_k": 501}, "top_k must be an integer between 0 and 500"),
        ({"stop_sequences": ["a", "b", "c", "d", "e", "f"]}, "stop_sequences"),
        ({"seed": -1}, "seed must be a non-negative unsigned 64-bit integer"),
        ({"seed": True}, "seed must be a non-negative unsigned 64-bit integer"),
        ({"topP": 0.8}, "does not declare generation parameter"),
    ],
)
def test_cohere_rejects_unsupported_or_invalid_generation_parameters_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, object],
    message: str,
) -> None:
    called = False

    def forbidden_request(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("invalid Cohere settings must fail before transport")

    monkeypatch.setattr("discourselens.providers._open_provider_request", forbidden_request)
    with pytest.raises(ModelProviderError, match=message):
        _provider().generate(GenerationRequest("prompt", "fixture-model", parameters=parameters))
    assert not called


def test_cohere_rejects_unsupported_reasoning_efforts_and_unprofiled_modes() -> None:
    with pytest.raises(ModelProviderError, match="reasoning effort"):
        _provider().generate(GenerationRequest("prompt", "fixture-model", reasoning="high"))
    with pytest.raises(ModelProviderError, match="reasoning mode"):
        _provider().generate(GenerationRequest("prompt", "fixture-model", reasoning_mode="enabled"))
    with pytest.raises(ModelProviderError, match="reasoning budget requires"):
        _provider().generate(
            GenerationRequest("prompt", "fixture-model", reasoning_budget_tokens=20)
        )


def test_cohere_stream_yields_text_only_and_closes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(_successful_stream(), max_chunk=7)
    captured: list[Request] = []

    def open_request(request: Request, *, timeout: float) -> _Response:
        captured.append(request)
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    provider = _provider()
    request = GenerationRequest("prompt", "fixture-model")

    assert list(ModelGateway({"cohere": provider}).stream("cohere", request)) == [
        "Hello ",
        "world",
    ]
    assert response.closed
    http_request = captured[0]
    assert http_request.full_url == "https://api.cohere.com/v2/chat"
    assert http_request.get_method() == "POST"
    assert http_request.get_header("Accept") == "text/event-stream"
    assert json.loads(cast(bytes, http_request.data).decode("utf-8"))["stream"] is True


@pytest.mark.parametrize(
    "stream_body",
    [
        _message_start_event(),
        _message_start_event() + _event({"type": "tool-call-start", "index": 0, "delta": {}}),
        _message_start_event()
        + _event({"type": "message-end", "delta": {"finish_reason": "MAX_TOKENS"}}),
    ],
    ids=("missing-terminal", "unsupported-tool-event", "incomplete-finish-reason"),
)
def test_cohere_stream_rejects_incomplete_or_non_text_events(
    monkeypatch: pytest.MonkeyPatch, stream_body: bytes
) -> None:
    response = _Response(stream_body)
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    with pytest.raises(ModelResponseError):
        list(_provider().stream_generate(GenerationRequest("prompt", "fixture-model")))
    assert response.closed


def test_cohere_stream_error_event_does_not_expose_provider_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "stream-provider-secret-marker"
    response = _Response(_message_start_event() + _event({"type": "error", "message": marker}))
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ModelResponseError) as captured:
        list(_provider().stream_generate(GenerationRequest("prompt", "fixture-model")))

    assert marker not in str(captured.value)
    assert response.closed


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [(498, AuthenticationError), (401, AuthenticationError), (400, ModelProviderError)],
)
def test_cohere_http_errors_are_classified_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch, status: int, expected_error: type[Exception]
) -> None:
    marker = "provider-error-secret-marker"
    body = json.dumps({"message": marker, "error": {"code": "invalid_request"}}).encode()

    def fail_request(*_args: object, **_kwargs: object) -> None:
        raise HTTPError(
            "https://api.cohere.com/v2/chat",
            status,
            "Request failed",
            Message(),
            BytesIO(body),
        )

    monkeypatch.setattr("discourselens.providers._open_provider_request", fail_request)
    with pytest.raises(expected_error) as captured:
        _provider().generate(GenerationRequest("prompt", "fixture-model"))
    assert marker not in str(captured.value)
    assert captured.value.__suppress_context__


def test_cohere_stream_maps_invalid_token_status_to_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b'{"message":"cohere-secret","error":{"code":"invalid_token"}}'

    def fail_request(*_args: object, **_kwargs: object) -> None:
        raise HTTPError(
            "https://api.cohere.com/v2/chat",
            498,
            "Invalid token",
            Message(),
            BytesIO(body),
        )

    monkeypatch.setattr("discourselens.providers._open_provider_request", fail_request)
    with pytest.raises(AuthenticationError, match="rejected the configured credential"):
        list(_provider().stream_generate(GenerationRequest("prompt", "fixture-model")))


def test_cohere_native_factory_has_default_and_override_endpoint() -> None:
    default_provider = _default_provider(None, "DL_COHERE_TEST_KEY", "cohere")
    custom_provider = _default_provider("https://cohere.example", "DL_COHERE_TEST_KEY", "cohere")

    assert isinstance(default_provider, CohereProvider)
    assert default_provider.endpoint == "https://api.cohere.com"
    assert isinstance(custom_provider, CohereProvider)
    assert custom_provider.endpoint == "https://cohere.example"


def test_cohere_requires_non_empty_credential_reference_and_valid_endpoint() -> None:
    with pytest.raises(ModelProviderError, match="credential reference"):
        CohereProvider(" ", _Secrets())
    with pytest.raises(ModelProviderError, match=r"absolute HTTP\(S\) URL"):
        CohereProvider("fixture", _Secrets(), "file:///tmp/cohere")
