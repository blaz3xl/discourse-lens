from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from typing import Self
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from discourselens.errors import ModelProviderError, ModelResponseError
from discourselens.models import (
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelGateway,
)
from discourselens.providers import OpenAICompatibleProvider


class _Secrets:
    def get(self, reference: str) -> str:
        assert reference == "fixture"
        return "fixture-secret"


class _Response:
    def __init__(self, lines: list[bytes], *, max_chunk: int = 4096) -> None:
        self.body = b"".join(lines)
        self.offset = 0
        self.max_chunk = max_chunk
        self.closed = False
        self.headers = Message()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def read1(self, limit: int = -1) -> bytes:
        if self.offset >= len(self.body):
            return b""
        size = self.max_chunk if limit < 0 else min(limit, self.max_chunk)
        result = self.body[self.offset : self.offset + size]
        self.offset += len(result)
        return result

    def read(self, limit: int = -1) -> bytes:
        remaining = self.body[self.offset :]
        result = remaining if limit < 0 else remaining[:limit]
        self.offset += len(result)
        return result


def _event(payload: object, *, newline: bytes = b"\n") -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + newline + newline


def _provider(*, streaming: bool = True) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "https://provider.example/v1",
        "fixture",
        _Secrets(),
        capabilities_by_model={
            "fixture-model": ModelCapabilities(streaming=streaming, system_prompt=True)
        },
    )


def test_gateway_streams_text_deltas_and_uses_chat_completions_sse(monkeypatch):
    response = _Response(
        [
            b": keepalive\n\n",
            _event({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
            _event({"choices": [{"delta": {"content": "Hola "}, "finish_reason": None}]}),
            _event({"choices": [{"delta": {"content": "mundo."}, "finish_reason": None}]}),
            _event({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            b"data: [DONE]\n\n",
        ]
    )
    captured_requests: list[Request] = []

    def open_request(request: Request, *, timeout: float):
        captured_requests.append(request)
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    gateway = ModelGateway({"openai-compatible": _provider()})
    request = GenerationRequest("prompt", "fixture-model", system_prompt="system")

    assert list(gateway.stream("openai-compatible", request)) == ["Hola ", "mundo."]
    http_request = captured_requests[0]
    assert http_request.full_url == "https://provider.example/v1/chat/completions"
    assert http_request.get_method() == "POST"
    assert http_request.get_header("Accept") == "text/event-stream"
    assert http_request.get_header("Authorization") == "Bearer fixture-secret"
    assert isinstance(http_request.data, (bytes, bytearray, str))
    payload = json.loads(http_request.data)
    assert payload["stream"] is True
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert response.closed


@pytest.mark.parametrize("newline", [b"\r", b"\n", b"\r\n"], ids=["cr", "lf", "crlf"])
def test_sse_accepts_all_standard_line_endings_across_read_boundaries(
    monkeypatch, newline: bytes
) -> None:
    response = _Response(
        [
            _event(
                {"choices": [{"delta": {"content": "line-ending-ok"}, "finish_reason": None}]},
                newline=newline,
            ),
            _event({"choices": [{"delta": {}, "finish_reason": "stop"}]}, newline=newline),
            b"data: [DONE]" + newline + newline,
        ],
        max_chunk=2,
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )

    chunks = ModelGateway({"openai-compatible": _provider()}).stream(
        "openai-compatible", GenerationRequest("prompt", "fixture-model")
    )
    assert list(chunks) == ["line-ending-ok"]
    assert response.closed


def test_buffered_generation_remains_buffered_and_does_not_send_stream_flag(monkeypatch):
    response = _Response(
        [b'{"choices":[{"message":{"content":"buffered"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1}}']
    )
    captured: list[Request] = []

    def open_request(request: Request, *, timeout: float):
        captured.append(request)
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    provider = _provider()
    result = provider.generate(GenerationRequest("prompt", "fixture-model"))

    assert isinstance(result, GenerationResponse)
    assert result.text == "buffered"
    assert isinstance(captured[0].data, (bytes, bytearray, str))
    assert "stream" not in json.loads(captured[0].data)
    assert response.closed


def test_stream_requires_explicit_model_capability_before_transport(monkeypatch):
    called = False

    def open_request(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("transport must not be called")

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    gateway = ModelGateway({"openai-compatible": _provider(streaming=False)})

    with pytest.raises(ModelProviderError, match="does not declare streaming support"):
        gateway.stream("openai-compatible", GenerationRequest("prompt", "fixture-model"))
    assert not called


@pytest.mark.parametrize(
    ("lines", "message", "secret"),
    [
        ([b"data: {bad json}\n\n"], "malformed JSON", None),
        (
            [_event({"error": {"message": "stream-secret-sentinel"}})],
            "streaming error event",
            "stream-secret-sentinel",
        ),
        ([b"data: [DONE]\n\n"], "before a finish reason", None),
        (
            [_event({"choices": [{"delta": {}, "finish_reason": "stop"}]})],
            "without a completion marker",
            None,
        ),
        (
            [_event({"choices": [{"delta": {"content": []}, "finish_reason": None}]})],
            "delta content must be text",
            None,
        ),
    ],
)
def test_malformed_or_incomplete_sse_is_a_sanitized_domain_error(
    monkeypatch, lines: list[bytes], message: str, secret: str | None
) -> None:
    response = _Response(lines)
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    iterator = ModelGateway({"openai-compatible": _provider()}).stream(
        "openai-compatible", GenerationRequest("prompt", "fixture-model")
    )

    with pytest.raises(ModelResponseError, match=message) as captured:
        list(iterator)
    if secret is not None:
        assert secret not in str(captured.value)
    assert response.closed


def test_openai_stream_rejects_nested_json_with_configured_depth_limit(monkeypatch):
    depth = 65
    nested = b"[" * depth + b"0" + b"]" * depth
    response = _Response([b"data: " + nested + b"\n\n"])
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    iterator = ModelGateway({"openai-compatible": _provider()}).stream(
        "openai-compatible", GenerationRequest("prompt", "fixture-model")
    )

    with pytest.raises(ModelResponseError, match="malformed JSON"):
        list(iterator)

    assert response.closed


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_non_complete_openai_finish_reason_rejects_partial_text_and_closes_response(
    monkeypatch, finish_reason: str
) -> None:
    response = _Response(
        [
            _event({"choices": [{"delta": {"content": "partial-secret"}, "finish_reason": None}]}),
            _event({"choices": [{"delta": {}, "finish_reason": finish_reason}]}),
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    iterator = ModelGateway({"openai-compatible": _provider()}).stream(
        "openai-compatible", GenerationRequest("prompt", "fixture-model")
    )

    assert next(iterator) == "partial-secret"
    with pytest.raises(ModelResponseError, match="without a complete text response") as caught:
        next(iterator)
    assert "partial-secret" not in str(caught.value)
    assert response.closed


def test_closing_stream_closes_underlying_http_response(monkeypatch):
    response = _Response(
        [
            _event({"choices": [{"delta": {"content": "first"}, "finish_reason": None}]}),
            _event({"choices": [{"delta": {"content": "second"}, "finish_reason": None}]}),
        ]
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    iterator = ModelGateway({"openai-compatible": _provider()}).stream(
        "openai-compatible", GenerationRequest("prompt", "fixture-model")
    )

    assert next(iterator) == "first"
    iterator.close()
    assert response.closed


def test_http_error_during_stream_does_not_expose_remote_body(monkeypatch):
    secret = "stream-http-error-secret"
    errors: list[HTTPError] = []

    def fail(*_args, **_kwargs):
        error = HTTPError(
            "https://provider.example/v1/chat/completions",
            500,
            "Server Error",
            Message(),
            BytesIO(json.dumps({"error": {"message": secret}}).encode()),
        )
        errors.append(error)
        raise error

    monkeypatch.setattr("discourselens.providers._open_provider_request", fail)
    iterator = ModelGateway({"openai-compatible": _provider()}).stream(
        "openai-compatible", GenerationRequest("prompt", "fixture-model")
    )

    with pytest.raises(ModelProviderError) as captured:
        next(iterator)
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert errors[0].fp is not None
    assert errors[0].fp.closed
