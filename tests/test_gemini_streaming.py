from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from typing import Self
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from discourselens import Project
from discourselens.errors import AuthenticationError, ModelProviderError, ModelResponseError
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
from discourselens.providers import GeminiProvider


class _Secrets:
    def get(self, reference: str) -> str:
        assert reference in {"fixture", "DL_GEMINI_STREAM_FIXTURE_KEY"}
        return "gemini-stream-secret"


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


def _event(payload: object) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


def _candidate(
    *,
    parts: list[dict[str, object]] | None = None,
    finish_reason: str | None = None,
    safety_ratings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {}
    if parts is not None:
        candidate["content"] = {"role": "model", "parts": parts}
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    if safety_ratings is not None:
        candidate["safetyRatings"] = safety_ratings
    return candidate


def _provider(*, streaming: bool = True) -> GeminiProvider:
    return GeminiProvider(
        "fixture",
        _Secrets(),
        capabilities_by_model={
            "fixture-model": ModelCapabilities(
                streaming=streaming,
                system_prompt=True,
                generation_parameters=("temperature", "max_output_tokens"),
            )
        },
    )


def test_gateway_streams_only_candidate_text_and_uses_gemini_sse_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(
        b"".join(
            [
                _event(
                    {
                        "candidates": [
                            _candidate(
                                parts=[
                                    {"text": "Hola "},
                                    {"text": "private thought", "thought": True},
                                    {"functionCall": {"name": "ignored"}},
                                ]
                            )
                        ]
                    }
                ),
                _event({"candidates": [_candidate(parts=[{"text": "mundo."}])]}),
                _event(
                    {
                        "candidates": [
                            _candidate(parts=[], finish_reason="STOP", safety_ratings=[])
                        ],
                        "usageMetadata": {"candidatesTokenCount": 4},
                    }
                ),
            ]
        ),
        max_chunk=5,
    )
    captured: list[tuple[Request, float]] = []

    def open_request(request: Request, *, timeout: float):
        captured.append((request, timeout))
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    provider = _provider()
    request = GenerationRequest(
        "prompt",
        "fixture-model",
        system_prompt="system",
        parameters={"temperature": 0.2, "max_output_tokens": 123},
    )

    assert list(ModelGateway({"gemini": provider}).stream("gemini", request)) == [
        "Hola ",
        "mundo.",
    ]
    assert response.closed

    http_request, _timeout = captured[0]
    assert http_request.full_url == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/fixture-model:streamGenerateContent?alt=sse"
    )
    assert http_request.get_method() == "POST"
    headers = {name.lower(): value for name, value in http_request.header_items()}
    assert headers["accept"] == "text/event-stream"
    assert headers["content-type"] == "application/json"
    assert headers["x-goog-api-key"] == "gemini-stream-secret"
    assert isinstance(http_request.data, (bytes, bytearray, str))
    payload = json.loads(http_request.data)
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "prompt"}]}]
    assert payload["systemInstruction"] == {"parts": [{"text": "system"}]}
    assert payload["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": 123}


@pytest.mark.parametrize(
    ("body", "error_text"),
    [
        (b"data: {remote-secret}\n\n", "malformed JSON"),
        (
            _event({"error": {"message": "remote-secret"}}),
            "streaming error event",
        ),
        (
            _event({"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}),
            "blocked the prompt",
        ),
        (
            _event(
                {
                    "candidates": [
                        _candidate(
                            parts=[{"text": "remote-secret"}],
                            finish_reason="SAFETY",
                        )
                    ]
                }
            ),
            "safety policy",
        ),
        (
            _event(
                {
                    "candidates": [
                        _candidate(
                            parts=[{"text": "remote-secret"}],
                            safety_ratings=[
                                {"category": "HARM_CATEGORY_DANGEROUS", "blocked": True}
                            ],
                        )
                    ]
                }
            ),
            "safety policy",
        ),
        (
            _event({"candidates": [_candidate(parts=[{"text": "partial"}])]}),
            "without a finish reason",
        ),
    ],
    ids=[
        "malformed",
        "error-event",
        "prompt-safety",
        "finish-safety",
        "safety-rating",
        "incomplete",
    ],
)
def test_gemini_stream_rejects_malformed_blocked_and_incomplete_events_safely(
    monkeypatch: pytest.MonkeyPatch, body: bytes, error_text: str
) -> None:
    response = _Response(body)
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ModelResponseError, match=error_text) as error:
        list(
            ModelGateway({"gemini": _provider()}).stream(
                "gemini", GenerationRequest("prompt", "fixture-model")
            )
        )
    assert "remote-secret" not in str(error.value)
    assert response.closed


def test_gemini_stream_rejects_truncated_max_tokens_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(
        _event({"candidates": [_candidate(parts=[{"text": "partial"}])]})
        + _event(
            {
                "candidates": [
                    _candidate(
                        parts=[{"text": " terminal-secret"}],
                        finish_reason="MAX_TOKENS",
                    )
                ]
            }
        )
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )
    stream = ModelGateway({"gemini": _provider()}).stream(
        "gemini", GenerationRequest("prompt", "fixture-model")
    )
    deltas = [next(stream)]
    with pytest.raises(ModelResponseError, match="output token limit") as error:
        next(stream)

    assert deltas == ["partial"]
    assert "terminal-secret" not in str(error.value)
    assert response.closed


def test_gemini_stream_closes_transport_when_consumer_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(
        _event({"candidates": [_candidate(parts=[{"text": "first"}])]})
        + _event({"candidates": [_candidate(parts=[{"text": "second"}], finish_reason="STOP")]})
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: response,
    )

    chunks = ModelGateway({"gemini": _provider()}).stream(
        "gemini", GenerationRequest("prompt", "fixture-model")
    )
    assert next(chunks) == "first"
    chunks.close()
    assert response.closed


def test_gemini_stream_requires_declared_exact_model_capability_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def open_request(*_args: object, **_kwargs: object):
        nonlocal opened
        opened = True
        raise AssertionError("transport must not be opened")

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    with pytest.raises(ModelProviderError, match="does not declare streaming support"):
        list(
            ModelGateway({"gemini": _provider(streaming=False)}).stream(
                "gemini", GenerationRequest("prompt", "fixture-model")
            )
        )
    assert not opened


def _preset_project(path, *, declared_streaming: bool | None) -> Project:
    project = Project.init(path)
    project.save_credential_reference(
        CredentialReference(
            "gemini",
            "research",
            "DL_GEMINI_STREAM_FIXTURE_KEY",
        )
    )
    project.save_model_preset(ModelPreset("generate", "gemini", "research", "fixture-model"))
    if declared_streaming is not None:
        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "gemini",
                "fixture-model",
                ModelCapabilities(streaming=declared_streaming, system_prompt=True),
            )
        )
    return project


def test_gemini_preset_stream_uses_exact_model_capability_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("DL_GEMINI_STREAM_FIXTURE_KEY", "preset-secret")
    response = _Response(
        _event({"candidates": [_candidate(parts=[{"text": "preset-text"}])]})
        + _event({"candidates": [_candidate(parts=[], finish_reason="STOP")]})
    )
    captured: list[Request] = []

    def open_request(request: Request, *, timeout: float):
        captured.append(request)
        return response

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    project = _preset_project(tmp_path / "gemini-preset-stream", declared_streaming=True)

    def provider_factory(endpoint: str | None, reference: str | None, provider_id: str):
        assert endpoint == "https://generativelanguage.googleapis.com/v1beta"
        assert reference == "DL_GEMINI_STREAM_FIXTURE_KEY"
        assert provider_id == "gemini"
        return GeminiProvider(reference, _Secrets(), endpoint=endpoint)

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
def test_gemini_preset_stream_rejects_missing_or_false_exact_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path, declared_streaming: bool | None
) -> None:
    project = _preset_project(
        tmp_path / f"gemini-preset-no-stream-{declared_streaming}",
        declared_streaming=declared_streaming,
    )
    opened = False
    constructed = False

    def open_request(*_args: object, **_kwargs: object):
        nonlocal opened
        opened = True
        raise AssertionError("transport must not be opened")

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


def test_gemini_stream_http_errors_use_sanitized_shared_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = BytesIO(b'{"error":{"message":"gemini-stream-secret"}}')
    error = HTTPError(
        "https://generativelanguage.googleapis.com/v1beta/models/fixture-model:streamGenerateContent?alt=sse",
        401,
        "gemini-stream-secret",
        Message(),
        body,
    )
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(AuthenticationError) as caught:
        list(
            ModelGateway({"gemini": _provider()}).stream(
                "gemini", GenerationRequest("prompt", "fixture-model")
            )
        )
    assert "gemini-stream-secret" not in str(caught.value)


def test_gemini_buffered_generation_contract_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    def post_json(url: str, payload: dict[str, object], *_args: object, **_kwargs: object):
        captured.append((url, payload))
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": "buffered"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2},
        }

    monkeypatch.setattr("discourselens.providers._post_json", post_json)
    response = _provider().generate(GenerationRequest("prompt", "fixture-model"))

    assert isinstance(response, GenerationResponse)
    assert response.text == "buffered"
    assert captured[0][0].endswith("/models/fixture-model:generateContent")
    assert "stream" not in captured[0][1]
