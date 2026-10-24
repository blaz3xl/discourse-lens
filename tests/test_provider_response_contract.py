"""Focused wire-contract checks for generation provider responses."""

from __future__ import annotations

import json
import traceback
from email.message import Message
from io import BytesIO
from typing import Self
from urllib.error import HTTPError

import pytest

from discourselens.errors import ModelProviderError, ModelResponseError
from discourselens.models import GenerationRequest
from discourselens.providers import (
    AnthropicCompatibleProvider,
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)


class _Secrets:
    def get(self, reference: str) -> str:
        return "fixture-secret"


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.headers = Message()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


_PROVIDER_KINDS = (
    "openai-compatible",
    "openai",
    "anthropic",
    "anthropic-compatible",
    "gemini",
)


def _provider(kind: str):
    secrets = _Secrets()
    if kind == "openai-compatible":
        return OpenAICompatibleProvider("https://example.test/v1", "KEY", secrets)
    if kind == "openai":
        return OpenAIProvider("KEY", secrets)
    if kind == "anthropic":
        return AnthropicProvider("KEY", secrets)
    if kind == "anthropic-compatible":
        return AnthropicCompatibleProvider("https://example.test/v1", "KEY", secrets)
    if kind == "gemini":
        return GeminiProvider("KEY", secrets)
    raise AssertionError(f"unknown provider kind: {kind}")


def _generate(kind: str):
    return _provider(kind).generate(GenerationRequest("fixture prompt", "fixture-model"))


def _complete_payload(kind: str, *, completion: str = "complete fixture") -> dict[str, object]:
    if kind in {"openai-compatible", "openai"}:
        return {
            "choices": [
                {"message": {"content": completion}, "finish_reason": "stop"}
            ]
        }
    if kind in {"anthropic", "anthropic-compatible"}:
        return {
            "content": [{"type": "text", "text": completion}],
            "stop_reason": "end_turn",
        }
    if kind == "gemini":
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": completion}]},
                    "finishReason": "STOP",
                }
            ]
        }
    raise AssertionError(f"unknown provider kind: {kind}")


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_buffered_generation_accepts_only_documented_complete_terminal_states(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    payload = _complete_payload(provider_kind)
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: _Response(json.dumps(payload).encode("utf-8")),
    )

    response = _generate(provider_kind)

    assert response.text == "complete fixture"


@pytest.mark.parametrize("provider_kind", ["anthropic", "anthropic-compatible"])
@pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
def test_anthropic_buffered_generation_accepts_both_complete_stop_reasons(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str, stop_reason: str
) -> None:
    payload = _complete_payload(provider_kind)
    payload["stop_reason"] = stop_reason
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: _Response(json.dumps(payload).encode("utf-8")),
    )

    response = _generate(provider_kind)

    assert response.text == "complete fixture"


@pytest.mark.parametrize(
    ("provider_kind", "payload", "message"),
    [
        (
            "openai-compatible",
            {"choices": [{"message": {"content": "partial-marker"}, "finish_reason": "length"}]},
            "output limit before completion",
        ),
        (
            "openai",
            {
                "choices": [
                    {"message": {"content": "filtered-marker"}, "finish_reason": "content_filter"}
                ]
            },
            "filtered the response before completion",
        ),
        (
            "openai-compatible",
            {
                "choices": [
                    {"message": {"content": None}, "finish_reason": "tool_calls"}
                ]
            },
            "unsupported tool handoff",
        ),
        (
            "openai",
            {
                "choices": [
                    {
                        "message": {"content": "legacy-tool-marker"},
                        "finish_reason": "function_call",
                    }
                ]
            },
            "unsupported tool handoff",
        ),
        (
            "openai",
            {
                "choices": [
                    {
                        "message": {
                            "content": "refusal-marker",
                            "refusal": "private refusal detail",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            "refused to complete",
        ),
        (
            "openai-compatible",
            {"choices": [{"message": {"content": "missing-marker"}}]},
            "missing or unsupported finish reason",
        ),
        (
            "openai-compatible",
            {
                "choices": [
                    {
                        "message": {"content": "malformed-marker"},
                        "finish_reason": ["stop"],
                    }
                ]
            },
            "missing or unsupported finish reason",
        ),
        (
            "openai",
            {
                "choices": [
                    {
                        "message": {"content": "unknown-marker"},
                        "finish_reason": "remote-secret-terminal-marker",
                    }
                ]
            },
            "missing or unsupported finish reason",
        ),
        (
            "anthropic",
            {"content": [{"type": "text", "text": "partial-marker"}], "stop_reason": "max_tokens"},
            "output limit before completion",
        ),
        (
            "anthropic-compatible",
            {
                "content": [{"type": "tool_use", "name": "private-tool-name"}],
                "stop_reason": "tool_use",
            },
            "unsupported tool handoff",
        ),
        (
            "anthropic",
            {
                "content": [{"type": "text", "text": "refusal-marker"}],
                "stop_reason": "refusal",
            },
            "refused to complete",
        ),
        (
            "anthropic-compatible",
            {"content": [{"type": "text", "text": "missing-marker"}]},
            "missing or unsupported stop reason",
        ),
        (
            "anthropic-compatible",
            {
                "content": [{"type": "text", "text": "malformed-marker"}],
                "stop_reason": ["end_turn"],
            },
            "missing or unsupported stop reason",
        ),
        (
            "anthropic",
            {
                "content": [{"type": "text", "text": "unknown-marker"}],
                "stop_reason": "remote-secret-terminal-marker",
            },
            "missing or unsupported stop reason",
        ),
        (
            "anthropic-compatible",
            {
                "content": [{"type": "text", "text": "partial-marker"}],
                "stop_reason": "model_context_window_exceeded",
            },
            "missing or unsupported stop reason",
        ),
        (
            "gemini",
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "partial-marker"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            },
            "output limit before completion",
        ),
        (
            "gemini",
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "filtered-marker"}]},
                        "finishReason": "SAFETY",
                    }
                ]
            },
            "blocked the candidate under a safety policy",
        ),
        (
            "gemini",
            {
                "promptFeedback": {"blockReason": "SAFETY"},
                "candidates": [],
            },
            "blocked the prompt under a safety policy",
        ),
        (
            "gemini",
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "blocked-marker"}]},
                        "finishReason": "STOP",
                        "safetyRatings": [{"blocked": True}],
                    }
                ]
            },
            "blocked the candidate under a safety policy",
        ),
        (
            "gemini",
            {
                "candidates": [
                    {
                        "content": {"parts": [{"functionCall": {"name": "private-tool"}}]},
                        "finishReason": "STOP",
                    }
                ]
            },
            "unsupported tool handoff",
        ),
        (
            "gemini",
            {"candidates": [{"content": {"parts": [{"text": "missing-marker"}]}}]},
            "missing or malformed finish reason",
        ),
        (
            "gemini",
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "unknown-marker"}]},
                        "finishReason": "REMOTE_SECRET_TERMINAL_MARKER",
                    }
                ]
            },
            "unsupported finish reason",
        ),
        (
            "gemini",
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "malformed-marker"}]},
                        "finishReason": ["STOP"],
                    }
                ]
            },
            "missing or malformed finish reason",
        ),
    ],
)
def test_buffered_generation_rejects_incomplete_or_non_text_terminal_states_safely(
    monkeypatch: pytest.MonkeyPatch,
    provider_kind: str,
    payload: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: _Response(json.dumps(payload).encode("utf-8")),
    )

    with pytest.raises(ModelResponseError, match=message) as caught:
        _generate(provider_kind)

    assert "marker" not in str(caught.value)
    assert "REMOTE_SECRET_TERMINAL_MARKER" not in str(caught.value)
    assert "private refusal detail" not in str(caught.value)


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
@pytest.mark.parametrize(
    "body", [b'{"partial":', b"[" * 10000 + b"]" * 10000], ids=("truncated", "over-nested")
)
def test_invalid_or_over_nested_success_json_becomes_a_safe_domain_error(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str, body: bytes
) -> None:
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: _Response(body),
    )

    with pytest.raises(ModelProviderError, match="provider response could not be read"):
        _generate(provider_kind)


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_non_object_success_json_becomes_a_response_error(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: _Response(b"[]"),
    )

    with pytest.raises(ModelResponseError, match="provider response must be a JSON object"):
        _generate(provider_kind)


@pytest.mark.parametrize(
    ("provider_kind", "payload", "message"),
    [
        (
            "openai-compatible",
            {"choices": []},
            "provider response lacks a chat completion",
        ),
        (
            "anthropic",
            {"content": [None], "stop_reason": "end_turn"},
            "Anthropic response lacks text content",
        ),
        (
            "anthropic-compatible",
            {"content": [None], "stop_reason": "end_turn"},
            "Anthropic-compatible response contains a malformed content block",
        ),
        ("gemini", {"candidates": []}, "Gemini response lacks text content"),
    ],
)
def test_wrong_shaped_success_json_becomes_a_response_error(
    monkeypatch: pytest.MonkeyPatch,
    provider_kind: str,
    payload: dict[str, object],
    message: str,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: _Response(body),
    )

    with pytest.raises(ModelResponseError, match=message):
        _generate(provider_kind)


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_provider_http_error_body_never_escapes_domain_error(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    marker = "response-secret-marker-9f318a"
    body = json.dumps({"error": {"code": "bad_request", "message": marker}}).encode()

    def fail_request(*_args: object, **_kwargs: object) -> None:
        raise HTTPError(
            "https://example.test/generate",
            400,
            "Bad Request",
            Message(),
            BytesIO(body),
        )

    monkeypatch.setattr("discourselens.providers._open_provider_request", fail_request)

    with pytest.raises(ModelProviderError) as captured:
        _generate(provider_kind)

    assert marker not in str(captured.value)
    assert marker not in "".join(traceback.format_exception(captured.value))
    assert captured.value.__suppress_context__


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
@pytest.mark.parametrize(
    "parameter_value",
    [pytest.param(float("nan"), id="non-finite-number"), pytest.param(object(), id="not-json")],
)
def test_generation_rejects_invalid_json_parameters_before_transport(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str, parameter_value: object
) -> None:
    called = False

    def forbidden_request(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("invalid request values must fail before transport")

    monkeypatch.setattr(
        "discourselens.providers._open_provider_request", forbidden_request
    )

    with pytest.raises(ModelProviderError, match="valid JSON values"):
        _provider(provider_kind).generate(
            GenerationRequest(
                "fixture prompt",
                "fixture-model",
                parameters={"temperature": parameter_value},
            )
        )

    assert not called


@pytest.mark.parametrize("provider_kind", _PROVIDER_KINDS)
def test_deeply_nested_http_error_body_still_becomes_a_safe_domain_error(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str
) -> None:
    marker = "nested-error-secret-marker"
    body = (
        b'{"error":"'
        + marker.encode()
        + b'","details":'
        + b"[" * 20_000
        + b"]" * 20_000
        + b"}"
    )

    def fail_request(*_args: object, **_kwargs: object) -> None:
        raise HTTPError(
            "https://example.test/generate",
            400,
            "Bad Request",
            Message(),
            BytesIO(body),
        )

    monkeypatch.setattr("discourselens.providers._open_provider_request", fail_request)

    with pytest.raises(ModelProviderError, match="HTTP error: 400") as captured:
        _generate(provider_kind)

    assert marker not in str(captured.value)
    assert marker not in "".join(traceback.format_exception(captured.value))
    assert captured.value.__suppress_context__
