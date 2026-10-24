from __future__ import annotations

import json
from email.message import Message
from typing import Self, cast
from urllib.request import Request

import pytest

from discourselens.errors import ModelProviderError, ModelResponseError
from discourselens.models import GenerationRequest, ModelCapabilities, TokenUsage
from discourselens.providers import AnthropicCompatibleProvider


class _Secrets:
    def __init__(self, secret: str = "fixture-secret") -> None:
        self.secret = secret
        self.references: list[str] = []

    def get(self, reference: str) -> str:
        self.references.append(reference)
        return self.secret


class _Response:
    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload).encode("utf-8")
        self.headers = Message()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def test_anthropic_compatible_messages_wire_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = _Secrets()
    provider = AnthropicCompatibleProvider(
        "https://gateway.example/v1/",
        "ANTHROPIC_COMPATIBLE_KEY",
        secrets,
        provider_id="gateway-anthropic",
        timeout_seconds=12.5,
    )
    captured_request: Request | None = None
    captured_timeout: float | None = None

    def open_request(request: Request, *, timeout: float) -> _Response:
        nonlocal captured_request, captured_timeout
        captured_request = request
        captured_timeout = timeout
        return _Response(
            {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
                "usage": {"input_tokens": 13, "output_tokens": 4},
                "stop_reason": "end_turn",
            }
        )

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    response = provider.generate(
        GenerationRequest(
            "Say hello",
            "fixture-model",
            system_prompt="Be concise",
            parameters={
                "max_tokens": 123,
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 7,
                "stop_sequences": ["END"],
            },
        )
    )

    request = captured_request
    assert request is not None
    assert request.full_url == "https://gateway.example/v1/messages"
    assert request.get_method() == "POST"
    assert request.get_header("X-api-key") == "fixture-secret"
    assert request.get_header("Anthropic-version") == "2023-06-01"
    assert request.get_header("Content-type") == "application/json"
    assert captured_timeout == 12.5
    assert secrets.references == ["ANTHROPIC_COMPATIBLE_KEY"]
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "fixture-model",
        "max_tokens": 123,
        "messages": [{"role": "user", "content": "Say hello"}],
        "system": "Be concise",
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 7,
        "stop_sequences": ["END"],
    }
    assert response.text == "Hello world"
    assert response.provider == "gateway-anthropic"
    assert response.model == "fixture-model"
    assert response.usage == TokenUsage(input_tokens=13, output_tokens=4)


def test_anthropic_compatible_default_capabilities_fail_closed() -> None:
    provider = AnthropicCompatibleProvider("https://gateway.example/v1", "KEY", _Secrets())

    capabilities = provider.capabilities("unprofiled-model")
    assert capabilities.system_prompt
    assert not capabilities.structured_output
    assert capabilities.reasoning_efforts == ()
    assert capabilities.reasoning_modes == ()
    assert capabilities.generation_parameters == (
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
    )


@pytest.mark.parametrize(
    "generation_request",
    [
        GenerationRequest(
            "prompt",
            "unprofiled-model",
            response_schema_name="output",
            response_schema={"type": "object", "properties": {}},
        ),
        GenerationRequest("prompt", "unprofiled-model", reasoning="high"),
        GenerationRequest("prompt", "unprofiled-model", reasoning_mode="adaptive"),
    ],
    ids=("structured-output", "reasoning-effort", "reasoning-mode"),
)
def test_anthropic_compatible_rejects_undeclared_capabilities(
    monkeypatch: pytest.MonkeyPatch, generation_request: GenerationRequest
) -> None:
    provider = AnthropicCompatibleProvider("https://gateway.example/v1", "KEY", _Secrets())
    called = False

    def forbidden_request(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("unsupported requests must fail before transport")

    monkeypatch.setattr("discourselens.providers._open_provider_request", forbidden_request)
    with pytest.raises(ModelProviderError):
        provider.generate(generation_request)
    assert not called


def test_anthropic_compatible_exact_model_profile_can_enable_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicCompatibleProvider(
        "https://gateway.example/v1",
        "KEY",
        _Secrets(),
        capabilities_by_model={
            "schema-model": ModelCapabilities(
                structured_output=True,
                generation_parameters=("max_tokens", "temperature"),
            )
        },
    )
    captured_payload: dict[str, object] | None = None

    def open_request(request: Request, *, timeout: float) -> _Response:
        nonlocal captured_payload
        body = request.data
        assert isinstance(body, bytes)
        captured_payload = json.loads(body.decode("utf-8"))
        return _Response({"content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn"})

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    provider.generate(
        GenerationRequest(
            "Return JSON",
            "schema-model",
            response_schema_name="result",
            response_schema=schema,
        )
    )

    assert captured_payload is not None
    assert captured_payload["output_config"] == {
        "format": {"type": "json_schema", "schema": schema}
    }


@pytest.mark.parametrize(
    ("reasoning_mode", "expected_thinking"),
    [
        ("adaptive", {"type": "adaptive"}),
        ("manual", {"type": "enabled", "budget_tokens": 2048}),
    ],
)
def test_anthropic_compatible_combines_structured_output_and_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_mode: str,
    expected_thinking: dict[str, object],
) -> None:
    provider = AnthropicCompatibleProvider(
        "https://gateway.example/v1",
        "KEY",
        _Secrets(),
        capabilities_by_model={
            "schema-reasoning-model": ModelCapabilities(
                structured_output=True,
                reasoning_efforts=("high",),
                reasoning_modes=("adaptive", "manual"),
                generation_parameters=("max_tokens",),
            )
        },
    )
    captured_payload: dict[str, object] | None = None

    def open_request(request: Request, *, timeout: float) -> _Response:
        nonlocal captured_payload
        body = request.data
        assert isinstance(body, bytes)
        captured_payload = json.loads(body.decode("utf-8"))
        return _Response(
            {"content": [{"type": "text", "text": '{"ok":true}'}], "stop_reason": "end_turn"}
        )

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    provider.generate(
        GenerationRequest(
            "Return JSON",
            "schema-reasoning-model",
            "high",
            response_schema_name="result",
            response_schema=schema,
            parameters={"max_tokens": 4096} if reasoning_mode == "manual" else {},
            reasoning_mode=reasoning_mode,
            reasoning_budget_tokens=2048 if reasoning_mode == "manual" else None,
        )
    )

    assert captured_payload is not None
    assert captured_payload["thinking"] == expected_thinking
    assert captured_payload["output_config"] == {
        "format": {"type": "json_schema", "schema": schema},
        "effort": "high",
    }


def test_anthropic_compatible_does_not_silently_drop_unmapped_profile_parameters() -> None:
    provider = AnthropicCompatibleProvider(
        "https://gateway.example/v1",
        "KEY",
        _Secrets(),
        capabilities_by_model={
            "profiled-model": ModelCapabilities(generation_parameters=("custom_option",))
        },
    )

    with pytest.raises(ModelProviderError, match="no wire mapping"):
        provider.generate(
            GenerationRequest("prompt", "profiled-model", parameters={"custom_option": True})
        )


@pytest.mark.parametrize(
    "endpoint",
    ["relative/path", "https://user:pass@gateway.example/v1", "https://gateway.example/v1?x=1"],
)
def test_anthropic_compatible_rejects_invalid_endpoint(endpoint: str) -> None:
    with pytest.raises(ModelProviderError, match="endpoint"):
        AnthropicCompatibleProvider(endpoint, "KEY", _Secrets())


def test_anthropic_compatible_rejects_empty_credential_reference() -> None:
    with pytest.raises(ModelProviderError, match="credential reference"):
        AnthropicCompatibleProvider("https://gateway.example/v1", "  ", _Secrets())


@pytest.mark.parametrize(
    "payload",
    [
        {"content": []},
        {"content": [None]},
        {"content": [{"type": "text"}]},
        {"content": [{"type": "tool_use", "name": "unexpected"}]},
        {
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": True, "output_tokens": 3},
        },
    ],
    ids=("empty-content", "non-object-block", "missing-text", "unexpected-block", "invalid-usage"),
)
def test_anthropic_compatible_rejects_malformed_responses(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    provider = AnthropicCompatibleProvider("https://gateway.example/v1", "KEY", _Secrets())
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        lambda *_args, **_kwargs: _Response(payload),
    )

    with pytest.raises(ModelResponseError):
        provider.generate(GenerationRequest("prompt", "fixture-model"))


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), True])
def test_anthropic_compatible_rejects_invalid_timeout(timeout: float | bool) -> None:
    with pytest.raises(ModelProviderError, match="timeout"):
        AnthropicCompatibleProvider(
            "https://gateway.example/v1",
            "KEY",
            _Secrets(),
            timeout_seconds=cast(float, timeout),
        )
