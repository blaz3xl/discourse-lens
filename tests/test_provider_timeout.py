from __future__ import annotations

import traceback
from collections.abc import Callable

import pytest

from discourselens.errors import ModelProviderError
from discourselens.models import GenerationRequest, ModelCapabilities
from discourselens.providers import OpenAICompatibleProvider


class _Secrets:
    def get(self, reference: str) -> str:
        assert reference == "timeout-fixture"
        return "timeout-credential-sentinel"


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "https://provider.example/v1",
        "timeout-fixture",
        _Secrets(),
        capabilities_by_model={"fixture-model": ModelCapabilities(streaming=True)},
    )


def _assert_timeout_is_sanitized(
    operation: Callable[[], object], *, expected_message: str
) -> None:
    with pytest.raises(ModelProviderError) as caught:
        operation()

    rendered_traceback = "".join(
        traceback.format_exception(caught.type, caught.value, caught.value.__traceback__)
    )
    assert str(caught.value) == expected_message
    assert "timeout-credential-sentinel" not in str(caught.value)
    assert "timeout-credential-sentinel" not in rendered_traceback
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
def test_provider_timeout_is_normalized_without_exposing_transport_detail(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError("timeout-credential-sentinel")

    monkeypatch.setattr("discourselens.providers._open_provider_request", timeout)
    provider = _provider()
    request = GenerationRequest("prompt", "fixture-model")

    if streaming:
        iterator = provider.stream_generate(request)
        _assert_timeout_is_sanitized(
            lambda: next(iterator),
            expected_message="provider streaming response could not be read",
        )
    else:
        _assert_timeout_is_sanitized(
            lambda: provider.generate(request),
            expected_message="provider response could not be read",
        )
