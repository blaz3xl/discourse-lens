from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest

from discourselens import Project
from discourselens.errors import ModelProviderError, ProjectError
from discourselens.execution import execute_model_preset_stream
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
)


class _StreamingProvider:
    provider_id = "openai-compatible"

    def __init__(self, *, fail: bool = False) -> None:
        self.started = False
        self.closed = False
        self.fail = fail

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities(streaming=True)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse("buffered", self.provider_id, request.model)

    def stream_generate(self, _request: GenerationRequest) -> Iterator[str]:
        self.started = True
        try:
            yield "hello"
            if self.fail:
                raise ModelProviderError("fixture stream failure")
            yield " world"
        finally:
            self.closed = True


def _streaming_project(path, *, capability: bool = True) -> Project:
    project = Project.init(path)
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible",
            "research",
            "DL_STREAMING_FIXTURE_KEY",
            "https://provider.example/v1",
        )
    )
    project.save_model_preset(
        ModelPreset("generate", "openai-compatible", "research", "fixture-model")
    )
    if capability:
        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "openai-compatible", "fixture-model", ModelCapabilities(streaming=True)
            )
        )
    return project


def test_streaming_preset_records_hash_only_after_full_exhaustion(tmp_path):
    project = _streaming_project(tmp_path / "stream-success")
    provider = _StreamingProvider()

    def factory(
        _endpoint: str | None, _reference: str | None, _provider_id: str
    ) -> _StreamingProvider:
        return provider

    try:
        chunks = execute_model_preset_stream(
            project,
            "generate",
            "fixture prompt",
            provider_factory=factory,
        )
        assert not provider.started
        assert list(chunks) == ["hello", " world"]
        assert provider.started and provider.closed

        row = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert row is not None
        assert row["error_type"] is None
        assert row["endpoint"] == "https://provider.example/v1"
        assert row["request_hash"] == hashlib.sha256(b"fixture prompt").hexdigest()
        assert row["response_hash"] == hashlib.sha256(b"hello world").hexdigest()
        assert "fixture prompt" not in str(dict(row))
        assert "hello world" not in str(dict(row))
        assert row["input_tokens"] is None and row["output_tokens"] is None
    finally:
        project.close()


def test_streaming_cancellation_closes_transport_and_is_not_success(tmp_path):
    project = _streaming_project(tmp_path / "stream-cancel")
    provider = _StreamingProvider()

    def factory(
        _endpoint: str | None, _reference: str | None, _provider_id: str
    ) -> _StreamingProvider:
        return provider

    try:
        stream = execute_model_preset_stream(
            project,
            "generate",
            "fixture prompt",
            provider_factory=factory,
        )
        assert next(stream) == "hello"
        stream.close()

        assert provider.closed
        row = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert row is not None
        assert row["error_type"] == "Cancelled"
        assert row["endpoint"] == "https://provider.example/v1"
        assert row["response_hash"] is None
    finally:
        project.close()


def test_stream_failure_records_endpoint_and_no_response_hash(tmp_path):
    project = _streaming_project(tmp_path / "stream-failure")
    provider = _StreamingProvider(fail=True)

    def factory(
        _endpoint: str | None, _reference: str | None, _provider_id: str
    ) -> _StreamingProvider:
        return provider

    try:
        stream = execute_model_preset_stream(
            project, "generate", "fixture prompt", provider_factory=factory
        )
        with pytest.raises(ModelProviderError, match="fixture stream failure"):
            list(stream)
        row = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert row["endpoint"] == "https://provider.example/v1"
        assert row["error_type"] == "ModelProviderError"
        assert row["response_hash"] is None
    finally:
        project.close()


def test_streaming_requires_exact_model_capability(tmp_path):
    project = _streaming_project(tmp_path / "stream-capability", capability=False)
    created = False

    def provider_factory(
        _endpoint: str | None, _reference: str | None, _provider_id: str
    ) -> _StreamingProvider:
        nonlocal created
        created = True
        return _StreamingProvider()

    try:
        stream = execute_model_preset_stream(
            project, "generate", "fixture prompt", provider_factory=provider_factory
        )
        with pytest.raises(ModelProviderError, match="exact-model capability profile"):
            next(stream)
        assert not created
        assert project.db.execute("SELECT count(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


def test_offline_policy_is_rechecked_immediately_before_stream_open(tmp_path):
    project = _streaming_project(tmp_path / "stream-offline-race")
    provider = _StreamingProvider()

    def provider_factory(
        _endpoint: str | None, _reference: str | None, _provider_id: str
    ) -> _StreamingProvider:
        project.set_offline_mode(True)
        return provider

    try:
        stream = execute_model_preset_stream(
            project, "generate", "fixture prompt", provider_factory=provider_factory
        )
        with pytest.raises(ProjectError, match="offline mode"):
            next(stream)
        assert not provider.started
        row = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert row is not None
        assert row["error_type"] == "ProjectError"
        assert row["response_hash"] is None
    finally:
        project.close()
