"""Focused regressions for fail-closed provider credential alias routing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.request import Request

import pytest

from discourselens import Project
from discourselens.errors import DocumentError, ModelProviderError
from discourselens.execution import (
    build_reranking_gateway,
    execute_model_preset,
    snapshot_model_preset,
)
from discourselens.models import CredentialReference, ModelPreset
from discourselens.secrets import ReferenceSecretStore


def _project_with_generation_aliases(path) -> Project:
    project = Project.init(path)
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "selected", "DL_TEST_SELECTED", "https://fixture.invalid/v1"
        )
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "other", "DL_TEST_OTHER", "https://fixture.invalid/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "selected", "fixture-model")
    )
    return project


def _project_with_reranking_aliases(path) -> Project:
    project = Project.init(path)
    project.save_credential_reference(
        CredentialReference("jina-reranker", "selected", "DL_TEST_RERANK_SELECTED")
    )
    project.save_credential_reference(
        CredentialReference("jina-reranker", "other", "DL_TEST_RERANK_OTHER")
    )
    return project


def _mock_response(payload: dict[str, object]) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    manager = MagicMock()
    manager.__enter__.return_value = response
    return manager


def test_generation_resolves_only_selected_alias_secret(tmp_path, monkeypatch):
    selected_secret = "fixture-selected-generation-token"
    other_secret = "fixture-other-generation-token"
    monkeypatch.setenv("DL_TEST_SELECTED", selected_secret)
    monkeypatch.setenv("DL_TEST_OTHER", other_secret)
    project = _project_with_generation_aliases(tmp_path / "generation-aliases")
    resolved_references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_get(store: ReferenceSecretStore, reference: str) -> str:
        resolved_references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_get)
    response = _mock_response(
        {
            "choices": [
                {"message": {"content": "fixture answer"}, "finish_reason": "stop"}
            ]
        }
    )

    with patch("discourselens.providers._open_provider_request", return_value=response) as opened:
        result = execute_model_preset(project, "review", "fixture prompt")

    request = opened.call_args.args[0]
    assert isinstance(request, Request)
    assert resolved_references == ["DL_TEST_SELECTED"]
    authorization = request.get_header("Authorization")
    assert authorization == f"Bearer {selected_secret}"
    assert authorization is not None and other_secret not in authorization
    assert result.response.text == "fixture answer"


@pytest.mark.parametrize("snapshot_alias", ["other", ""])
def test_generation_rejects_invalid_or_remapped_snapshot_alias_without_fallback(
    tmp_path, snapshot_alias
):
    project = _project_with_generation_aliases(tmp_path / f"generation-{snapshot_alias or 'empty'}")
    snapshot = snapshot_model_preset(project, "review")
    snapshot["credential_alias"] = snapshot_alias
    provider_inputs: list[tuple[str | None, str]] = []

    def provider_factory(endpoint: str | None, reference: str | None, provider_id: str):
        provider_inputs.append((reference, provider_id))
        raise AssertionError("invalid credential selection must fail before provider creation")

    with pytest.raises(ModelProviderError):
        execute_model_preset(
            project,
            "review",
            "fixture prompt",
            preset_snapshot=snapshot,
            provider_factory=provider_factory,
        )

    assert provider_inputs == []


def test_generation_missing_selected_alias_does_not_fall_back_to_other_alias(tmp_path):
    project = _project_with_generation_aliases(tmp_path / "missing-generation-alias")
    snapshot = snapshot_model_preset(project, "review")
    project.db.execute(
        "DELETE FROM provider_credentials WHERE provider_id=? AND credential_alias=?",
        ("openai-compatible", "selected"),
    )
    provider_inputs: list[tuple[str | None, str]] = []

    def provider_factory(endpoint: str | None, reference: str | None, provider_id: str):
        provider_inputs.append((reference, provider_id))
        raise AssertionError("missing selected alias must fail before provider creation")

    with pytest.raises(ModelProviderError, match="credential alias captured.*unavailable"):
        execute_model_preset(
            project,
            "review",
            "fixture prompt",
            preset_snapshot=snapshot,
            provider_factory=provider_factory,
        )

    assert provider_inputs == []


def test_reranking_resolves_only_selected_alias_secret(tmp_path, monkeypatch):
    selected_secret = "fixture-selected-reranking-token"
    other_secret = "fixture-other-reranking-token"
    monkeypatch.setenv("DL_TEST_RERANK_SELECTED", selected_secret)
    monkeypatch.setenv("DL_TEST_RERANK_OTHER", other_secret)
    project = _project_with_reranking_aliases(tmp_path / "reranking-aliases")
    resolved_references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_get(store: ReferenceSecretStore, reference: str) -> str:
        resolved_references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_get)
    response = _mock_response({"results": [{"index": 0, "relevance_score": 0.8}]})
    gateway = build_reranking_gateway(project, "jina-reranker", "selected")

    with patch("discourselens.providers._open_provider_request", return_value=response) as opened:
        result = gateway.rerank("jina-reranker", "fixture-model", "query", ["candidate"])

    request = opened.call_args.args[0]
    assert isinstance(request, Request)
    assert resolved_references == ["DL_TEST_RERANK_SELECTED"]
    authorization = request.get_header("Authorization")
    assert authorization == f"Bearer {selected_secret}"
    assert authorization is not None and other_secret not in authorization
    assert [(item.index, item.relevance_score) for item in result] == [(0, 0.8)]


@pytest.mark.parametrize("invalid_alias", ["missing", ""])
def test_reranking_missing_or_invalid_alias_fails_closed_without_fallback(
    tmp_path, invalid_alias
):
    project = _project_with_reranking_aliases(tmp_path / f"reranking-{invalid_alias or 'empty'}")
    with (
        patch("discourselens.providers._open_provider_request") as opened,
        pytest.raises(DocumentError, match="unknown provider credential alias"),
    ):
        build_reranking_gateway(project, "jina-reranker", invalid_alias)

    opened.assert_not_called()


def test_generation_snapshot_rejects_changed_selected_reference_before_resolution(tmp_path):
    project = _project_with_generation_aliases(tmp_path / "changed-generation-alias")
    snapshot = snapshot_model_preset(project, "review")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "selected", "DL_TEST_REPLACEMENT", "https://fixture.invalid/v1"
        )
    )
    provider_inputs: list[str | None] = []

    def provider_factory(endpoint: str | None, reference: str | None, provider_id: str):
        provider_inputs.append(reference)
        raise AssertionError("changed selected credential must fail before provider creation")

    with pytest.raises(ModelProviderError, match="refusing implicit credential substitution"):
        execute_model_preset(
            project,
            "review",
            "fixture prompt",
            preset_snapshot=snapshot,
            provider_factory=provider_factory,
        )

    assert provider_inputs == []
