from __future__ import annotations

import pytest

from discourselens import Project
from discourselens.errors import ScopeViolationError
from discourselens.models import CredentialReference, ModelPreset
from discourselens.vectors import (
    MAX_PRESET_QUERY_CHARS,
    build_paragraph_embedding_index,
    hybrid_search_with_preset,
    semantic_search_with_preset,
)


class _QueryEmbeddingFixture:
    provider_id = "cohere-embeddings"
    endpoint = "https://api.cohere.com"

    def __init__(self) -> None:
        self.query_texts: list[str] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        assert model == "fixture-model"
        return [(0.25, 0.75) for _ in texts]

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        assert model == "fixture-model"
        self.query_texts.extend(texts)
        return [(0.25, 0.75) for _ in texts]


def _prepared_project(root):
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "A local fixture paragraph.")
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "fixture", "DL_QUERY_BOUND_SECRET")
    )
    project.save_model_preset(
        ModelPreset("fixture-preset", "cohere-embeddings", "fixture", "fixture-model")
    )
    provider = _QueryEmbeddingFixture()
    artifact_id = build_paragraph_embedding_index(
        project, "fixture-preset", revision_id, provider
    )
    scope_id = project.snapshot([revision_id])
    return project, revision_id, artifact_id, scope_id, provider


@pytest.mark.parametrize("search_kind", ["semantic", "hybrid"])
def test_preset_search_accepts_exact_unicode_character_limit(tmp_path, search_kind):
    project, _, artifact_id, scope_id, provider = _prepared_project(tmp_path / search_kind)
    # Keep one searchable lexical token for hybrid mode; the rest are
    # supplementary-plane code points to distinguish characters from bytes.
    query_text = "A" + "🧪" * (MAX_PRESET_QUERY_CHARS - 1)
    try:
        if search_kind == "semantic":
            hits = semantic_search_with_preset(
                project,
                "fixture-preset",
                scope_id,
                [artifact_id],
                query_text,
                embedding_provider=provider,
                limit=1,
            )
        else:
            hits = hybrid_search_with_preset(
                project,
                "fixture-preset",
                scope_id,
                [artifact_id],
                query_text,
                embedding_provider=provider,
                limit=1,
            )

        assert hits
        assert len(query_text) == MAX_PRESET_QUERY_CHARS
        assert len(query_text.encode("utf-8")) == 1 + (MAX_PRESET_QUERY_CHARS - 1) * 4
        assert provider.query_texts == [query_text]
    finally:
        project.close()


@pytest.mark.parametrize("search_kind", ["semantic", "hybrid"])
def test_oversized_preset_query_rejects_before_project_or_provider_lookup(
    tmp_path, monkeypatch, search_kind
):
    project, _, artifact_id, scope_id, provider = _prepared_project(tmp_path / search_kind)
    query_text = "private-query-marker-" + "é" * MAX_PRESET_QUERY_CHARS

    def forbidden_lookup(*args, **kwargs):
        raise AssertionError("oversized query validation must happen before preset/credential lookup")

    monkeypatch.setattr(project, "model_preset", forbidden_lookup)
    monkeypatch.setattr(project, "credential_reference", forbidden_lookup)
    try:
        with pytest.raises(ScopeViolationError) as error:
            if search_kind == "semantic":
                semantic_search_with_preset(
                    project,
                    "fixture-preset",
                    scope_id,
                    [artifact_id],
                    query_text,
                    embedding_provider=provider,
                )
            else:
                hybrid_search_with_preset(
                    project,
                    "fixture-preset",
                    scope_id,
                    [artifact_id],
                    query_text,
                    embedding_provider=provider,
                )

        assert f"{MAX_PRESET_QUERY_CHARS}-character limit" in str(error.value)
        assert "private-query-marker" not in str(error.value)
        assert provider.query_texts == []
    finally:
        project.close()
