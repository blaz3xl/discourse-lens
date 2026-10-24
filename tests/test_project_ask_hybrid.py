"""Focused Core contracts for optional hybrid retrieval in ask_question."""

from __future__ import annotations

import json
from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)
from discourselens.vectors import build_paragraph_embedding_index


class _EmbeddingFixture:
    provider_id = "cohere-embeddings"
    endpoint = "https://api.cohere.com"

    def __init__(self) -> None:
        self.query_texts: list[list[str]] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        assert model == "embedding-model"
        return [(1.0, 0.0) for _ in texts]

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        assert model == "embedding-model"
        self.query_texts.append(list(texts))
        return [(1.0, 0.0) for _ in texts]


class _GenerationFixture:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        marker = "UNTRUSTED_RESEARCH_DATA_JSON:\n"
        research = json.loads(request.prompt.split(marker, 1)[1])
        records = cast(list[dict[str, object]], research["evidence_records"])
        evidence = next(item for item in records if item.get("evidence_id") and item.get("excerpt"))
        payload = {
            "interpretation": "The passage may foreground the selected actors.",
            "alternative_interpretation": "It may also be a concise event report.",
            "evidence_ids": [evidence["evidence_id"]],
            "counterevidence_ids": [],
            "limitations": ["Hybrid retrieval is targeted and may miss relevant passages."],
            "observations": [
                {
                    "evidence_id": evidence["evidence_id"],
                    "excerpt": evidence["excerpt"],
                    "description": "The passage explicitly describes the event.",
                }
            ],
        }
        return GenerationResponse(json.dumps(payload), self.provider_id, request.model)


def _hybrid_project(tmp_path, monkeypatch, name: str = "project"):
    monkeypatch.setenv("DL_ASK_TEST_KEY", "fixture-generation-secret")
    monkeypatch.setenv("DL_ASK_EMBED_KEY", "fixture-embedding-secret")
    project = Project.init(tmp_path / name)
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "review", "DL_ASK_TEST_KEY", "http://127.0.0.1:9/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "review", "review-model")
    )
    project.save_credential_reference(
        CredentialReference(
            "cohere-embeddings",
            "embedding",
            "DL_ASK_EMBED_KEY",
            "https://api.cohere.com",
        )
    )
    project.save_model_preset(
        ModelPreset("embedding", "cohere-embeddings", "embedding", "embedding-model")
    )
    return project


def _install_generation_fixture(monkeypatch) -> None:
    from discourselens import interpretation

    original_execute = interpretation.execute_model_preset
    generation_provider = _GenerationFixture()

    def execute_with_fixture(project, preset_name, prompt, **kwargs):
        kwargs["provider_factory"] = lambda endpoint, reference, provider_id: generation_provider
        return original_execute(project, preset_name, prompt, **kwargs)

    monkeypatch.setattr(interpretation, "execute_model_preset", execute_with_fixture)


def test_ask_question_hybrid_links_candidate_retrieval_and_preserves_scope(
    tmp_path, monkeypatch
):
    project = _hybrid_project(tmp_path, monkeypatch)
    revision_id = project.import_text(
        "speech.txt", "The council welcomed migrants after the public meeting."
    )[1]
    embedding = _EmbeddingFixture()
    artifact_id = build_paragraph_embedding_index(project, "embedding", revision_id, embedding)
    _install_generation_fixture(monkeypatch)

    result = project.ask_question(
        [revision_id],
        "What did the council do about migrants?",
        "fairclough-text",
        "review",
        retrieval_mode="hybrid",
        embedding_preset_name="embedding",
        embedding_artifact_ids=[artifact_id],
        embedding_provider=embedding,
    )

    assert result["status"] == "answered"
    assert result["retrieval_mode"] == "hybrid"
    assert result["candidate_analysis_run_id"]
    assert result["candidate_skill_run_id"]
    assert result["candidate_retrieval_run_id"]
    run_config = json.loads(
        project.db.execute(
            "SELECT config_json FROM analysis_runs WHERE id=?",
            (result["candidate_analysis_run_id"],),
        ).fetchone()["config_json"]
    )
    embedding_snapshot = run_config["embedding_configuration_snapshot"]
    assert embedding_snapshot["preset_name"] == "embedding"
    assert embedding_snapshot["provider_id"] == "cohere-embeddings"
    assert embedding_snapshot["model"] == "embedding-model"
    assert embedding_snapshot["endpoint_fingerprint"]
    assert embedding_snapshot["artifacts"][0]["artifact_id"] == artifact_id
    assert embedding_snapshot["artifacts"][0]["file_hash"]
    assert "fixture-embedding-secret" not in json.dumps(embedding_snapshot)
    retrieval = project.db.execute(
        "SELECT analysis_run_id,skill_run_id,scope_id,coverage_id "
        "FROM retrieval_runs WHERE id=?",
        (result["candidate_retrieval_run_id"],),
    ).fetchone()
    assert retrieval["analysis_run_id"] == result["candidate_analysis_run_id"]
    assert retrieval["skill_run_id"] == result["candidate_skill_run_id"]
    assert retrieval["scope_id"] == result["scope_snapshot_id"]
    assert retrieval["coverage_id"] == result["coverage_id"]
    retrieval_context = project.db.execute(
        "SELECT operation,config_json FROM retrieval_run_context WHERE retrieval_run_id=?",
        (result["candidate_retrieval_run_id"],),
    ).fetchone()
    assert retrieval_context["operation"] == "linked_retrieval"
    retrieval_config = json.loads(retrieval_context["config_json"])
    assert retrieval_config["operation"] == "hybrid_search_with_preset"
    assert retrieval_config["preset_name"] == "embedding"
    assert retrieval_config["provider_id"] == "cohere-embeddings"
    assert retrieval_config["model"] == "embedding-model"
    assert retrieval_config["artifact_ids"] == [artifact_id]
    assert retrieval_config["query_characters"] == len("What did the council do about migrants?")
    assert retrieval_config["query_sha256"]
    assert "What did the council do about migrants?" not in retrieval_context["config_json"]
    assert "fixture-embedding-secret" not in retrieval_context["config_json"]
    skill = project.db.execute(
        "SELECT skill_id,status,coverage_id FROM skill_runs WHERE id=?",
        (result["candidate_skill_run_id"],),
    ).fetchone()
    assert skill["skill_id"] == "builtin.scope-question-hybrid-candidates"
    assert skill["status"] == "COMPLETED"
    assert skill["coverage_id"] == result["coverage_id"]
    hits = project.db.execute(
        "SELECT s.revision_id FROM retrieval_hits h JOIN source_spans s "
        "ON s.id=h.source_span_id WHERE h.retrieval_run_id=?",
        (result["candidate_retrieval_run_id"],),
    ).fetchall()
    assert hits and {row["revision_id"] for row in hits} == {revision_id}
    assert embedding.query_texts == [["What did the council do about migrants?"]]
    assert result["evidence"]


def test_hybrid_ask_requires_complete_index_coverage_before_embedding(tmp_path, monkeypatch):
    project = _hybrid_project(tmp_path, monkeypatch, name="partial-index")
    first = project.import_text("first.txt", "The council welcomed migrants.")[1]
    second = project.import_text("second.txt", "The council discussed a public meeting.")[1]
    embedding = _EmbeddingFixture()
    first_index = build_paragraph_embedding_index(project, "embedding", first, embedding)

    with pytest.raises(ScopeViolationError, match="exactly one vector index"):
        project.ask_question(
            [first, second],
            "What did the council do?",
            "fairclough-text",
            "review",
            retrieval_mode="hybrid",
            embedding_preset_name="embedding",
            embedding_artifact_ids=[first_index],
            embedding_provider=embedding,
        )

    assert embedding.query_texts == []
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retrieval_mode": "hybrid"},
        {"retrieval_mode": "lexical", "embedding_preset_name": "embedding"},
        {
            "retrieval_mode": "hybrid",
            "embedding_preset_name": "embedding",
            "embedding_artifact_ids": ["same", "same"],
        },
        {
            "retrieval_mode": "hybrid",
            "embedding_preset_name": "embedding",
            "embedding_artifact_ids": [f"artifact-{index}" for index in range(101)],
        },
    ],
)
def test_ask_question_rejects_invalid_hybrid_configuration_before_run(
    tmp_path, monkeypatch, kwargs
):
    project = _hybrid_project(tmp_path, monkeypatch, name="invalid-config")
    revision_id = project.import_text("speech.txt", "The council welcomed migrants.")[1]
    with pytest.raises(ProjectError):
        project.ask_question(
            [revision_id],
            "What did the council do?",
            "fairclough-text",
            "review",
            **kwargs,
        )
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
