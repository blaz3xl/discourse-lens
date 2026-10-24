import json
import os
import shutil
import sqlite3
import stat
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from discourselens import Project
from discourselens.artifact_integrity import sha256_file
from discourselens.errors import (
    ArtifactIntegrityError,
    DocumentError,
    ModelProviderError,
    ModelResponseError,
    ProjectError,
    ScopeViolationError,
)
from discourselens.models import (
    CredentialReference,
    EmbeddingProvider,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
)
from discourselens.providers import GeminiEmbeddingProvider, JinaEmbeddingProvider
from discourselens.reporting import analysis_payload
from discourselens.secrets import NO_AUTH_REFERENCE, EnvironmentSecretStore
from discourselens.types import CoverageDescriptor, CoverageMode
from discourselens.vectors import (
    VectorChunk,
    _connection,
    _embedding_batch_size,
    _embedding_endpoint_fingerprint,
    _validate_built_vector_index,
    build_paragraph_embedding_index,
    build_vector_index,
    hybrid_search,
    hybrid_search_with_preset,
    preview_paragraph_embedding_request,
    search_vectors,
    search_vectors_with_provenance,
    semantic_search_with_preset,
)


class _EmbeddingFixture:
    provider_id = "openai-compatible"
    endpoint = "http://local.test/v1"

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        assert model == "embedding-model"
        return [(float(index + 1), 0.0) for index, _ in enumerate(texts)]


class _QueryAwareEmbeddingFixture(_EmbeddingFixture):
    provider_id = "cohere-embeddings"
    endpoint = "https://api.cohere.com"

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint or type(self).endpoint
        self.query_requests: list[tuple[str, list[str]]] = []

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.query_requests.append((model, texts))
        return [(0.1, 0.0) for _ in texts]


class _JinaQueryAwareEmbeddingFixture(_QueryAwareEmbeddingFixture):
    provider_id = "jina-embeddings"
    endpoint = "https://api.jina.ai/v1"

    def __init__(self):
        super().__init__(endpoint=type(self).endpoint)
        self.document_requests: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.document_requests.append((model, texts))
        return super().embed(model, texts)


class _EmptyQueryEmbeddingFixture(_QueryAwareEmbeddingFixture):
    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.query_requests.append((model, texts))
        return []


class _EmptyEmbeddingFixture(_EmbeddingFixture):
    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        return []


class _BatchingEmbeddingFixture:
    provider_id = "openai-compatible"
    endpoint = "http://local.test/v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        assert model == "embedding-model"
        self.calls.append(list(texts))
        return [
            (float(int(text.removeprefix("Paragraph ").removesuffix("."))), 0.0)
            for text in texts
        ]


class _RebuildEmbeddingFixture:
    provider_id = "cohere-embeddings"
    endpoint = "https://api.cohere.com"

    def __init__(self):
        self.document_models: list[str] = []
        self.query_models: list[str] = []

    @staticmethod
    def _vector(model: str) -> tuple[float, float]:
        return (0.0, 1.0) if model == "embed-v2" else (1.0, 0.0)

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.document_models.append(model)
        return [self._vector(model) for _ in texts]

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.query_models.append(model)
        return [self._vector(model) for _ in texts]


def _rewrite_vector_artifact_as_legacy_l2(
    project: Project,
    artifact_id: str,
    chunks: list[VectorChunk],
    *,
    schema_version: str,
) -> None:
    """Give a fixture the default L2 vec0 table used by historical indexes."""
    artifact = project.db.execute(
        "SELECT relative_path FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone()
    assert artifact is not None
    artifact_path = project.hidden / artifact["relative_path"]
    connection = _connection(artifact_path)
    try:
        connection.execute("DROP TABLE vectors")
        dimension = len(chunks[0].embedding)
        connection.execute(
            f"CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[{dimension}])"
        )
        for chunk in chunks:
            connection.execute(
                "INSERT INTO vectors VALUES(?,?)",
                (chunk.chunk_id, json.dumps(chunk.embedding)),
            )
        connection.commit()
    finally:
        connection.close()
    project.db.execute(
        "UPDATE artifact_registry SET file_hash=?,schema_version=? WHERE id=?",
        (sha256_file(artifact_path), schema_version, artifact_id),
    )
    project.db.commit()


def test_vector_search_enforces_scope_before_ranking(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision_a = project.import_text("a.txt", "Apples are green.")
    _, revision_b = project.import_text("b.txt", "Elephants live in Africa.")
    artifact_a = build_vector_index(project, revision_a, [VectorChunk("a", 0, 6, (0.1, 0.0))])
    artifact_b = build_vector_index(project, revision_b, [VectorChunk("b", 0, 9, (0.01, 0.0))])

    indexes = project.vector_index_artifacts(revision_a)
    assert len(indexes) == 1
    assert indexes[0]["artifact_id"] == artifact_a
    assert indexes[0]["revision_id"] == revision_a
    assert "relative_path" not in indexes[0]

    hits = search_vectors(project, project.snapshot([revision_a]), [artifact_a], (1.0, 0.0))

    assert [hit.revision_id for hit in hits] == [revision_a]
    with pytest.raises(ScopeViolationError):
        search_vectors(project, project.snapshot([revision_a]), [artifact_b], (1.0, 0.0))


def test_cosine_metric_ranks_unequal_norms_and_legacy_l2_metadata_is_truthful(tmp_path):
    project = Project.init(tmp_path / "vector-distance-metrics")
    text = "First candidate. Second candidate."
    _, cosine_revision = project.import_text("cosine.txt", text)
    _, legacy_revision = project.import_text("legacy.txt", text)
    chunks = [
        VectorChunk("high-norm-aligned", 0, 5, (10.0, 1.0)),
        VectorChunk("low-norm-near", 6, 12, (1.0, 0.5)),
    ]
    cosine_artifact = build_vector_index(project, cosine_revision, chunks)
    endpoint_fingerprint = _embedding_endpoint_fingerprint(
        "cohere-embeddings", "lab", "DL_COHERE", "https://api.cohere.com"
    )
    legacy_artifact = build_vector_index(
        project,
        legacy_revision,
        chunks,
        embedding_provider_id="cohere-embeddings",
        embedding_model="embedding-model",
        chunker_version="paragraph-v1",
        embedding_input_type="search_document",
        embedding_endpoint_fingerprint=endpoint_fingerprint,
    )
    _rewrite_vector_artifact_as_legacy_l2(
        project, legacy_artifact, chunks, schema_version="vector-index-v3:2"
    )

    cosine_hits = search_vectors(
        project, project.snapshot([cosine_revision]), [cosine_artifact], (1.0, 0.0)
    )
    legacy_scope = project.snapshot([legacy_revision])
    l2_hits = search_vectors(project, legacy_scope, [legacy_artifact], (1.0, 0.0))
    assert [hit.chunk_id for hit in cosine_hits] == [
        "high-norm-aligned",
        "low-norm-near",
    ]
    assert [hit.chunk_id for hit in l2_hits] == ["low-norm-near", "high-norm-aligned"]

    legacy_result = search_vectors_with_provenance(
        project, legacy_scope, [legacy_artifact], (1.0, 0.0)
    )
    legacy_run = project.db.execute(
        "SELECT config_json FROM retrieval_run_context WHERE retrieval_run_id=?",
        (legacy_result.retrieval_run_id,),
    ).fetchone()
    assert json.loads(legacy_run["config_json"])["score_definition"] == (
        "sqlite-vec squared-L2 score; lower is nearer; score is an ordering value, "
        "not a calibrated similarity"
    )
    legacy_policy = project.db.execute(
        "SELECT selection_policy FROM retrieval_runs WHERE id=?",
        (legacy_result.retrieval_run_id,),
    ).fetchone()[0]
    assert "squared-L2-score top" in legacy_policy
    assert project.db.execute(
        "SELECT schema_version FROM artifact_registry WHERE id=?", (legacy_artifact,)
    ).fetchone()[0] == "vector-index-v3:2"

    with pytest.raises(ScopeViolationError, match="cannot mix L2 and cosine"):
        search_vectors(
            project,
            project.snapshot([cosine_revision, legacy_revision]),
            [cosine_artifact, legacy_artifact],
            (1.0, 0.0),
        )


def test_cosine_index_rejects_zero_norm_embeddings_before_artifact_creation(tmp_path):
    project = Project.init(tmp_path / "zero-cosine-embedding")
    _, revision = project.import_text("source.txt", "Apples are green.")

    with pytest.raises(ScopeViolationError, match="cosine vector index embeddings.*non-zero norm"):
        build_vector_index(
            project, revision, [VectorChunk("zero", 0, 6, (0.0, 0.0))]
        )

    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0
    assert not list((project.hidden / "artifacts" / "vectors").glob("*.sqlite"))


def test_cosine_search_rejects_zero_query_but_legacy_l2_allows_zero_vectors(tmp_path):
    project = Project.init(tmp_path / "zero-vector-search")
    _, cosine_revision = project.import_text("cosine.txt", "Apples are green.")
    cosine_artifact = build_vector_index(
        project, cosine_revision, [VectorChunk("cosine", 0, 6, (0.1, 0.0))]
    )
    cosine_scope = project.snapshot([cosine_revision])
    with pytest.raises(ScopeViolationError, match="cosine vector search requires a non-zero query"):
        search_vectors(project, cosine_scope, [cosine_artifact], (0.0, 0.0))

    _, legacy_revision = project.import_text("legacy.txt", "Apples are green.")
    legacy_chunks = [VectorChunk("legacy-zero", 0, 6, (0.0, 0.0))]
    legacy_artifact = build_vector_index(
        project,
        legacy_revision,
        [VectorChunk("legacy-zero", 0, 6, (0.1, 0.0))],
        embedding_provider_id="fixture",
        embedding_model="model",
        chunker_version="paragraph-v1",
        embedding_input_type="symmetric",
    )
    _rewrite_vector_artifact_as_legacy_l2(
        project, legacy_artifact, legacy_chunks, schema_version="vector-index-v2:2"
    )

    legacy_hits = search_vectors(
        project,
        project.snapshot([legacy_revision]),
        [legacy_artifact],
        (0.0, 0.0),
    )
    assert len(legacy_hits) == 1
    assert legacy_hits[0].distance == 0.0


def test_v4_raw_search_allows_missing_endpoint_but_presets_reject_before_embedding(
    tmp_path,
):
    project = Project.init(tmp_path / "optional-vector-endpoint")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )
    artifact_id = build_vector_index(
        project,
        revision,
        [VectorChunk("source", 0, 6, (0.1, 0.0))],
        embedding_provider_id="cohere-embeddings",
        embedding_model="embedding-model",
        chunker_version="paragraph-v1",
        embedding_input_type="search_document",
    )
    scope_id = project.snapshot([revision])

    assert search_vectors(project, scope_id, [artifact_id], (1.0, 0.0))
    provider = _QueryAwareEmbeddingFixture()
    with pytest.raises(ScopeViolationError, match="endpoint provenance is missing or incompatible"):
        semantic_search_with_preset(
            project, "embed", scope_id, [artifact_id], "green fruit", provider
        )
    with pytest.raises(ScopeViolationError, match="endpoint provenance is missing or incompatible"):
        hybrid_search_with_preset(
            project, "embed", scope_id, [artifact_id], "green fruit", provider
        )
    assert provider.query_requests == []


def test_vector_artifact_metadata_lookups_batch_below_sqlite_variable_limit(tmp_path):
    project = Project.init(tmp_path / "large-vector-artifact-selection")
    _, revision = project.import_text("source.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("source", 0, 6, (0.1, 0.0))]
    )
    source_row = dict(
        project.db.execute("SELECT * FROM artifact_registry WHERE id=?", (artifact_id,)).fetchone()
    )
    columns = tuple(source_row)
    artifact_ids = [artifact_id]
    rows = []
    for index in range(1_000):
        row = dict(source_row)
        row["id"] = f"duplicate-revision-index-{index}"
        artifact_ids.append(row["id"])
        rows.append(tuple(row[column] for column in columns))
    placeholders = ",".join("?" for _ in columns)
    project.db.executemany(
        f"INSERT INTO artifact_registry ({','.join(columns)}) VALUES ({placeholders})", rows
    )
    project.db.commit()
    project.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)

    with pytest.raises(ScopeViolationError, match="at most one vector artifact per scoped revision"):
        hybrid_search(project, scope_id, artifact_ids, "Apples", (0.1, 0.0))


def test_raw_search_rejects_overlapping_indexes_for_one_revision_before_candidates(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "duplicate-revision-indexes")
    _, revision = project.import_text(
        "source.txt", "Apples are green.\n\nOranges are ripe."
    )
    chunks = [VectorChunk("overlapping-chunk", 0, len(project.revision_text(revision)), (1.0, 0.0))]
    older_index = build_vector_index(project, revision, chunks)
    newer_index = build_vector_index(project, revision, chunks)
    scope_id = project.snapshot([revision])

    def forbidden_index_query(_path):
        pytest.fail("duplicate revision indexes must be rejected before vector candidates")

    monkeypatch.setattr("discourselens.vectors._connection", forbidden_index_query)
    with pytest.raises(ScopeViolationError, match="at most one vector artifact per scoped revision"):
        search_vectors(project, scope_id, [older_index, newer_index], (1.0, 0.0), limit=2)

    def forbidden_lexical_query(*_args, **_kwargs):
        pytest.fail("duplicate revision indexes must be rejected before lexical candidates")

    monkeypatch.setattr(project, "search_exact_ranked", forbidden_lexical_query)
    with pytest.raises(ScopeViolationError, match="at most one vector artifact per scoped revision"):
        hybrid_search(
            project,
            scope_id,
            [older_index, newer_index],
            "Apples",
            (1.0, 0.0),
            limit=2,
        )


def test_raw_hybrid_search_accepts_one_index_per_multiple_revisions(tmp_path):
    project = Project.init(tmp_path / "multi-revision-hybrid-indexes")
    _, first_revision = project.import_text("first.txt", "Apples are green.")
    _, second_revision = project.import_text("second.txt", "Apples are ripe.")
    indexes = []
    for revision_id in (first_revision, second_revision):
        paragraph = project.db.execute(
            "SELECT id,start_offset,end_offset FROM paragraphs WHERE revision_id=?",
            (revision_id,),
        ).fetchone()
        indexes.append(
            build_vector_index(
                project,
                revision_id,
                [
                    VectorChunk(
                        paragraph["id"],
                        paragraph["start_offset"],
                        paragraph["end_offset"],
                        (1.0, 0.0),
                    )
                ],
            )
        )

    hits = hybrid_search(
        project,
        project.snapshot([first_revision, second_revision]),
        indexes,
        "Apples",
        (1.0, 0.0),
        limit=2,
    )

    assert {hit.revision_id for hit in hits} == {first_revision, second_revision}
    assert all(hit.vector_rank is not None for hit in hits)


def test_vector_search_rejects_closer_out_of_scope_index_before_querying_candidates(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "p")
    _, selected_revision = project.import_text(
        "selected.txt", "The selected passage is intentionally less similar."
    )
    _, outside_revision = project.import_text(
        "outside.txt", "This passage is the closest vector candidate."
    )
    selected_artifact = build_vector_index(
        project,
        selected_revision,
        [VectorChunk("selected-candidate", 0, 8, (0.0, 1.0))],
    )
    outside_artifact = build_vector_index(
        project,
        outside_revision,
        [VectorChunk("outside-closest-candidate", 0, 8, (0.99, 0.01))],
    )
    all_revisions = project.snapshot([selected_revision, outside_revision])
    unrestricted_hits = search_vectors(
        project,
        all_revisions,
        [selected_artifact, outside_artifact],
        (1.0, 0.0),
    )
    assert unrestricted_hits[0].revision_id == outside_revision

    queried_indexes = []
    def record_index_query(path):
        queried_indexes.append(path)
        return _connection(path)

    def forbidden_artifact_verification(*_args, **_kwargs):
        pytest.fail("out-of-scope indexes must be rejected before artifact verification")

    monkeypatch.setattr("discourselens.vectors._connection", record_index_query)
    monkeypatch.setattr(
        "discourselens.vectors.verify_project_artifact", forbidden_artifact_verification
    )
    selected_scope = project.snapshot([selected_revision])
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        search_vectors(
            project,
            selected_scope,
            [selected_artifact, outside_artifact],
            (1.0, 0.0),
        )

    assert queried_indexes == []


def test_vector_search_rejects_indexes_with_incompatible_provenance(tmp_path):
    project = Project.init(tmp_path / "incompatible-vector-spaces")
    _, revision_a = project.import_text("a.txt", "Apples are green.")
    _, revision_b = project.import_text("b.txt", "Elephants live nearby.")
    artifact_a = build_vector_index(
        project,
        revision_a,
        [VectorChunk("a", 0, 6, (0.1, 0.0))],
        embedding_provider_id="fixture-a",
        embedding_model="model-a",
        chunker_version="paragraph-v1",
        embedding_input_type="symmetric",
    )
    artifact_b = build_vector_index(
        project,
        revision_b,
        [VectorChunk("b", 0, 9, (0.01, 0.0))],
        embedding_provider_id="fixture-b",
        embedding_model="model-b",
        chunker_version="paragraph-v1",
        embedding_input_type="symmetric",
    )

    with pytest.raises(ScopeViolationError, match="incompatible embedding or chunker provenance"):
        search_vectors(
            project,
            project.snapshot([revision_a, revision_b]),
            [artifact_a, artifact_b],
            (0.0, 0.0),
        )


def test_raw_vector_search_rejects_indexes_from_distinct_embedding_endpoints(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "distinct-embedding-endpoints")
    _, revision_a = project.import_text("a.txt", "Apples are green.")
    _, revision_b = project.import_text("b.txt", "Oranges are ripe.")
    common_provenance = {
        "embedding_provider_id": "cohere-embeddings",
        "embedding_model": "embedding-model",
        "chunker_version": "paragraph-v1",
        "embedding_input_type": "search_document",
    }
    endpoint_a = _embedding_endpoint_fingerprint(
        "cohere-embeddings",
        "lab",
        "DL_COHERE",
        "https://embedding-a.example.test/v1",
    )
    endpoint_b = _embedding_endpoint_fingerprint(
        "cohere-embeddings",
        "lab",
        "DL_COHERE",
        "https://embedding-b.example.test/v1",
    )
    artifact_a = build_vector_index(
        project,
        revision_a,
        [VectorChunk("a", 0, 6, (0.9, 0.1))],
        embedding_endpoint_fingerprint=endpoint_a,
        **common_provenance,
    )
    artifact_b = build_vector_index(
        project,
        revision_b,
        [VectorChunk("b", 0, 7, (0.1, 0.9))],
        embedding_endpoint_fingerprint=endpoint_b,
        **common_provenance,
    )
    scope_id = project.snapshot([revision_a, revision_b])

    def forbidden_candidate_query(_path):
        pytest.fail("mixed endpoint indexes must be rejected before vector candidates")

    monkeypatch.setattr("discourselens.vectors._connection", forbidden_candidate_query)
    with pytest.raises(ScopeViolationError, match="incompatible embedding or chunker provenance"):
        search_vectors(project, scope_id, [artifact_a, artifact_b], (1.0, 0.0))

    with pytest.raises(ScopeViolationError, match="incompatible embedding or chunker provenance"):
        hybrid_search(
            project,
            scope_id,
            [artifact_a, artifact_b],
            "Apples",
            (1.0, 0.0),
        )


def test_vector_search_respects_document_corpus_subcorpus_and_metadata_scopes(tmp_path):
    project = Project.init(tmp_path / "p")
    document_a, revision_a = project.import_text("a.txt", "Apples are green.")
    document_b, revision_b = project.import_text("b.txt", "Oranges are orange.")
    document_c, revision_c = project.import_text("c.txt", "Elephants live nearby.")
    project.set_document_metadata(document_a, "group", "selected")
    project.set_document_metadata(document_b, "group", "other")
    project.set_document_metadata(document_c, "group", "other")
    artifact_a = build_vector_index(
        project, revision_a, [VectorChunk("a", 0, 6, (0.3, 0.0))]
    )
    artifact_b = build_vector_index(
        project, revision_b, [VectorChunk("b", 0, 7, (0.2, 0.0))]
    )
    artifact_c = build_vector_index(
        project, revision_c, [VectorChunk("c", 0, 9, (0.1, 0.0))]
    )
    corpus_id = project.create_corpus("two documents")
    project.add_revision_to_corpus(corpus_id, revision_a)
    project.add_revision_to_corpus(corpus_id, revision_b)
    subcorpus_id = project.create_subcorpus(corpus_id, "second only", [revision_b])

    document_scope = project.snapshot([revision_a])
    corpus_scope = project.corpus_snapshot(corpus_id)
    subcorpus_scope = project.subcorpus_snapshot(subcorpus_id)
    metadata_scope, _ = project.snapshot_by_document_metadata({"group": "selected"})

    assert [
        hit.revision_id
        for hit in search_vectors(project, document_scope, [artifact_a], (1.0, 0.0))
    ] == [revision_a]
    assert {
        hit.revision_id
        for hit in search_vectors(
            project, corpus_scope, [artifact_a, artifact_b], (1.0, 0.0)
        )
    } == {revision_a, revision_b}
    assert [
        hit.revision_id
        for hit in search_vectors(project, subcorpus_scope, [artifact_b], (1.0, 0.0))
    ] == [revision_b]
    assert [
        hit.revision_id
        for hit in search_vectors(project, metadata_scope, [artifact_a], (1.0, 0.0))
    ] == [revision_a]

    with pytest.raises(ScopeViolationError):
        search_vectors(project, corpus_scope, [artifact_c], (1.0, 0.0))
    with pytest.raises(ScopeViolationError):
        search_vectors(project, subcorpus_scope, [artifact_a], (1.0, 0.0))
    with pytest.raises(ScopeViolationError):
        search_vectors(project, metadata_scope, [artifact_b], (1.0, 0.0))


def test_vector_metadata_scope_stays_frozen_across_updates_and_revision_removal(tmp_path):
    project = Project.init(tmp_path / "p")
    document_id, original_revision = project.import_text(
        "a.txt", "The original selected document."
    )
    project.set_document_metadata(document_id, "group", "selected")
    scope_id, frozen_revisions = project.snapshot_by_document_metadata(
        {"group": "selected"}
    )
    original_artifact = build_vector_index(
        project,
        original_revision,
        [VectorChunk("original-candidate", 0, 8, (0.4, 0.0))],
    )

    project.set_document_metadata(document_id, "group", "excluded")
    _, latest_revision = project.import_text(
        "a.txt",
        "A new excluded document revision.",
        document_id=document_id,
    )
    latest_artifact = build_vector_index(
        project,
        latest_revision,
        [VectorChunk("latest-candidate", 0, 5, (0.1, 0.0))],
    )

    assert frozen_revisions == [original_revision]
    assert [
        hit.revision_id
        for hit in search_vectors(project, scope_id, [original_artifact], (1.0, 0.0))
    ] == [original_revision]
    with pytest.raises(ScopeViolationError):
        search_vectors(project, scope_id, [latest_artifact], (1.0, 0.0))
    with pytest.raises(ScopeViolationError, match="matched no document revisions"):
        project.snapshot_by_document_metadata({"group": "selected"})

    # Simulate removal of a revision still named by an immutable historical scope.
    project.db.execute("DELETE FROM document_revisions WHERE id=?", (original_revision,))
    project.db.commit()
    with pytest.raises(ScopeViolationError, match="unknown project revision"):
        search_vectors(project, scope_id, [original_artifact], (1.0, 0.0))


def test_vector_search_rejects_registry_revision_rebinding(tmp_path):
    project = Project.init(tmp_path / "p")
    _, selected_revision = project.import_text("selected.txt", "Apples are green.")
    _, outside_revision = project.import_text("outside.txt", "Elephants live nearby.")
    outside_artifact = build_vector_index(
        project,
        outside_revision,
        [VectorChunk("outside-only-candidate", 0, 9, (0.01, 0.0))],
    )
    project.db.execute(
        "UPDATE artifact_registry SET revision_id=? WHERE id=?",
        (selected_revision, outside_artifact),
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="identity|revision"):
        search_vectors(
            project,
            project.snapshot([selected_revision]),
            [outside_artifact],
            (0.0, 0.0),
        )


def test_vector_search_rejects_malformed_scope_revision_data(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("chunk-a", 0, 6, (0.1, 0.0))]
    )
    scope_id = project.snapshot([revision])
    with pytest.raises(sqlite3.IntegrityError, match="ScopeSnapshot records are immutable"):
        project.db.execute(
            "UPDATE scope_snapshots SET revision_ids_json=? WHERE id=?",
            (json.dumps([revision, 123]), scope_id),
        )
    # Simulate damaged persisted data only after confirming the normal immutability guard.
    project.db.execute("DROP TRIGGER scope_snapshots_immutable_update")
    project.db.execute(
        "UPDATE scope_snapshots SET revision_ids_json=? WHERE id=?",
        (json.dumps([revision, 123]), scope_id),
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="scope"):
        search_vectors(project, scope_id, [artifact_id], (0.0, 0.0))


def test_vector_search_rejects_index_identity_from_another_project(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("chunk-a", 0, 6, (0.1, 0.0))]
    )
    registry_row = project.db.execute(
        "SELECT relative_path FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone()
    artifact_path = project.hidden / registry_row["relative_path"]
    connection = _connection(artifact_path)
    try:
        connection.execute(
            "UPDATE vector_index_identity SET project_id=?", ("another-project",)
        )
        connection.commit()
    finally:
        connection.close()
    project.db.execute(
        "UPDATE artifact_registry SET file_hash=? WHERE id=?",
        (sha256_file(artifact_path), artifact_id),
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="identity"):
        search_vectors(project, project.snapshot([revision]), [artifact_id], (1.0, 0.0))


def test_vector_search_rejects_non_vector_artifact_within_scope(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = project.uid()
    project.db.execute(
        "INSERT INTO artifact_registry "
        "(id,revision_id,artifact_type,backend,schema_version,content_hash,relative_path,created_at) "
        "VALUES(?,?,?,?,?,?,?,datetime('now'))",
        (
            artifact_id,
            revision,
            "surface_tokens",
            "fixture",
            "tokens-v1",
            "derived",
            "artifacts/nlp/not-a-vector.parquet",
        ),
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="not a vector index"):
        search_vectors(project, project.snapshot([revision]), [artifact_id], (0.0, 0.0))


def test_vector_search_rejects_duplicate_artifact_ids(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("chunk-a", 0, 6, (0.1, 0.0))]
    )

    with pytest.raises(ScopeViolationError, match="must not be duplicated"):
        search_vectors(
            project,
            project.snapshot([revision]),
            [artifact_id, artifact_id],
            (0.0, 0.0),
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [(0.5, 6), (0, 6.5), (True, 6), (0, True)],
)
def test_vector_index_rejects_noninteger_chunk_offsets_before_persistence(
    tmp_path, start, end
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")

    with pytest.raises(ScopeViolationError, match="offsets must be integers"):
        build_vector_index(
            project,
            revision,
            [VectorChunk("chunk-a", start, end, (0.1, 0.0))],  # type: ignore[arg-type]
        )

    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0
    assert not list((project.hidden / "artifacts" / "vectors").glob("*.sqlite"))


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf"), 1e300, 5e-324, True],
)
def test_vector_index_rejects_non_finite_or_boolean_components_before_persistence(
    tmp_path, invalid_value
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")

    with pytest.raises(ScopeViolationError, match="finite numeric values"):
        build_vector_index(
            project,
            revision,
            [VectorChunk("chunk-a", 0, 6, (invalid_value, 0.0))],  # type: ignore[arg-type]
        )

    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0
    assert not list((project.hidden / "artifacts" / "vectors").glob("*.sqlite"))


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf"), 1e300, 5e-324, False],
)
def test_vector_search_rejects_non_finite_or_boolean_query_components(
    tmp_path, invalid_value
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("chunk-a", 0, 6, (0.1, 0.0))]
    )

    with pytest.raises(ScopeViolationError, match="finite numeric values"):
        search_vectors(
            project,
            project.snapshot([revision]),
            [artifact_id],
            (invalid_value, 0.0),  # type: ignore[arg-type]
        )


def test_vector_search_rejects_legacy_index_with_float32_infinity(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))]
    )
    registry_row = project.db.execute(
        "SELECT relative_path FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone()
    artifact_path = project.hidden / registry_row["relative_path"]
    connection = _connection(artifact_path)
    try:
        connection.execute(
            "UPDATE vectors SET embedding=? WHERE id=?", (json.dumps((1e300, 0.0)), "a")
        )
        connection.commit()
    finally:
        connection.close()
    project.db.execute(
        "UPDATE artifact_registry SET file_hash=? WHERE id=?",
        (sha256_file(artifact_path), artifact_id),
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="malformed distance"):
        search_vectors(project, project.snapshot([revision]), [artifact_id], (1.0, 0.0))


def test_vector_search_rejects_persisted_offsets_outside_canonical_revision(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))]
    )
    registry_row = project.db.execute(
        "SELECT relative_path FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone()
    artifact_path = project.hidden / registry_row["relative_path"]
    connection = _connection(artifact_path)
    try:
        connection.execute("UPDATE chunks SET end_offset=? WHERE id=?", (10_000, "a"))
        connection.commit()
    finally:
        connection.close()
    project.db.execute(
        "UPDATE artifact_registry SET file_hash=? WHERE id=?",
        (sha256_file(artifact_path), artifact_id),
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="malformed offsets"):
        search_vectors(project, project.snapshot([revision]), [artifact_id], (1.0, 0.0))


def test_vector_index_file_integrity_is_checked_before_search(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])
    row = project.db.execute(
        "SELECT relative_path,file_hash FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone()
    assert row["file_hash"]
    (project.hidden / row["relative_path"]).write_bytes(b"corrupted index")

    with pytest.raises(ArtifactIntegrityError, match="integrity check"):
        search_vectors(project, project.snapshot([revision]), [artifact_id], (0.0, 0.0))


def test_vector_search_checks_artifact_metric_ddl_against_registered_schema(tmp_path):
    project = Project.init(tmp_path / "vector-metric-ddl-mismatch")
    _, revision = project.import_text("source.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("chunk", 0, 6, (0.1, 0.0))]
    )
    row = project.db.execute(
        "SELECT relative_path FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone()
    artifact_path = project.hidden / row["relative_path"]
    connection = _connection(artifact_path)
    try:
        connection.execute("DROP TABLE vectors")
        connection.execute(
            "CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[2])"
        )
        connection.execute(
            "INSERT INTO vectors VALUES(?,?)", ("chunk", json.dumps((0.1, 0.0)))
        )
        connection.commit()
    finally:
        connection.close()
    project.db.execute(
        "UPDATE artifact_registry SET file_hash=? WHERE id=?",
        (sha256_file(artifact_path), artifact_id),
    )
    project.db.commit()

    with pytest.raises(
        ArtifactIntegrityError, match="metric does not match its schema version"
    ):
        search_vectors(project, project.snapshot([revision]), [artifact_id], (1.0, 0.0))


@pytest.mark.parametrize("search_kind", ["semantic", "hybrid"])
def test_preset_search_checks_metric_ddl_before_query_embedding(tmp_path, search_kind):
    project = Project.init(tmp_path / f"preset-metric-ddl-mismatch-{search_kind}")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )
    provider = _QueryAwareEmbeddingFixture()
    artifact_id = build_paragraph_embedding_index(project, "embed", revision, provider)
    row = project.db.execute(
        "SELECT relative_path FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone()
    artifact_path = project.hidden / row["relative_path"]
    connection = _connection(artifact_path)
    try:
        vectors = connection.execute("SELECT id,embedding FROM vectors").fetchall()
        connection.execute("DROP TABLE vectors")
        connection.execute(
            "CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[2])"
        )
        connection.executemany(
            "INSERT INTO vectors VALUES(?,?)",
            [(item["id"], item["embedding"]) for item in vectors],
        )
        connection.commit()
    finally:
        connection.close()
    project.db.execute(
        "UPDATE artifact_registry SET file_hash=? WHERE id=?",
        (sha256_file(artifact_path), artifact_id),
    )
    project.db.commit()
    scope_id = project.snapshot([revision])

    with pytest.raises(
        ArtifactIntegrityError, match="metric does not match its schema version"
    ):
        if search_kind == "semantic":
            semantic_search_with_preset(
                project, "embed", scope_id, [artifact_id], "green fruit", provider
            )
        else:
            hybrid_search_with_preset(
                project, "embed", scope_id, [artifact_id], "green fruit", provider
            )

    assert provider.query_requests == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode permissions are not portable to Windows")
def test_vector_index_is_created_with_private_file_permissions(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))]
    )
    row = project.db.execute(
        "SELECT relative_path FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone()

    mode = stat.S_IMODE((project.hidden / row["relative_path"]).stat().st_mode)

    assert mode == 0o600


def test_vector_index_path_cannot_escape_project_root(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])
    project.db.execute(
        "UPDATE artifact_registry SET relative_path=? WHERE id=?",
        ("../../../../etc/passwd", artifact_id),
    )
    project.db.commit()

    with pytest.raises(ArtifactIntegrityError, match="path is invalid"):
        search_vectors(project, project.snapshot([revision]), [artifact_id], (0.0, 0.0))


def test_vector_index_rejects_symlinked_storage_replaced_after_open(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    vector_directory = project.hidden / "artifacts" / "vectors"
    preserved_directory = vector_directory.with_name("vectors-preserved")
    vector_directory.rename(preserved_directory)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        vector_directory.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ArtifactIntegrityError, match="vector index path"):
        build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])

    assert list(outside.iterdir()) == []
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0


def test_vector_build_removes_unregistered_file_when_hashing_fails(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")

    def fail_hash(_path):
        raise OSError("fixture hash failure")

    monkeypatch.setattr("discourselens.vectors.sha256_file", fail_hash)
    with pytest.raises(OSError, match="fixture hash failure"):
        build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])

    assert not list((project.hidden / "artifacts" / "vectors").glob("*.sqlite"))
    assert not list((project.hidden / "artifacts" / "vectors").glob(".*.building.sqlite"))
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0


def test_vector_build_serialization_failure_removes_staging_file(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "serialization-failure")
    _, revision = project.import_text("a.txt", "Apples are green.")
    original_dumps = json.dumps

    def fail_embedding_serialization(value, *args, **kwargs):
        if value == (0.1, 0.0):
            raise TypeError("fixture vector serialization failure")
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr("discourselens.vectors.json.dumps", fail_embedding_serialization)
    with pytest.raises(TypeError, match="fixture vector serialization failure"):
        build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])

    vector_directory = project.hidden / "artifacts" / "vectors"
    assert list(vector_directory.iterdir()) == []
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0


def test_vector_build_integrity_validation_failure_never_publishes(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "integrity-failure")
    _, revision = project.import_text("a.txt", "Apples are green.")
    observed_staging_paths = []
    real_validator = _validate_built_vector_index

    def reject_staged_index(path, **kwargs):
        observed_staging_paths.append(path)
        assert path.exists()
        assert ".building.sqlite" in path.name
        assert not path.with_name(path.name[1:].removesuffix(".building.sqlite") + ".sqlite").exists()
        connection = _connection(path)
        try:
            connection.execute("DELETE FROM vectors")
            connection.commit()
        finally:
            connection.close()
        real_validator(path, **kwargs)

    monkeypatch.setattr("discourselens.vectors._validate_built_vector_index", reject_staged_index)
    with pytest.raises(ArtifactIntegrityError, match="embedding validation failed"):
        build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])

    vector_directory = project.hidden / "artifacts" / "vectors"
    assert len(observed_staging_paths) == 1
    assert not list(vector_directory.iterdir())
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0


def test_vector_registry_failure_rolls_back_only_just_published_artifact(tmp_path):
    project = Project.init(tmp_path / "registry-failure")
    _, revision = project.import_text("a.txt", "Apples are green.")
    existing_id = build_vector_index(
        project, revision, [VectorChunk("existing", 0, 6, (0.2, 0.0))]
    )
    existing_row = dict(
        project.db.execute("SELECT * FROM artifact_registry WHERE id=?", (existing_id,)).fetchone()
    )
    existing_path = project.hidden / existing_row["relative_path"]
    existing_bytes = existing_path.read_bytes()
    project.db.execute("PRAGMA foreign_keys=ON")
    project.db.execute("CREATE TABLE commit_guard_parent(id TEXT PRIMARY KEY)")
    project.db.execute(
        "CREATE TABLE commit_guard_child(parent_id TEXT, "
        "FOREIGN KEY(parent_id) REFERENCES commit_guard_parent(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    project.db.execute(
        "CREATE TRIGGER reject_vector_registry_commit "
        "AFTER INSERT ON artifact_registry WHEN NEW.artifact_type='vector_index' "
        "BEGIN INSERT INTO commit_guard_child(parent_id) VALUES('missing-parent'); END"
    )
    project.db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        build_vector_index(project, revision, [VectorChunk("new", 0, 6, (0.1, 0.0))])

    vector_directory = project.hidden / "artifacts" / "vectors"
    assert sorted(path.name for path in vector_directory.glob("*.sqlite")) == [existing_path.name]
    assert not list(vector_directory.glob(".*.building.sqlite"))
    assert existing_path.read_bytes() == existing_bytes
    assert dict(
        project.db.execute("SELECT * FROM artifact_registry WHERE id=?", (existing_id,)).fetchone()
    ) == existing_row
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 1


def test_vector_build_does_not_overwrite_competing_final_path(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "publication-race")
    _, revision = project.import_text("a.txt", "Apples are green.")
    real_link = os.link
    competing_paths = []

    def occupy_destination_before_publish(source, destination):
        destination_path = Path(destination)
        competing_paths.append(destination_path)
        destination_path.write_bytes(b"pre-existing immutable artifact")
        return real_link(source, destination)

    monkeypatch.setattr("discourselens.vectors.os.link", occupy_destination_before_publish)
    with pytest.raises(ArtifactIntegrityError, match="vector index path"):
        build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])

    assert len(competing_paths) == 1
    assert competing_paths[0].read_bytes() == b"pre-existing immutable artifact"
    assert not list(competing_paths[0].parent.glob(".*.building.sqlite"))
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0


def test_vector_build_cleans_publication_if_link_raises_after_create(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "publication-interrupt")
    _, revision = project.import_text("a.txt", "Apples are green.")
    real_link = os.link
    published_paths = []

    def link_then_raise(source, destination):
        published_paths.append(Path(destination))
        real_link(source, destination)
        raise OSError("fixture interruption after successful link")

    monkeypatch.setattr("discourselens.vectors.os.link", link_then_raise)
    with pytest.raises(ArtifactIntegrityError, match="could not be atomically published"):
        build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])

    vector_directory = project.hidden / "artifacts" / "vectors"
    assert len(published_paths) == 1
    assert not list(vector_directory.iterdir())
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0


def test_vector_search_rejects_malformed_registry_dimension_as_scope_error(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(
        project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))]
    )
    project.db.execute(
        "UPDATE artifact_registry SET schema_version=? WHERE id=?", ("broken", artifact_id)
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="invalid dimension metadata"):
        search_vectors(project, project.snapshot([revision]), [artifact_id], (0.0, 0.0))


def test_hashed_vector_index_remains_valid_after_project_copy(tmp_path):
    project = Project.init(tmp_path / "source")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact_id = build_vector_index(project, revision, [VectorChunk("a", 0, 6, (0.1, 0.0))])
    project.close()

    shutil.copytree(tmp_path / "source", tmp_path / "copied")
    copied = Project.open(tmp_path / "copied")
    try:
        hits = search_vectors(copied, copied.snapshot([revision]), [artifact_id], (0.1, 0.0))
        assert len(hits) == 1 and hits[0].revision_id == revision
    finally:
        copied.close()


def test_paragraph_embedding_index_uses_explicit_preset_and_paragraph_offsets(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.\nSecond paragraph.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    project.save_model_preset(ModelPreset("embed", "openai-compatible", "local", "embedding-model"))

    artifact = build_paragraph_embedding_index(project, "embed", revision, _EmbeddingFixture())

    hits = search_vectors(project, project.snapshot([revision]), [artifact], (0.1, 0.0))
    assert {hit.start for hit in hits} == {0, len("First paragraph.\n")}
    semantic_hits = semantic_search_with_preset(
        project, "embed", project.snapshot([revision]), [artifact], "query", _EmbeddingFixture()
    )
    assert {hit.chunk_id for hit in semantic_hits} == {hit.chunk_id for hit in hits}


def test_paragraph_embedding_index_rejects_noncanonical_paragraph_before_embedding(tmp_path):
    project = Project.init(tmp_path / "corrupt-paragraph")
    _, revision = project.import_text("source.txt", "Canonical paragraph text.")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "local", "DL_LOCAL", "http://local.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("embed", "openai-compatible", "local", "embedding-model")
    )

    class _CaptureEmbedding:
        provider_id = "openai-compatible"
        endpoint = "http://local.test/v1"

        def __init__(self):
            self.inputs: list[str] = []

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.inputs.extend(texts)
            return [(0.1, 0.2) for _ in texts]

    provider = _CaptureEmbedding()
    original_index = build_paragraph_embedding_index(project, "embed", revision, provider)
    assert provider.inputs == ["Canonical paragraph text."]
    provider.inputs.clear()
    project.db.execute(
        "UPDATE paragraphs SET text=? WHERE revision_id=?",
        ("non-canonical injected paragraph", revision),
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="canonical revision text"):
        build_paragraph_embedding_index(project, "embed", revision, provider)
    with pytest.raises(ScopeViolationError, match="canonical revision text"):
        preview_paragraph_embedding_request(project, "embed", revision)

    assert provider.inputs == []
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 1
    assert search_vectors(
        project, project.snapshot([revision]), [original_index], (0.1, 0.2)
    )


def test_exact_embedding_capability_gate_covers_preview_build_and_preset_queries(tmp_path):
    project = Project.init(tmp_path / "embedding-capability-gate")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "local", "DL_LOCAL", "http://local.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("embed", "openai-compatible", "local", "embedding-model")
    )

    class CapabilityEmbeddingFixture:
        provider_id = "openai-compatible"
        endpoint = "http://local.test/v1"

        def __init__(self):
            self.calls: list[tuple[str, list[str]]] = []

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.calls.append((model, list(texts)))
            return [(0.1, 0.2) for _ in texts]

    provider = CapabilityEmbeddingFixture()
    try:
        # No exact-model profile preserves the existing embedding flow.
        preview_paragraph_embedding_request(project, "embed", revision)
        artifact_id = build_paragraph_embedding_index(project, "embed", revision, provider)
        scope_id = project.snapshot([revision])
        semantic_search_with_preset(
            project, "embed", scope_id, [artifact_id], "green", provider
        )
        hybrid_search_with_preset(
            project, "embed", scope_id, [artifact_id], "Apples", provider
        )
        assert len(provider.calls) == 3

        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "openai-compatible", "embedding-model", ModelCapabilities(embeddings=False)
            )
        )
        with pytest.raises(
            ModelProviderError, match="disabled by the exact-model capability profile"
        ):
            preview_paragraph_embedding_request(project, "embed", revision)
        with pytest.raises(
            ModelProviderError, match="disabled by the exact-model capability profile"
        ):
            build_paragraph_embedding_index(project, "embed", revision, provider)
        with pytest.raises(
            ModelProviderError, match="disabled by the exact-model capability profile"
        ):
            semantic_search_with_preset(
                project, "embed", scope_id, [artifact_id], "green", provider
            )
        with pytest.raises(
            ModelProviderError, match="disabled by the exact-model capability profile"
        ):
            hybrid_search_with_preset(
                project, "embed", scope_id, [artifact_id], "Apples", provider
            )
        assert len(provider.calls) == 3

        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "openai-compatible", "embedding-model", ModelCapabilities(embeddings=True)
            )
        )
        preview_paragraph_embedding_request(project, "embed", revision)
        build_paragraph_embedding_index(project, "embed", revision, provider)
        semantic_search_with_preset(
            project, "embed", scope_id, [artifact_id], "green", provider
        )
        hybrid_search_with_preset(
            project, "embed", scope_id, [artifact_id], "Apples", provider
        )
        assert len(provider.calls) == 6
    finally:
        project.close()


def test_paragraph_embedding_build_and_preview_share_bounded_batches(tmp_path, monkeypatch):
    import discourselens.vectors as vectors_module

    project = Project.init(tmp_path / "p")
    paragraphs = [f"Paragraph {index:03d}." for index in range(1, 66)]
    _, revision = project.import_text("many-paragraphs.txt", "\n".join(paragraphs))
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "local", "DL_LOCAL", "http://local.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("embed", "openai-compatible", "local", "embedding-model")
    )
    provider = _BatchingEmbeddingFixture()
    captured_chunks: list[VectorChunk] = []
    original_build_vector_index = vectors_module.build_vector_index

    def capture_build_vector_index(project_arg, revision_arg, chunks, **kwargs):
        captured_chunks.extend(chunks)
        return original_build_vector_index(project_arg, revision_arg, chunks, **kwargs)

    monkeypatch.setattr(vectors_module, "build_vector_index", capture_build_vector_index)
    artifact_id = build_paragraph_embedding_index(project, "embed", revision, provider)

    rows = project.db.execute(
        "SELECT id,text FROM paragraphs WHERE revision_id=? ORDER BY start_offset,id",
        (revision,),
    ).fetchall()
    expected_ids = [row["id"] for row in rows]
    assert [len(batch) for batch in provider.calls] == [64, 1]
    assert provider.calls == [paragraphs[:64], paragraphs[64:]]
    assert [chunk.chunk_id for chunk in captured_chunks] == expected_ids
    assert [chunk.embedding for chunk in captured_chunks] == [
        (float(index), 0.0) for index in range(1, 66)
    ]
    indexed_artifact = project.db.execute(
        "SELECT embedding_provider_id,embedding_model,embedding_dimension,chunker_version "
        "FROM artifact_registry WHERE id=?",
        (artifact_id,),
    ).fetchone()
    assert tuple(indexed_artifact) == (
        "openai-compatible",
        "embedding-model",
        2,
        "plain-text-lines-v2",
    )
    hits = search_vectors(
        project, project.snapshot([revision]), [artifact_id], (65.0, 0.0), limit=1
    )
    assert len(hits) == 1 and hits[0].chunk_id == expected_ids[-1]

    preview = preview_paragraph_embedding_request(project, "embed", revision, limit=65)
    assert [len(batch.inputs) for batch in preview.batches] == [64, 1]
    assert [item.paragraph_id for batch in preview.batches for item in batch.inputs] == (
        expected_ids
    )


def test_blank_lines_are_not_persisted_as_empty_embedding_paragraphs(tmp_path):
    project = Project.init(tmp_path / "blank-lines")
    source = "First paragraph.\n\nCafé speech.\n   \nFinal paragraph.\n"
    _, revision = project.import_text("notes.txt", source)
    rows = project.db.execute(
        "SELECT start_offset,end_offset,text,segmentation_version FROM paragraphs "
        "WHERE revision_id=? ORDER BY ordinal",
        (revision,),
    ).fetchall()

    assert [row["text"] for row in rows] == [
        "First paragraph.",
        "Café speech.",
        "Final paragraph.",
    ]
    assert all(row["end_offset"] > row["start_offset"] for row in rows)
    assert all(
        source[row["start_offset"] : row["end_offset"]] == row["text"] for row in rows
    )
    assert {row["segmentation_version"] for row in rows} == {"plain-text-lines-v2"}

    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    project.save_model_preset(
        ModelPreset("embed", "openai-compatible", "local", "embedding-model")
    )
    artifact_id = build_paragraph_embedding_index(project, "embed", revision, _EmbeddingFixture())
    preview = preview_paragraph_embedding_request(project, "embed", revision)
    assert preview.total_paragraphs == 3
    assert project.vector_index_artifacts(revision)[0]["artifact_id"] == artifact_id


def test_embedding_batch_size_falls_back_without_weakening_declared_limit(monkeypatch):
    class AdapterWithoutBatchLimit:
        pass

    class AdapterWithStrictBatchLimit:
        MAX_INPUTS_PER_REQUEST = 12

    monkeypatch.setattr(
        "discourselens.vectors.GeminiEmbeddingProvider", AdapterWithoutBatchLimit
    )
    assert _embedding_batch_size("gemini-embeddings") == 64
    assert _embedding_batch_size("ollama") == 8

    monkeypatch.setattr("discourselens.vectors.CohereEmbeddingProvider", AdapterWithStrictBatchLimit)
    assert _embedding_batch_size("cohere-embeddings") == 12


def test_ollama_paragraph_embeddings_allow_loopback_no_auth(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.\nSecond paragraph.")
    project.save_credential_reference(
        CredentialReference("ollama", "local", NO_AUTH_REFERENCE, "http://localhost:11434/v1")
    )
    project.save_model_preset(ModelPreset("embed", "ollama", "local", "embedding-model"))
    response = MagicMock()
    response.read.return_value = json.dumps(
        {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    captured = {}

    def capture_index(project_arg, revision_arg, chunks, **kwargs):
        captured["revision_id"] = revision_arg
        captured["chunks"] = chunks
        captured["kwargs"] = kwargs
        return "fixture-index"

    monkeypatch.setattr("discourselens.vectors.build_vector_index", capture_index)
    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = build_paragraph_embedding_index(project, "embed", revision)

    assert result == "fixture-index"
    assert captured["revision_id"] == revision
    assert len(captured["chunks"]) == 2
    assert captured["kwargs"]["embedding_provider_id"] == "ollama"
    assert captured["kwargs"]["embedding_endpoint_fingerprint"].startswith(
        "embedding-endpoint-v1:"
    )
    assert (
        project.credential_reference("ollama", "local")["secret_reference"]
        == NO_AUTH_REFERENCE
    )
    assert opened.call_args.args[0].get_header("Authorization") is None
    assert opened.call_args.args[0].full_url == "http://localhost:11434/api/embed"
    assert json.loads(opened.call_args.args[0].data) == {
        "model": "embedding-model",
        "input": ["First paragraph.", "Second paragraph."],
        "truncate": False,
    }


def test_semantic_search_uses_query_aware_embedding_operation_when_available(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    project.save_credential_reference(CredentialReference("cohere-embeddings", "lab", "DL_COHERE"))
    project.save_model_preset(ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model"))
    provider = _QueryAwareEmbeddingFixture()
    artifact = build_paragraph_embedding_index(project, "embed", revision, provider)

    hits = semantic_search_with_preset(
        project,
        "embed",
        project.snapshot([revision]),
        [artifact],
        "green fruit",
        provider,
    )

    assert provider.query_requests == [("embedding-model", ["green fruit"])]
    assert [hit.revision_id for hit in hits] == [revision]


def test_semantic_search_rejects_model_mismatch_before_embedding_call(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    project.save_credential_reference(CredentialReference("cohere-embeddings", "lab", "DL_COHERE"))
    project.save_model_preset(
        ModelPreset("indexed-model", "cohere-embeddings", "lab", "embedding-model")
    )
    project.save_model_preset(
        ModelPreset("query-model", "cohere-embeddings", "lab", "different-model")
    )
    provider = _QueryAwareEmbeddingFixture()
    artifact = build_paragraph_embedding_index(project, "indexed-model", revision, provider)

    with pytest.raises(ScopeViolationError, match="does not match vector index provenance"):
        semantic_search_with_preset(
            project,
            "query-model",
            project.snapshot([revision]),
            [artifact],
            "green fruit",
            provider,
        )

    assert provider.query_requests == []


def test_semantic_search_rejects_injected_provider_mismatching_selected_preset(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference("jina-embeddings", "indexed", "DL_JINA")
    )
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "selected", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("jina-index", "jina-embeddings", "indexed", "embedding-model")
    )
    project.save_model_preset(
        ModelPreset("cohere-query", "cohere-embeddings", "selected", "embedding-model")
    )
    provider = _JinaQueryAwareEmbeddingFixture()
    artifact_id = build_paragraph_embedding_index(project, "jina-index", revision, provider)

    with pytest.raises(ScopeViolationError, match="does not match selected embedding preset"):
        semantic_search_with_preset(
            project,
            "cohere-query",
            project.snapshot([revision]),
            [artifact_id],
            "green fruit",
            provider,
        )

    assert provider.query_requests == []


def test_paragraph_embedding_index_rejects_injected_provider_mismatching_selected_preset(
    tmp_path,
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "selected", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("cohere-index", "cohere-embeddings", "selected", "embedding-model")
    )
    provider = _JinaQueryAwareEmbeddingFixture()

    with pytest.raises(ScopeViolationError, match="does not match selected embedding preset"):
        build_paragraph_embedding_index(project, "cohere-index", revision, provider)

    assert provider.document_requests == []
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0


def test_semantic_search_rejects_legacy_index_without_provenance_before_embedding_call(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    project.save_credential_reference(CredentialReference("cohere-embeddings", "lab", "DL_COHERE"))
    project.save_model_preset(ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model"))
    artifact = build_vector_index(project, revision, [VectorChunk("legacy", 0, 6, (0.1, 0.0))])
    provider = _QueryAwareEmbeddingFixture()

    with pytest.raises(ScopeViolationError, match="rebuild the index"):
        semantic_search_with_preset(
            project,
            "embed",
            project.snapshot([revision]),
            [artifact],
            "green fruit",
            provider,
        )

    assert provider.query_requests == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in (
            "embedding_provider_id",
            "embedding_model",
            "chunker_version",
            "embedding_input_type",
        )
        for value in ("", "   ")
    ],
)
def test_vector_index_rejects_empty_or_whitespace_provenance_before_artifact_creation(
    tmp_path, field, value
):
    project = Project.init(tmp_path / "blank-provenance")
    _, revision = project.import_text("source.txt", "Apples are green.")
    provenance = {
        "embedding_provider_id": "fixture",
        "embedding_model": "model",
        "chunker_version": "chunker-v1",
        "embedding_input_type": "symmetric",
    }
    provenance[field] = value

    with pytest.raises(ScopeViolationError, match="non-empty provider, model, chunker"):
        build_vector_index(
            project,
            revision,
            [VectorChunk("chunk", 0, 6, (0.1, 0.0))],
            **provenance,
        )

    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0
    assert not list((project.hidden / "artifacts" / "vectors").glob("*.sqlite"))


def test_paragraph_embedding_index_fingerprints_endpoint_without_persisting_it(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "endpoint-fingerprint")
    _, revision = project.import_text("source.txt", "Apples are green.")
    endpoint = "https://embedding-endpoint-marker.example.test/v1/tenant-marker"
    secret_reference = "DL_ENDPOINT_SECRET_REFERENCE_MARKER"
    secret_value = "secret-value-marker-that-must-not-be-persisted"
    monkeypatch.setenv(secret_reference, secret_value)
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", secret_reference, endpoint)
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )
    provider = _QueryAwareEmbeddingFixture(endpoint=endpoint)

    artifact_id = build_paragraph_embedding_index(project, "embed", revision, provider)

    row = project.db.execute(
        "SELECT relative_path,schema_version FROM artifact_registry WHERE id=?",
        (artifact_id,),
    ).fetchone()
    assert row["schema_version"].startswith("vector-index-v4:")
    artifact_path = project.hidden / row["relative_path"]
    with sqlite3.connect(artifact_path) as connection:
        fingerprint = connection.execute(
            "SELECT embedding_endpoint_fingerprint FROM vector_index_identity"
        ).fetchone()[0]
    assert fingerprint.startswith("embedding-endpoint-v1:")
    artifact_bytes = artifact_path.read_bytes()
    assert endpoint.encode() not in artifact_bytes
    assert secret_reference.encode() not in artifact_bytes
    assert secret_value.encode() not in artifact_bytes
    assert "endpoint" not in row

    assert semantic_search_with_preset(
        project,
        "embed",
        project.snapshot([revision]),
        [artifact_id],
        "green fruit",
        provider,
    )
    assert provider.query_requests == [("embedding-model", ["green fruit"])]


def test_semantic_search_rejects_changed_endpoint_before_query_embedding(tmp_path):
    project = Project.init(tmp_path / "changed-endpoint")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference(
            "cohere-embeddings", "lab", "DL_COHERE", "https://index.example.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )
    provider = _QueryAwareEmbeddingFixture(endpoint="https://index.example.test/v1")
    artifact_id = build_paragraph_embedding_index(project, "embed", revision, provider)

    project.save_credential_reference(
        CredentialReference(
            "cohere-embeddings", "lab", "DL_COHERE", "https://query.example.test/v1"
        )
    )
    provider.endpoint = "https://query.example.test/v1"
    with pytest.raises(ScopeViolationError, match="endpoint provenance.*incompatible"):
        semantic_search_with_preset(
            project,
            "embed",
            project.snapshot([revision]),
            [artifact_id],
            "green fruit",
            provider,
        )

    assert provider.query_requests == []


def test_injected_embedding_provider_endpoint_must_match_selected_credential(tmp_path):
    project = Project.init(tmp_path / "injected-provider-endpoint")
    _, revision = project.import_text("source.txt", "Apples are green.")
    selected_endpoint = "https://selected.example.test/v1"
    other_endpoint = "https://other.example.test/v1"
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE", selected_endpoint)
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )

    class _WrongEndpointProvider:
        provider_id = "cohere-embeddings"
        endpoint = other_endpoint

        def __init__(self):
            self.document_calls = 0

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.document_calls += 1
            return [(0.1, 0.0) for _ in texts]

    wrong_document_provider = _WrongEndpointProvider()
    with pytest.raises(ScopeViolationError, match="provider endpoint does not match"):
        build_paragraph_embedding_index(
            project, "embed", revision, wrong_document_provider
        )
    assert wrong_document_provider.document_calls == 0
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0

    index_provider = _QueryAwareEmbeddingFixture(endpoint=selected_endpoint)
    artifact_id = build_paragraph_embedding_index(project, "embed", revision, index_provider)
    wrong_query_provider = _QueryAwareEmbeddingFixture(endpoint=other_endpoint)
    with pytest.raises(ScopeViolationError, match="provider endpoint does not match"):
        semantic_search_with_preset(
            project,
            "embed",
            project.snapshot([revision]),
            [artifact_id],
            "green fruit",
            wrong_query_provider,
        )
    assert wrong_query_provider.query_requests == []


def test_injected_embedding_provider_must_declare_endpoint(tmp_path):
    project = Project.init(tmp_path / "injected-provider-without-endpoint")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference(
            "cohere-embeddings", "lab", "DL_COHERE", "https://selected.example.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )

    class _UnidentifiedEndpointProvider:
        provider_id = "cohere-embeddings"

        def __init__(self):
            self.document_calls = 0

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.document_calls += 1
            return [(0.1, 0.0) for _ in texts]

    provider = _UnidentifiedEndpointProvider()
    with pytest.raises(ScopeViolationError, match="must declare an endpoint"):
        build_paragraph_embedding_index(
            project, "embed", revision, cast(EmbeddingProvider, provider)
        )
    assert provider.document_calls == 0


@pytest.mark.parametrize(
    ("endpoint", "secret_markers"),
    [
        (
            "https://user-marker:password-marker@embedding.example.test/v1",
            ("user-marker", "password-marker"),
        ),
        ("https://embedding.example.test/v1?token=query-marker", ("query-marker",)),
        ("https://embedding.example.test/v1#fragment-marker", ("fragment-marker",)),
        ("https://embedding.example.test/v1?", ()),
        ("https://embedding.example.test/v1#", ()),
    ],
)
def test_paragraph_embedding_index_rejects_endpoint_secrets_and_suffixes_before_creation(
    tmp_path, endpoint: str, secret_markers: tuple[str, ...]
):
    project = Project.init(tmp_path / "invalid-endpoint")
    _, revision = project.import_text("source.txt", "Apples are green.")
    safe_endpoint = "https://embedding.example.test/v1"
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE", safe_endpoint)
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )
    project.db.execute(
        "UPDATE provider_credentials SET endpoint=? WHERE provider_id=? AND credential_alias=?",
        (endpoint, "cohere-embeddings", "lab"),
    )
    project.db.commit()
    provider = _QueryAwareEmbeddingFixture()

    with pytest.raises(DocumentError, match="stored credential metadata is unsafe") as error:
        build_paragraph_embedding_index(project, "embed", revision, provider)
    assert endpoint not in str(error.value)
    assert all(marker not in str(error.value) for marker in secret_markers)

    assert provider.query_requests == []
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0
    assert not list((project.hidden / "artifacts" / "vectors").glob("*.sqlite"))


def test_legacy_v2_vectors_remain_raw_searchable_but_require_rebuild_for_preset_queries(
    tmp_path,
):
    project = Project.init(tmp_path / "legacy-v2-vector")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(CredentialReference("cohere-embeddings", "lab", "DL_COHERE"))
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )
    legacy_id = build_vector_index(
        project,
        revision,
        [VectorChunk("legacy", 0, 6, (0.1, 0.0))],
        embedding_provider_id="cohere-embeddings",
        embedding_model="embedding-model",
        chunker_version="plain-text-lines-v1",
        embedding_input_type="search_document",
    )
    registry_row = project.db.execute(
        "SELECT relative_path FROM artifact_registry WHERE id=?", (legacy_id,)
    ).fetchone()
    legacy_path = project.hidden / registry_row["relative_path"]
    with sqlite3.connect(legacy_path) as connection:
        connection.execute(
            "ALTER TABLE vector_index_identity RENAME TO vector_index_identity_v4"
        )
        connection.execute(
            "CREATE TABLE vector_index_identity(project_id TEXT NOT NULL,"
            "artifact_id TEXT NOT NULL,revision_id TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO vector_index_identity SELECT project_id,artifact_id,revision_id "
            "FROM vector_index_identity_v4"
        )
        connection.execute("DROP TABLE vector_index_identity_v4")
    _rewrite_vector_artifact_as_legacy_l2(
        project,
        legacy_id,
        [VectorChunk("legacy", 0, 6, (0.1, 0.0))],
        schema_version="vector-index-v2:2",
    )

    assert project.db.execute(
        "SELECT id FROM artifact_registry WHERE id=?", (legacy_id,)
    ).fetchone()[0] == legacy_id
    assert search_vectors(
        project, project.snapshot([revision]), [legacy_id], (0.1, 0.0)
    )
    provider = _QueryAwareEmbeddingFixture()
    with pytest.raises(ScopeViolationError, match="endpoint-identified.*rebuild the index"):
        semantic_search_with_preset(
            project,
            "embed",
            project.snapshot([revision]),
            [legacy_id],
            "green fruit",
            provider,
        )
    assert provider.query_requests == []

    rebuilt_id = build_paragraph_embedding_index(project, "embed", revision, provider)
    assert rebuilt_id != legacy_id
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE id IN (?,?)", (legacy_id, rebuilt_id)
    ).fetchone()[0] == 2
    assert semantic_search_with_preset(
        project,
        "embed",
        project.snapshot([revision]),
        [rebuilt_id],
        "green fruit",
        provider,
    )


def test_embedding_rebuild_keeps_history_and_rejects_mixed_provenance(tmp_path):
    project = Project.init(tmp_path / "rebuild")
    _, revision = project.import_text("source.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("embed-v1", "cohere-embeddings", "lab", "embed-v1")
    )
    provider = _RebuildEmbeddingFixture()

    original_id = build_paragraph_embedding_index(project, "embed-v1", revision, provider)
    original_row = dict(
        project.db.execute("SELECT * FROM artifact_registry WHERE id=?", (original_id,)).fetchone()
    )
    assert original_row["chunker_version"] == "plain-text-lines-v2"
    assert len(
        semantic_search_with_preset(
            project, "embed-v1", scope_id, [original_id], "green", provider
        )
    ) == 1

    # Simulate a local re-segmentation of the same immutable revision before rebuilding.
    project.db.execute(
        "UPDATE paragraphs SET segmentation_version=? WHERE revision_id=?",
        ("plain-text-lines-v3", revision),
    )
    project.db.commit()
    project.save_model_preset(
        ModelPreset("embed-chunker-v2", "cohere-embeddings", "lab", "embed-v1")
    )
    chunker_rebuild_id = build_paragraph_embedding_index(
        project, "embed-chunker-v2", revision, provider
    )
    assert chunker_rebuild_id != original_id
    chunker_row = dict(
        project.db.execute(
            "SELECT * FROM artifact_registry WHERE id=?", (chunker_rebuild_id,)
        ).fetchone()
    )
    assert chunker_row["chunker_version"] == "plain-text-lines-v3"
    assert dict(
        project.db.execute("SELECT * FROM artifact_registry WHERE id=?", (original_id,)).fetchone()
    ) == original_row

    query_count = len(provider.query_models)
    with pytest.raises(ScopeViolationError, match="incompatible embedding or chunker provenance"):
        semantic_search_with_preset(
            project,
            "embed-chunker-v2",
            scope_id,
            [original_id, chunker_rebuild_id],
            "green",
            provider,
        )
    with pytest.raises(ScopeViolationError, match="incompatible embedding or chunker provenance"):
        hybrid_search(
            project,
            scope_id,
            [original_id, chunker_rebuild_id],
            "Apples",
            (1.0, 0.0),
        )
    assert len(provider.query_models) == query_count

    project.db.execute(
        "UPDATE paragraphs SET segmentation_version=? WHERE revision_id=?",
        ("plain-text-lines-v3", revision),
    )
    project.db.commit()
    project.save_model_preset(
        ModelPreset("embed-v2", "cohere-embeddings", "lab", "embed-v2")
    )
    model_rebuild_id = build_paragraph_embedding_index(project, "embed-v2", revision, provider)
    model_row = dict(
        project.db.execute(
            "SELECT * FROM artifact_registry WHERE id=?", (model_rebuild_id,)
        ).fetchone()
    )
    assert model_rebuild_id not in {original_id, chunker_rebuild_id}
    assert model_row["embedding_model"] == "embed-v2"
    assert model_row["chunker_version"] == "plain-text-lines-v3"
    project.close()
    project = Project.open(tmp_path / "rebuild")
    assert dict(
        project.db.execute("SELECT * FROM artifact_registry WHERE id=?", (original_id,)).fetchone()
    ) == original_row
    assert dict(
        project.db.execute("SELECT * FROM artifact_registry WHERE id=?", (model_rebuild_id,)).fetchone()
    ) == model_row
    assert len(
        semantic_search_with_preset(
            project, "embed-v1", scope_id, [original_id], "green", provider
        )
    ) == 1
    assert len(
        semantic_search_with_preset(
            project, "embed-v2", scope_id, [model_rebuild_id], "green", provider
        )
    ) == 1
    query_count = len(provider.query_models)
    with pytest.raises(ScopeViolationError, match="does not match vector index provenance"):
        semantic_search_with_preset(
            project, "embed-v2", scope_id, [original_id], "green", provider
        )
    with pytest.raises(ScopeViolationError, match="incompatible embedding or chunker provenance"):
        hybrid_search(
            project, scope_id, [original_id, model_rebuild_id], "Apples", (0.0, 1.0)
        )
    with pytest.raises(ScopeViolationError, match="explicit indexes"):
        semantic_search_with_preset(project, "embed-v2", scope_id, [], "green", provider)
    with pytest.raises(ScopeViolationError, match="artifact IDs"):
        hybrid_search(project, scope_id, [], "Apples", (0.0, 1.0))
    assert len(provider.query_models) == query_count

    run_id = project.create_run(scope_id, {"profile": "vector-rebuild"})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition="one paragraph",
        )
    )
    skill_run_id = project.create_skill_run(
        run_id,
        scope_id,
        "fixture.vector-search",
        "1.0.0",
        "test-fixture",
        "fixture-config-hash",
        coverage_id=coverage_id,
    )
    plan_ref = f"semantic-search-with-preset/v1/{model_rebuild_id}"
    retrieval_id = project.create_retrieval_run(
        analysis_run_id=run_id,
        skill_run_id=skill_run_id,
        scope_id=scope_id,
        retrieval_plan_ref=plan_ref,
        query_terms=["green"],
        selection_policy="explicit-vector-artifact-id",
        candidate_count=1,
        inspected_count=1,
        returned_count=1,
        candidate_limit_reached=False,
        coverage_id=coverage_id,
    )
    result = semantic_search_with_preset(
        project, "embed-v2", scope_id, [model_rebuild_id], "green", provider
    )[0]
    span_id = project.create_span(revision, result.start, result.end)
    project.add_retrieval_hit(retrieval_id, span_id, "semantic-vector", 1, 1.0)
    report = analysis_payload(project, run_id)
    reported_run = report["retrieval_runs"][0]
    assert reported_run["retrieval_plan_ref"] == plan_ref
    assert report["retrieval_hits"][0]["retrieval_run_id"] == retrieval_id
    assert model_row["id"] in reported_run["retrieval_plan_ref"]


def test_semantic_search_rejects_malformed_scope_before_query_embedding(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    project.save_credential_reference(CredentialReference("cohere-embeddings", "lab", "DL_COHERE"))
    project.save_model_preset(ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model"))
    provider = _QueryAwareEmbeddingFixture()
    artifact_id = build_paragraph_embedding_index(project, "embed", revision, provider)
    scope_id = project.snapshot([revision])
    monkeypatch.setattr(project, "scope_revisions", lambda _scope_id: [revision, 123])

    with pytest.raises(ScopeViolationError, match="scope"):
        semantic_search_with_preset(
            project, "embed", scope_id, [artifact_id], "research query", provider
        )

    assert provider.query_requests == []


def test_paragraph_embedding_index_rejects_provider_count_mismatch(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.\nSecond paragraph.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    project.save_model_preset(ModelPreset("embed", "openai-compatible", "local", "embedding-model"))

    with pytest.raises(ModelResponseError, match="invalid vector count"):
        build_paragraph_embedding_index(project, "embed", revision, _EmptyEmbeddingFixture())

    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
    ).fetchone()[0] == 0


def test_semantic_search_rejects_provider_without_exactly_one_query_vector(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green.")
    project.save_credential_reference(CredentialReference("cohere-embeddings", "lab", "DL_COHERE"))
    project.save_model_preset(ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model"))
    index_provider = _QueryAwareEmbeddingFixture()
    artifact_id = build_paragraph_embedding_index(project, "embed", revision, index_provider)
    query_provider = _EmptyQueryEmbeddingFixture()

    with pytest.raises(ModelResponseError, match="invalid vector count"):
        semantic_search_with_preset(
            project,
            "embed",
            project.snapshot([revision]),
            [artifact_id],
            "green fruit",
            query_provider,
        )

    assert query_provider.query_requests == [("embedding-model", ["green fruit"])]


def test_cohere_preset_builds_and_queries_with_distinct_search_input_types(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.\nSecond paragraph.")
    project.save_credential_reference(CredentialReference("cohere-embeddings", "lab", "DL_COHERE"))
    project.save_model_preset(ModelPreset("embed", "cohere-embeddings", "lab", "embed-v4.0"))
    monkeypatch.setenv("DL_COHERE", "fixture-secret")

    managers = []
    for vectors in ([[0.1, 0.0], [0.2, 0.0]], [[0.1, 0.0]]):
        response = MagicMock()
        response.read.return_value = json.dumps({"embeddings": {"float": vectors}}).encode()
        manager = MagicMock()
        manager.__enter__.return_value = response
        managers.append(manager)

    with patch("discourselens.providers._open_provider_request", side_effect=managers) as opened:
        artifact_id = build_paragraph_embedding_index(project, "embed", revision)
        metadata = project.db.execute(
            "SELECT embedding_provider_id,embedding_model,embedding_dimension,chunker_version,"
            "embedding_input_type FROM artifact_registry WHERE id=?",
            (artifact_id,),
        ).fetchone()
        hits = semantic_search_with_preset(
            project, "embed", project.snapshot([revision]), [artifact_id], "research query"
        )

    assert len(hits) == 2
    assert {hit.revision_id for hit in hits} == {revision}
    assert tuple(metadata) == (
        "cohere-embeddings",
        "embed-v4.0",
        2,
        "plain-text-lines-v2",
        "search_document",
    )
    requests = [call.args[0] for call in opened.call_args_list]
    assert [request.full_url for request in requests] == [
        "https://api.cohere.com/v2/embed",
        "https://api.cohere.com/v2/embed",
    ]
    payloads = [json.loads(request.data.decode()) for request in requests]
    assert [payload["input_type"] for payload in payloads] == [
        "search_document",
        "search_query",
    ]


def test_voyage_preset_builds_and_queries_with_distinct_search_input_types(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.\nSecond paragraph.")
    project.save_credential_reference(CredentialReference("voyage-embeddings", "lab", "DL_VOYAGE"))
    project.save_model_preset(ModelPreset("embed", "voyage-embeddings", "lab", "voyage-4"))
    monkeypatch.setenv("DL_VOYAGE", "fixture-secret")

    managers = []
    for vectors in ([[0.1, 0.0], [0.2, 0.0]], [[0.1, 0.0]]):
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "data": [
                    {"index": index, "embedding": vector} for index, vector in enumerate(vectors)
                ]
            }
        ).encode()
        manager = MagicMock()
        manager.__enter__.return_value = response
        managers.append(manager)

    with patch("discourselens.providers._open_provider_request", side_effect=managers) as opened:
        artifact_id = build_paragraph_embedding_index(project, "embed", revision)
        metadata = project.db.execute(
            "SELECT embedding_provider_id,embedding_model,embedding_dimension,chunker_version,"
            "embedding_input_type FROM artifact_registry WHERE id=?",
            (artifact_id,),
        ).fetchone()
        hits = semantic_search_with_preset(
            project, "embed", project.snapshot([revision]), [artifact_id], "research query"
        )

    assert len(hits) == 2
    assert {hit.revision_id for hit in hits} == {revision}
    assert tuple(metadata) == (
        "voyage-embeddings",
        "voyage-4",
        2,
        "plain-text-lines-v2",
        "document",
    )
    requests = [call.args[0] for call in opened.call_args_list]
    assert all(request.full_url == "https://api.voyageai.com/v1/embeddings" for request in requests)
    payloads = [json.loads(request.data.decode()) for request in requests]
    assert [payload["input_type"] for payload in payloads] == ["document", "query"]
    assert all(payload["truncation"] is False for payload in payloads)


def test_jina_embedding_adapter_orders_vectors_and_uses_document_query_tasks(monkeypatch):
    monkeypatch.setenv("DL_JINA", "fixture-secret")
    provider = JinaEmbeddingProvider("DL_JINA", EnvironmentSecretStore())
    responses = []
    for data in (
        [
            {"index": 1, "embedding": [3.0, 4.0]},
            {"index": 0, "embedding": [1.0, 2.0]},
        ],
        [{"index": 0, "embedding": [0.1, 0.2]}],
    ):
        response = MagicMock()
        response.read.return_value = json.dumps({"data": data}).encode()
        manager = MagicMock()
        manager.__enter__.return_value = response
        responses.append(manager)

    with patch("discourselens.providers._open_provider_request", side_effect=responses) as opened:
        passages = provider.embed("jina-embeddings-v5-text-small", ["first", "second"])
        query = provider.embed_query("jina-embeddings-v5-text-small", ["query"])

    assert passages == [(1.0, 2.0), (3.0, 4.0)]
    assert query == [(0.1, 0.2)]
    requests = [call.args[0] for call in opened.call_args_list]
    assert all(request.full_url == "https://api.jina.ai/v1/embeddings" for request in requests)
    assert all(
        request.get_header("Authorization") == "Bearer fixture-secret" for request in requests
    )
    payloads = [json.loads(request.data.decode()) for request in requests]
    assert [payload["task"] for payload in payloads] == ["retrieval.passage", "retrieval.query"]
    assert all(payload["embedding_type"] == "float" for payload in payloads)


@pytest.mark.parametrize(
    ("model", "expected_document", "expected_query", "expected_config"),
    [
        (
            "gemini-embedding-001",
            "Corpus passage",
            "research question",
            [
                {"autoTruncate": False, "taskType": "RETRIEVAL_DOCUMENT"},
                {"autoTruncate": False, "taskType": "RETRIEVAL_QUERY"},
            ],
        ),
        (
            "gemini-embedding-2",
            "title: none | text: Corpus passage",
            "task: search result | query: research question",
            [{"autoTruncate": False}, {"autoTruncate": False}],
        ),
    ],
)
def test_gemini_embedding_model_families_use_documented_retrieval_contract(
    monkeypatch, model, expected_document, expected_query, expected_config
):
    monkeypatch.setenv("DL_GEMINI", "fixture-secret")
    responses = []
    for values in ([[0.25, 0.75]], [[0.5, 0.5]]):
        response = MagicMock()
        response.read.return_value = json.dumps({"embeddings": [{"values": values[0]}]}).encode()
        manager = MagicMock()
        manager.__enter__.return_value = response
        responses.append(manager)
    provider = GeminiEmbeddingProvider("DL_GEMINI", EnvironmentSecretStore())

    with patch("discourselens.providers._open_provider_request", side_effect=responses) as opened:
        document = provider.embed(model, ["Corpus passage"])
        query = provider.embed_query(model, ["research question"])

    assert document == [(0.25, 0.75)]
    assert query == [(0.5, 0.5)]
    requests = [call.args[0] for call in opened.call_args_list]
    assert all(
        request.full_url
        == f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
        for request in requests
    )
    assert all(request.get_header("X-goog-api-key") == "fixture-secret" for request in requests)
    payloads = [json.loads(request.data.decode()) for request in requests]
    assert [payload["requests"][0]["content"]["parts"][0]["text"] for payload in payloads] == [
        expected_document,
        expected_query,
    ]
    assert [payload["requests"][0]["embedContentConfig"] for payload in payloads] == (
        expected_config
    )
    assert all(payload["requests"][0]["model"] == f"models/{model}" for payload in payloads)


def test_gemini_embedding_rejects_unknown_model_and_invalid_response(monkeypatch):
    monkeypatch.setenv("DL_GEMINI", "fixture-secret")
    provider = GeminiEmbeddingProvider("DL_GEMINI", EnvironmentSecretStore())
    with pytest.raises(ModelProviderError, match="unsupported Gemini embedding model"):
        provider.embed("unreviewed-model", ["text"])

    response = MagicMock()
    response.read.return_value = json.dumps({"embeddings": [{"values": [float("nan")]}]}).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    with (
        patch("discourselens.providers._open_provider_request", return_value=manager),
        pytest.raises(ModelResponseError, match="valid ordered embeddings"),
    ):
        provider.embed("gemini-embedding-2", ["text"])


def test_gemini_embedding_index_provenance_and_query_mode(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.\nSecond paragraph.")
    project.save_credential_reference(CredentialReference("gemini-embeddings", "lab", "DL_GEMINI"))
    project.save_model_preset(
        ModelPreset("embed", "gemini-embeddings", "lab", "gemini-embedding-2")
    )
    monkeypatch.setenv("DL_GEMINI", "fixture-secret")
    managers = []
    for values in ([[0.1, 0.0], [0.2, 0.0]], [[0.1, 0.0]]):
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"embeddings": [{"values": vector} for vector in values]}
        ).encode()
        manager = MagicMock()
        manager.__enter__.return_value = response
        managers.append(manager)

    with patch("discourselens.providers._open_provider_request", side_effect=managers) as opened:
        artifact_id = build_paragraph_embedding_index(project, "embed", revision)
        row = project.db.execute(
            "SELECT embedding_provider_id,embedding_model,embedding_dimension,"
            "chunker_version,embedding_input_type FROM artifact_registry WHERE id=?",
            (artifact_id,),
        ).fetchone()
        hits = semantic_search_with_preset(
            project, "embed", project.snapshot([revision]), [artifact_id], "research query"
        )

    assert tuple(row) == (
        "gemini-embeddings",
        "gemini-embedding-2",
        2,
        "plain-text-lines-v2",
        "title-none-text-v1",
    )
    assert len(hits) == 2
    payloads = [json.loads(call.args[0].data.decode()) for call in opened.call_args_list]
    assert payloads[0]["requests"][0]["content"]["parts"][0]["text"] == (
        "title: none | text: First paragraph."
    )
    assert payloads[-1]["requests"][0]["content"]["parts"][0]["text"] == (
        "task: search result | query: research query"
    )


def test_embedding_preview_is_paged_secret_free_and_does_not_resolve_secrets(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text(
        "notes.txt", "First private paragraph.\nSecond private paragraph.\nThird paragraph."
    )
    project.save_credential_reference(
        CredentialReference("gemini-embeddings", "lab", "DL_GEMINI_PREVIEW")
    )
    project.save_model_preset(
        ModelPreset("embed", "gemini-embeddings", "lab", "gemini-embedding-2")
    )
    monkeypatch.setenv("DL_GEMINI_PREVIEW", "must-never-be-read-or-printed")
    monkeypatch.setattr(
        "discourselens.secrets.ReferenceSecretStore.get",
        lambda *_: pytest.fail("preview must never resolve a secret value"),
    )

    preview = preview_paragraph_embedding_request(project, "embed", revision, offset=1, limit=1)

    assert preview.total_paragraphs == 3
    assert preview.selected_count == 1
    assert preview.offset == 1
    assert preview.credential_alias == "lab"
    assert preview.endpoint == "https://generativelanguage.googleapis.com/v1beta"
    assert preview.input_type == "title-none-text-v1"
    assert preview.secret_values_resolved is False
    assert preview.network_request_made is False
    assert preview.batches[0].request_body is None
    item = preview.batches[0].inputs[0]
    assert item.prepared_text is None
    assert item.character_count == len("Second private paragraph.")
    assert "Second private paragraph." not in repr(preview)
    assert preview.batches[0].request_body_template == {
        "requests": [
            {
                "model": "models/gemini-embedding-2",
                "content": {
                    "parts": [
                        {
                            "text": f"title: none | text: <paragraph text omitted: {item.paragraph_id}>"
                        }
                    ]
                },
                "embedContentConfig": {"autoTruncate": False},
            }
        ]
    }
    assert "must-never-be-read-or-printed" not in repr(preview)


def test_embedding_preview_include_text_matches_gemini_document_wire_body(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("notes.txt", "First private paragraph.")
    project.save_credential_reference(
        CredentialReference("gemini-embeddings", "lab", "DL_GEMINI_PREVIEW")
    )
    project.save_model_preset(
        ModelPreset("embed", "gemini-embeddings", "lab", "gemini-embedding-2")
    )

    preview = preview_paragraph_embedding_request(
        project, "embed", revision, limit=1, include_text=True
    )

    assert preview.network_request_made is False
    assert preview.secret_values_resolved is False
    assert preview.batches[0].inputs[0].prepared_text == (
        "title: none | text: First private paragraph."
    )
    assert preview.batches[0].request_body == {
        "requests": [
            GeminiEmbeddingProvider.document_request_item(
                "gemini-embedding-2", "First private paragraph."
            )
        ]
    }


def test_embedding_preview_rejects_unknown_gemini_embedding_model(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("notes.txt", "A paragraph.")
    project.save_credential_reference(
        CredentialReference("gemini-embeddings", "lab", "DL_GEMINI_PREVIEW")
    )
    project.save_model_preset(ModelPreset("embed", "gemini-embeddings", "lab", "unknown"))

    with pytest.raises(ModelProviderError, match="unsupported Gemini embedding model"):
        preview_paragraph_embedding_request(project, "embed", revision)


@pytest.mark.parametrize(
    ("texts", "data"),
    [
        (["first", "second"], [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]),
        (["first"], [{"index": 0, "embedding": [float("nan")]}]),
    ],
)
def test_jina_embedding_rejects_misaligned_or_nonfinite_vectors(monkeypatch, texts, data):
    monkeypatch.setenv("DL_JINA", "fixture-secret")
    response = MagicMock()
    response.read.return_value = json.dumps({"data": data}).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = JinaEmbeddingProvider("DL_JINA", EnvironmentSecretStore())

    with (
        patch("discourselens.providers._open_provider_request", return_value=manager),
        pytest.raises(ModelResponseError, match="valid ordered embeddings"),
    ):
        provider.embed("jina-embeddings-v5-text-small", texts)


def test_jina_paragraph_index_records_task_mode_and_queries_semantically(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.\nSecond paragraph.")
    project.save_credential_reference(CredentialReference("jina-embeddings", "lab", "DL_JINA"))
    project.save_model_preset(
        ModelPreset("embed", "jina-embeddings", "lab", "jina-embeddings-v5-text-small")
    )
    monkeypatch.setenv("DL_JINA", "fixture-secret")

    managers = []
    for vectors in ([[0.1, 0.0], [0.2, 0.0]], [[0.1, 0.0]]):
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "data": [
                    {"index": index, "embedding": vector} for index, vector in enumerate(vectors)
                ]
            }
        ).encode()
        manager = MagicMock()
        manager.__enter__.return_value = response
        managers.append(manager)

    with patch("discourselens.providers._open_provider_request", side_effect=managers) as opened:
        artifact_id = build_paragraph_embedding_index(project, "embed", revision)
        row = project.db.execute(
            "SELECT embedding_provider_id,embedding_model,embedding_dimension,"
            "chunker_version,embedding_input_type FROM artifact_registry WHERE id=?",
            (artifact_id,),
        ).fetchone()
        hits = semantic_search_with_preset(
            project, "embed", project.snapshot([revision]), [artifact_id], "research query"
        )

    assert tuple(row) == (
        "jina-embeddings",
        "jina-embeddings-v5-text-small",
        2,
        "plain-text-lines-v2",
        "retrieval.passage",
    )
    assert len(hits) == 2
    payloads = [json.loads(call.args[0].data.decode()) for call in opened.call_args_list]
    assert [payload["task"] for payload in payloads] == [
        "retrieval.passage",
        "retrieval.query",
    ]


def test_jina_legacy_model_is_explicitly_symmetric_and_offline_blocks_remote(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "One paragraph.")
    project.save_credential_reference(CredentialReference("jina-embeddings", "lab", "DL_JINA"))
    project.save_model_preset(
        ModelPreset("embed", "jina-embeddings", "lab", "jina-embeddings-v2-base-en")
    )
    assert JinaEmbeddingProvider.document_input_type("jina-embeddings-v2-base-en") == "symmetric"
    project.set_offline_mode(True)

    with pytest.raises(ProjectError, match="offline mode"):
        build_paragraph_embedding_index(project, "embed", revision)


def test_offline_mode_blocks_remote_embedding_operations(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "remote", "DL_REMOTE", "https://api.example.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("embed", "openai-compatible", "remote", "embedding-model")
    )
    project.set_offline_mode(True)
    with pytest.raises(ProjectError, match="offline mode"):
        build_paragraph_embedding_index(project, "embed", revision, _EmbeddingFixture())


def test_offline_embedding_rejection_precedes_secret_resolution_and_transport(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "First paragraph.")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "remote", "DL_REMOTE", "https://api.example.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("embed", "openai-compatible", "remote", "embedding-model")
    )
    project.set_offline_mode(True)
    secret_lookups: list[str] = []
    requests: list[object] = []

    def get_secret(_store, reference: str) -> str:
        secret_lookups.append(reference)
        raise AssertionError("offline rejection must precede secret resolution")

    def open_request(request, **_kwargs):
        requests.append(request)
        raise AssertionError("offline rejection must precede provider transport")

    monkeypatch.setattr("discourselens.secrets.ReferenceSecretStore.get", get_secret)
    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)

    with pytest.raises(ProjectError, match="offline mode"):
        build_paragraph_embedding_index(project, "embed", revision)

    assert secret_lookups == []
    assert requests == []


def test_offline_mode_blocks_openai_default_embedding_endpoint(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Evidence remains local.")
    project.save_credential_reference(CredentialReference("openai", "remote", "DL_OPENAI"))
    project.save_model_preset(ModelPreset("embed", "openai", "remote", "embedding-model"))
    project.set_offline_mode(True)

    with pytest.raises(ProjectError, match="offline mode"):
        build_paragraph_embedding_index(project, "embed", revision, _EmbeddingFixture())


def test_hybrid_search_fuses_explicit_scoped_lexical_and_vector_candidates(tmp_path):
    project = Project.init(tmp_path / "p")
    _, selected = project.import_text("selected.txt", "Apples are green.")
    _, outside = project.import_text("outside.txt", "Apples are not local.")
    selected_paragraph = project.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (selected,)
    ).fetchone()[0]
    outside_paragraph = project.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (outside,)
    ).fetchone()[0]
    selected_index = build_vector_index(
        project, selected, [VectorChunk(selected_paragraph, 0, 17, (0.3, 0.0))]
    )
    outside_index = build_vector_index(
        project, outside, [VectorChunk(outside_paragraph, 0, 20, (0.01, 0.0))]
    )
    scope = project.snapshot([selected])

    hits = hybrid_search(project, scope, [selected_index], "Apples", (1.0, 0.0))

    assert len(hits) == 1
    assert hits[0].revision_id == selected
    assert hits[0].unit_id == selected_paragraph
    assert hits[0].lexical_rank == 1 and hits[0].vector_rank == 1
    with pytest.raises(ScopeViolationError):
        hybrid_search(project, scope, [outside_index], "Apples", (1.0, 0.0))


def test_hybrid_search_with_preset_embeds_once_and_keeps_scope_strict(tmp_path):
    project = Project.init(tmp_path / "p")
    _, selected = project.import_text("selected.txt", "Apples are green.")
    _, outside = project.import_text("outside.txt", "Apples are not local.")
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )
    provider = _RebuildEmbeddingFixture()
    selected_index = build_paragraph_embedding_index(project, "embed", selected, provider)
    outside_index = build_paragraph_embedding_index(project, "embed", outside, provider)
    scope_id = project.snapshot([selected])

    hits = hybrid_search_with_preset(
        project, "embed", scope_id, [selected_index], "Apples", provider
    )

    selected_paragraph = project.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (selected,)
    ).fetchone()[0]
    assert provider.query_models == ["embedding-model"]
    assert len(hits) == 1
    assert hits[0].revision_id == selected
    assert hits[0].unit_id == selected_paragraph
    assert hits[0].lexical_rank == 1 and hits[0].vector_rank == 1
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        hybrid_search_with_preset(
            project, "embed", scope_id, [outside_index], "Apples", provider
        )
    assert provider.query_models == ["embedding-model"]


def test_hybrid_search_with_preset_rejects_model_mismatch_before_query_embedding(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("indexed", "cohere-embeddings", "lab", "embed-v1")
    )
    project.save_model_preset(
        ModelPreset("query", "cohere-embeddings", "lab", "embed-v2")
    )
    provider = _RebuildEmbeddingFixture()
    artifact_id = build_paragraph_embedding_index(project, "indexed", revision, provider)

    with pytest.raises(ScopeViolationError, match="does not match vector index provenance"):
        hybrid_search_with_preset(
            project,
            "query",
            project.snapshot([revision]),
            [artifact_id],
            "Apples",
            provider,
        )

    assert provider.query_models == []


def test_hybrid_search_with_preset_rejects_chunker_mismatch_before_query_embedding(
    tmp_path,
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("source.txt", "Apples are green.")
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embed-v1")
    )
    provider = _RebuildEmbeddingFixture()
    original_index = build_paragraph_embedding_index(project, "embed", revision, provider)
    project.db.execute(
        "UPDATE paragraphs SET segmentation_version=? WHERE revision_id=?",
        ("plain-text-lines-v3", revision),
    )
    project.db.commit()
    rebuilt_index = build_paragraph_embedding_index(project, "embed", revision, provider)

    with pytest.raises(ScopeViolationError, match="incompatible embedding or chunker provenance"):
        hybrid_search_with_preset(
            project,
            "embed",
            project.snapshot([revision]),
            [original_index, rebuilt_index],
            "Apples",
            provider,
        )

    assert provider.query_models == []
