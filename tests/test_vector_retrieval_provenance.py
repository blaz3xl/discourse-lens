import hashlib
import json
import sqlite3
from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import ScopeViolationError
from discourselens.migrations import SCHEMA_VERSION
from discourselens.models import CredentialReference, ModelPreset
from discourselens.types import CoverageDescriptor, CoverageMode
from discourselens.vectors import (
    HybridHit,
    VectorHit,
    build_paragraph_embedding_index,
    hybrid_search_with_preset,
    semantic_search_with_preset,
)


class _EmbeddingFixture:
    provider_id = "cohere-embeddings"
    endpoint = "https://api.cohere.com"

    def __init__(self):
        self.query_requests: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        assert model == "embedding-model"
        return [(1.0, 0.0) for _ in texts]

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.query_requests.append((model, texts))
        return [(1.0, 0.0) for _ in texts]


def _configure_embedding(project: Project) -> _EmbeddingFixture:
    project.save_credential_reference(
        CredentialReference("cohere-embeddings", "lab", "DL_COHERE")
    )
    project.save_model_preset(
        ModelPreset("embed", "cohere-embeddings", "lab", "embedding-model")
    )
    return _EmbeddingFixture()


def _standalone_runs(project: Project, scope_id: str):
    return project.db.execute(
        "SELECT r.*,c.operation,c.config_json FROM retrieval_runs r "
        "JOIN retrieval_run_context c ON c.retrieval_run_id=r.id "
        "WHERE r.scope_id=? ORDER BY r.created_at,r.id",
        (scope_id,),
    ).fetchall()


def test_v42_retrieval_rows_survive_context_link_migration(tmp_path):
    root = tmp_path / "legacy-project"
    project = Project.init(root)
    _, revision = project.import_text("source.txt", "Legacy retrieval result.")
    scope_id = project.snapshot([revision])
    analysis_run_id = project.create_run(scope_id, {"profile": "legacy-fixture"})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition="one fixture paragraph",
            selection_policy="fixed fixture selection",
        )
    )
    skill_run_id = project.create_skill_run(
        analysis_run_id,
        scope_id,
        "fixture.retrieval",
        "1.0.0",
        "fixture",
        "0" * 64,
        coverage_id=coverage_id,
    )
    retrieval_run_id = project.create_retrieval_run(
        analysis_run_id=analysis_run_id,
        skill_run_id=skill_run_id,
        scope_id=scope_id,
        retrieval_plan_ref="retrieval/fixture-v1",
        query_terms=["legacy"],
        selection_policy="fixed fixture selection",
        candidate_count=1,
        inspected_count=1,
        returned_count=1,
        candidate_limit_reached=False,
        coverage_id=coverage_id,
    )
    span_id = project.create_span(revision, 0, len("Legacy retrieval result."))
    hit_id = project.add_retrieval_hit(retrieval_run_id, span_id, "fixture", 1, 1.0)
    old_run = tuple(
        project.db.execute(
            "SELECT * FROM retrieval_runs WHERE id=?", (retrieval_run_id,)
        ).fetchone()
    )
    old_hit = tuple(
        project.db.execute("SELECT * FROM retrieval_hits WHERE id=?", (hit_id,)).fetchone()
    )
    project.close()

    db = sqlite3.connect(root / ".discourselens" / "project.sqlite")
    db.execute("ALTER TABLE retrieval_runs RENAME TO retrieval_runs_v43_fixture")
    db.execute(
        "CREATE TABLE retrieval_runs("
        "id TEXT PRIMARY KEY,analysis_run_id TEXT NOT NULL,skill_run_id TEXT NOT NULL,"
        "scope_id TEXT NOT NULL,retrieval_plan_ref TEXT NOT NULL,query_terms_json TEXT NOT NULL,"
        "selection_policy TEXT NOT NULL,candidate_count INTEGER NOT NULL,inspected_count INTEGER NOT NULL,"
        "returned_count INTEGER NOT NULL,candidate_limit_reached INTEGER NOT NULL,"
        "coverage_id TEXT NOT NULL,created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO retrieval_runs SELECT * FROM retrieval_runs_v43_fixture"
    )
    db.execute("DROP TABLE retrieval_runs_v43_fixture")
    db.execute("DELETE FROM schema_migrations WHERE version>42")
    db.execute("INSERT INTO schema_migrations VALUES(42,'fixture-v42')")
    db.commit()
    db.close()

    reopened = Project.open(root)
    try:
        migrated_run = tuple(
            reopened.db.execute(
                "SELECT * FROM retrieval_runs WHERE id=?", (retrieval_run_id,)
            ).fetchone()
        )
        migrated_hit = tuple(
            reopened.db.execute("SELECT * FROM retrieval_hits WHERE id=?", (hit_id,)).fetchone()
        )
        columns = {
            row[1]: row for row in reopened.db.execute("PRAGMA table_info(retrieval_runs)")
        }

        assert migrated_run == old_run
        assert migrated_hit == old_hit
        assert columns["analysis_run_id"][3] == 0
        assert columns["skill_run_id"][3] == 0
        assert reopened.db.execute(
            "SELECT max(version) FROM schema_migrations"
        ).fetchone()[0] == SCHEMA_VERSION
        assert reopened.db.execute(
            "SELECT count(*) FROM retrieval_run_context"
        ).fetchone()[0] == 0
        assert reopened.db.execute(
            "SELECT count(*) FROM retrieval_hit_context"
        ).fetchone()[0] == 0
    finally:
        reopened.close()


def test_preset_vector_searches_persist_scope_bound_ranked_span_lineage(tmp_path):
    project = Project.init(tmp_path / "search-project")
    _, selected = project.import_text("selected.txt", "Apples are green. Pears are sweet.")
    _, outside = project.import_text("outside.txt", "Apples are outside this scope.")
    provider = _configure_embedding(project)
    selected_index = build_paragraph_embedding_index(project, "embed", selected, provider)
    outside_index = build_paragraph_embedding_index(project, "embed", outside, provider)
    scope_id = project.snapshot([selected])
    analysis_count_before = project.db.execute("SELECT count(*) FROM analysis_runs").fetchone()[0]
    skill_count_before = project.db.execute("SELECT count(*) FROM skill_runs").fetchone()[0]
    evidence_count_before = project.db.execute("SELECT count(*) FROM evidence_artifacts").fetchone()[0]

    semantic_hits = semantic_search_with_preset(
        project, "embed", scope_id, [selected_index], "green apples", provider, limit=5
    )
    hybrid_hits = hybrid_search_with_preset(
        project, "embed", scope_id, [selected_index], "Apples", provider, limit=5
    )
    runs = _standalone_runs(project, scope_id)

    assert {run["operation"] for run in runs} == {
        "semantic_search_with_preset",
        "hybrid_search_with_preset",
    }
    assert provider.query_requests == [
        ("embedding-model", ["green apples"]),
        ("embedding-model", ["Apples"]),
    ]
    assert project.db.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == (
        analysis_count_before
    )
    assert project.db.execute("SELECT count(*) FROM skill_runs").fetchone()[0] == (
        skill_count_before
    )
    assert project.db.execute("SELECT count(*) FROM evidence_artifacts").fetchone()[0] == (
        evidence_count_before
    )

    for run, results, method in (
        (next(run for run in runs if run["operation"] == "semantic_search_with_preset"), semantic_hits, "semantic-vector"),
        (next(run for run in runs if run["operation"] == "hybrid_search_with_preset"), hybrid_hits, "hybrid-rrf"),
    ):
        config = json.loads(run["config_json"])
        query = "green apples" if method == "semantic-vector" else "Apples"
        assert results.scope_id == scope_id
        assert results.retrieval_run_id == run["id"]
        assert results.hits is results
        assert run["analysis_run_id"] is None
        assert run["skill_run_id"] is None
        assert run["scope_id"] == scope_id
        assert run["candidate_count"] >= run["returned_count"] == len(results)
        assert config["artifact_ids"] == [selected_index]
        assert config["provider_id"] == "cohere-embeddings"
        assert config["model"] == "embedding-model"
        assert config["query_sha256"] == hashlib.sha256(query.encode()).hexdigest()
        assert query not in run["config_json"]
        hit_rows = project.db.execute(
            "SELECT h.rank,h.score,h.retrieval_method,s.revision_id,s.start_offset,s.end_offset,"
            "c.artifact_id,c.unit_id,c.lexical_rank,c.vector_rank "
            "FROM retrieval_hits h JOIN source_spans s ON s.id=h.source_span_id "
            "JOIN retrieval_hit_context c ON c.retrieval_hit_id=h.id "
            "WHERE h.retrieval_run_id=? ORDER BY h.rank",
            (run["id"],),
        ).fetchall()
        assert [row["rank"] for row in hit_rows] == list(range(1, len(results) + 1))
        assert all(row["revision_id"] == selected for row in hit_rows)
        assert all(row["retrieval_method"] == method for row in hit_rows)
        if method == "semantic-vector":
            semantic_results = cast(list[VectorHit], results)
            assert [
                (row["start_offset"], row["end_offset"], row["score"])
                for row in hit_rows
            ] == [
                (hit.start, hit.end, hit.distance) for hit in semantic_results
            ]
            assert [
                (row["artifact_id"], row["unit_id"], row["lexical_rank"], row["vector_rank"])
                for row in hit_rows
            ] == [
                (hit.artifact_id, hit.chunk_id, None, rank)
                for rank, hit in enumerate(semantic_results, start=1)
            ]
        else:
            hybrid_results = cast(list[HybridHit], results)
            paragraph_rows = project.db.execute(
                "SELECT id,start_offset,end_offset FROM paragraphs WHERE revision_id=?",
                (selected,),
            ).fetchall()
            offsets_by_id = {
                row["id"]: (row["start_offset"], row["end_offset"])
                for row in paragraph_rows
            }
            assert [
                (row["start_offset"], row["end_offset"], row["score"])
                for row in hit_rows
            ] == [
                (*offsets_by_id[hit.unit_id], hit.fusion_score)
                for hit in hybrid_results
            ]
            assert [
                (row["artifact_id"], row["unit_id"], row["lexical_rank"], row["vector_rank"])
                for row in hit_rows
            ] == [
                (hit.artifact_id, hit.unit_id, hit.lexical_rank, hit.vector_rank)
                for hit in hybrid_results
            ]
            assert all(
                hit.artifact_id == selected_index
                for hit in hybrid_results
                if hit.vector_rank is not None
            )

    assert all(hit.revision_id != outside for hit in semantic_hits + hybrid_hits)
    provider.query_requests.clear()
    run_count_before = project.db.execute(
        "SELECT count(*) FROM retrieval_run_context"
    ).fetchone()[0]
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        semantic_search_with_preset(
            project, "embed", scope_id, [outside_index], "outside", provider
        )
    assert provider.query_requests == []
    assert project.db.execute("SELECT count(*) FROM retrieval_run_context").fetchone()[0] == (
        run_count_before
    )

    expected_runs = [tuple(run) for run in runs]
    expected_hits = [
        tuple(row)
        for row in project.db.execute(
            "SELECT h.* FROM retrieval_hits h JOIN retrieval_runs r "
            "ON r.id=h.retrieval_run_id WHERE r.scope_id=? ORDER BY r.created_at,r.id,h.rank",
            (scope_id,),
        ).fetchall()
    ]
    expected_hit_contexts = [
        tuple(row)
        for row in project.db.execute(
            "SELECT c.* FROM retrieval_hit_context c JOIN retrieval_hits h "
            "ON h.id=c.retrieval_hit_id JOIN retrieval_runs r ON r.id=h.retrieval_run_id "
            "WHERE r.scope_id=? ORDER BY r.created_at,r.id,h.rank",
            (scope_id,),
        ).fetchall()
    ]
    root = project.root
    project.close()
    reopened = Project.open(root)
    try:
        assert [tuple(run) for run in _standalone_runs(reopened, scope_id)] == expected_runs
        assert [
            tuple(row)
            for row in reopened.db.execute(
                "SELECT h.* FROM retrieval_hits h JOIN retrieval_runs r "
                "ON r.id=h.retrieval_run_id WHERE r.scope_id=? ORDER BY r.created_at,r.id,h.rank",
                (scope_id,),
            ).fetchall()
        ] == expected_hits
        assert [
            tuple(row)
            for row in reopened.db.execute(
                "SELECT c.* FROM retrieval_hit_context c JOIN retrieval_hits h "
                "ON h.id=c.retrieval_hit_id JOIN retrieval_runs r ON r.id=h.retrieval_run_id "
                "WHERE r.scope_id=? ORDER BY r.created_at,r.id,h.rank",
                (scope_id,),
            ).fetchall()
        ] == expected_hit_contexts
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("operation", "plan_ref", "selection_policy"),
    [
        (
            "semantic_search_with_preset",
            "retrieval/semantic-vector-preset-v1",
            "semantic empty fixture",
        ),
        (
            "hybrid_search_with_preset",
            "retrieval/hybrid-rrf-preset-v1",
            "hybrid empty fixture",
        ),
    ],
)
def test_empty_preset_search_is_auditable_without_hits_or_evidence(
    tmp_path, monkeypatch, operation, plan_ref, selection_policy
):
    project = Project.init(tmp_path / operation)
    _, revision = project.import_text("empty.txt", "No matching retrieval result.")
    provider = _configure_embedding(project)
    artifact_id = build_paragraph_embedding_index(project, "embed", revision, provider)
    scope_id = project.snapshot([revision])

    monkeypatch.setattr(
        "discourselens.vectors._search_vectors_with_candidate_count",
        lambda *_args, **_kwargs: ([], 0, "cosine"),
    )
    if operation == "semantic_search_with_preset":
        hits = semantic_search_with_preset(
            project, "embed", scope_id, [artifact_id], "missing", provider, limit=5
        )
    else:
        hits = hybrid_search_with_preset(
            project, "embed", scope_id, [artifact_id], "missing", provider, limit=5
        )

    assert hits == []
    run = _standalone_runs(project, scope_id)[0]
    assert hits.scope_id == scope_id
    assert hits.retrieval_run_id == run["id"]
    coverage = project.coverage_descriptor(run["coverage_id"])
    assert run["retrieval_plan_ref"] == plan_ref
    assert run["candidate_count"] == run["inspected_count"] == run["returned_count"] == 0
    assert run["candidate_limit_reached"] == 0
    assert json.loads(run["config_json"])["artifact_ids"] == [artifact_id]
    assert coverage.negative_result_status.value == "NO_MATCHES_UNDER_PLAN"
    assert project.db.execute(
        "SELECT count(*) FROM retrieval_hits WHERE retrieval_run_id=?", (run["id"],)
    ).fetchone()[0] == 0
    assert project.db.execute("SELECT count(*) FROM evidence_artifacts").fetchone()[0] == 0


@pytest.mark.parametrize(
    "operation", ["semantic_search_with_preset", "hybrid_search_with_preset"]
)
def test_preset_search_rejects_multiple_indexes_for_one_revision_before_egress(
    tmp_path, operation
):
    project = Project.init(tmp_path / operation)
    _, revision = project.import_text("source.txt", "One paragraph for duplicate indexes.")
    provider = _configure_embedding(project)
    first_index = build_paragraph_embedding_index(project, "embed", revision, provider)
    second_index = build_paragraph_embedding_index(project, "embed", revision, provider)
    scope_id = project.snapshot([revision])
    provider.query_requests.clear()
    search = (
        semantic_search_with_preset
        if operation == "semantic_search_with_preset"
        else hybrid_search_with_preset
    )

    with pytest.raises(
        ScopeViolationError, match="at most one vector artifact per scoped revision"
    ):
        search(project, "embed", scope_id, [first_index, second_index], "paragraph", provider)

    assert provider.query_requests == []
    assert _standalone_runs(project, scope_id) == []
