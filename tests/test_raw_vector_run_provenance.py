import hashlib
import json
import struct

import pytest

from discourselens import Project
from discourselens.errors import ScopeViolationError
from discourselens.vectors import (
    RetrievalSearchResult,
    VectorChunk,
    build_vector_index,
    hybrid_search,
    hybrid_search_with_provenance,
    search_vectors,
    search_vectors_with_provenance,
)


def _paragraph_chunks(project: Project, revision_id: str) -> list[VectorChunk]:
    rows = project.db.execute(
        "SELECT id,start_offset,end_offset FROM paragraphs "
        "WHERE revision_id=? AND end_offset>start_offset "
        "ORDER BY start_offset",
        (revision_id,),
    ).fetchall()
    return [
        VectorChunk(
            row["id"],
            row["start_offset"],
            row["end_offset"],
            (1.0, 0.0) if index == 0 else (0.0, 1.0),
        )
        for index, row in enumerate(rows)
    ]


def _run_for_result(project: Project, result: RetrievalSearchResult):
    row = project.db.execute(
        "SELECT r.*,c.operation,c.config_json FROM retrieval_runs r "
        "JOIN retrieval_run_context c ON c.retrieval_run_id=r.id WHERE r.id=?",
        (result.retrieval_run_id,),
    ).fetchone()
    assert row is not None
    return row


def _hit_lineage(project: Project, retrieval_run_id: str):
    return project.db.execute(
        "SELECT h.rank,h.score,h.retrieval_method,s.revision_id,s.start_offset,s.end_offset,"
        "c.artifact_id,c.unit_id,c.lexical_rank,c.vector_rank "
        "FROM retrieval_hits h JOIN source_spans s ON s.id=h.source_span_id "
        "JOIN retrieval_hit_context c ON c.retrieval_hit_id=h.id "
        "WHERE h.retrieval_run_id=? ORDER BY h.rank",
        (retrieval_run_id,),
    ).fetchall()


def test_raw_vector_searches_persist_scope_bound_privacy_safe_hit_lineage(tmp_path):
    project = Project.init(tmp_path / "raw-vector-provenance")
    _, selected = project.import_text(
        "selected.txt", "Apples are green.\n\nPears are sweet."
    )
    _, outside = project.import_text("outside.txt", "Apples are outside the selected scope.")
    selected_index = build_vector_index(project, selected, _paragraph_chunks(project, selected))
    outside_index = build_vector_index(project, outside, _paragraph_chunks(project, outside))
    scope_id = project.snapshot([selected])
    query = (0.987654, 0.012346)
    query_hash = hashlib.sha256(struct.pack("<2f", *query)).hexdigest()

    vector_results = project.search_vectors_with_provenance(
        scope_id, [selected_index], query, limit=2
    )
    hybrid_results = project.hybrid_search_with_provenance(
        scope_id,
        [selected_index],
        "Apples",
        query,
        limit=2,
        rrf_constant=17,
    )

    assert isinstance(vector_results, RetrievalSearchResult)
    assert isinstance(hybrid_results, RetrievalSearchResult)
    assert vector_results.hits is vector_results and hybrid_results.hits is hybrid_results
    assert vector_results.scope_id == hybrid_results.scope_id == scope_id
    assert vector_results and hybrid_results
    assert all(hit.revision_id == selected for hit in vector_results)
    assert all(hit.revision_id == selected for hit in hybrid_results)

    vector_run = _run_for_result(project, vector_results)
    vector_config = json.loads(vector_run["config_json"])
    assert vector_run["operation"] == "search_vectors_with_provenance"
    assert vector_run["scope_id"] == scope_id
    assert vector_run["retrieval_plan_ref"] == "retrieval/raw-vector-search-v1"
    assert vector_run["candidate_count"] == vector_run["inspected_count"]
    assert vector_run["returned_count"] == len(vector_results)
    assert vector_config["artifact_ids"] == [selected_index]
    assert vector_config["vector_provenance"] == "caller-supplied-raw-vector"
    assert vector_config["query_sha256"] == query_hash
    assert vector_config["query_dimension"] == 2
    assert vector_config["limit"] == 2
    assert "score_definition" in vector_config
    assert json.loads(vector_run["query_terms_json"]) == []
    assert json.dumps(query) not in json.dumps(vector_config)
    assert [row["rank"] for row in _hit_lineage(project, vector_results.retrieval_run_id)] == [
        1,
        2,
    ]
    vector_lineage = _hit_lineage(project, vector_results.retrieval_run_id)
    assert [row["revision_id"] for row in vector_lineage] == [selected, selected]
    assert [row["retrieval_method"] for row in vector_lineage] == [
        "raw-vector",
        "raw-vector",
    ]
    assert [
        (row["artifact_id"], row["unit_id"], row["lexical_rank"], row["vector_rank"])
        for row in vector_lineage
    ] == [
        (hit.artifact_id, hit.chunk_id, None, rank)
        for rank, hit in enumerate(vector_results, start=1)
    ]
    assert [
        (row["start_offset"], row["end_offset"], row["score"])
        for row in vector_lineage
    ] == [(hit.start, hit.end, hit.distance) for hit in vector_results]

    hybrid_run = _run_for_result(project, hybrid_results)
    hybrid_config = json.loads(hybrid_run["config_json"])
    assert hybrid_run["operation"] == "hybrid_search_with_provenance"
    assert hybrid_run["scope_id"] == scope_id
    assert hybrid_run["retrieval_plan_ref"] == "retrieval/raw-hybrid-rrf-v1"
    assert hybrid_run["candidate_count"] == hybrid_run["inspected_count"]
    assert hybrid_run["returned_count"] == len(hybrid_results)
    assert hybrid_config["artifact_ids"] == [selected_index]
    assert hybrid_config["vector_provenance"] == "caller-supplied-raw-vector"
    assert hybrid_config["query_sha256"] == hashlib.sha256(b"Apples").hexdigest()
    assert hybrid_config["query_vector_sha256"] == query_hash
    assert hybrid_config["query_vector_dimension"] == 2
    assert hybrid_config["query_characters"] == len("Apples")
    assert hybrid_config["limit"] == 2
    assert hybrid_config["rrf_constant"] == 17
    assert "score_definition" in hybrid_config
    assert "Apples" not in hybrid_run["config_json"]
    assert json.dumps(query) not in hybrid_run["config_json"]
    assert json.loads(hybrid_run["query_terms_json"]) == []
    hybrid_lineage = _hit_lineage(project, hybrid_results.retrieval_run_id)
    assert [row["rank"] for row in hybrid_lineage] == list(
        range(1, len(hybrid_results) + 1)
    )
    assert [row["revision_id"] for row in hybrid_lineage] == [selected] * len(
        hybrid_results
    )
    assert [
        (row["artifact_id"], row["unit_id"], row["lexical_rank"], row["vector_rank"])
        for row in hybrid_lineage
    ] == [
        (hit.artifact_id, hit.unit_id, hit.lexical_rank, hit.vector_rank)
        for hit in hybrid_results
    ]
    assert all(
        hit.artifact_id == selected_index
        for hit in hybrid_results
        if hit.vector_rank is not None
    )

    prior_run_count = project.db.execute("SELECT count(*) FROM retrieval_runs").fetchone()[0]
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        search_vectors_with_provenance(project, scope_id, [outside_index], query)
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        hybrid_search_with_provenance(
            project, scope_id, [outside_index], "Apples", query
        )
    assert project.db.execute("SELECT count(*) FROM retrieval_runs").fetchone()[0] == (
        prior_run_count
    )

    legacy_vector_results = search_vectors(project, scope_id, [selected_index], query)
    legacy_hybrid_results = hybrid_search(
        project, scope_id, [selected_index], "Apples", query
    )
    assert type(legacy_vector_results) is list
    assert type(legacy_hybrid_results) is list
    assert outside not in [hit.revision_id for hit in legacy_vector_results]
    assert outside not in [hit.revision_id for hit in legacy_hybrid_results]


@pytest.mark.parametrize("operation", ["vector", "hybrid"])
def test_raw_vector_searches_persist_empty_runs_without_hits(tmp_path, monkeypatch, operation):
    project = Project.init(tmp_path / f"empty-{operation}-provenance")
    _, revision = project.import_text("empty.txt", "No phrase matches this candidate.")
    artifact_id = build_vector_index(project, revision, _paragraph_chunks(project, revision))
    scope_id = project.snapshot([revision])
    monkeypatch.setattr(
        "discourselens.vectors._search_vectors_with_candidate_count",
        lambda *_args, **_kwargs: ([], 0, "cosine"),
    )

    if operation == "vector":
        result = search_vectors_with_provenance(
            project, scope_id, [artifact_id], (0.5, 0.5)
        )
        expected_operation = "search_vectors_with_provenance"
    else:
        result = hybrid_search_with_provenance(
            project, scope_id, [artifact_id], "MissingPhrase", (0.5, 0.5)
        )
        expected_operation = "hybrid_search_with_provenance"

    assert result == []
    assert result.scope_id == scope_id
    run = _run_for_result(project, result)
    assert run["operation"] == expected_operation
    assert run["candidate_count"] == run["inspected_count"] == run["returned_count"] == 0
    assert project.db.execute(
        "SELECT count(*) FROM retrieval_hits WHERE retrieval_run_id=?",
        (result.retrieval_run_id,),
    ).fetchone()[0] == 0
    assert project.db.execute(
        "SELECT count(*) FROM retrieval_hit_context c JOIN retrieval_hits h "
        "ON h.id=c.retrieval_hit_id WHERE h.retrieval_run_id=?",
        (result.retrieval_run_id,),
    ).fetchone()[0] == 0


def test_raw_hybrid_vector_only_hit_resolves_arbitrary_chunk_span(tmp_path):
    project = Project.init(tmp_path / "arbitrary-chunk-provenance")
    _, revision = project.import_text("source.txt", "The matching phrase is present.")
    chunk_start = 4
    chunk_end = 13
    artifact_id = build_vector_index(
        project,
        revision,
        [VectorChunk("caller-defined-chunk", chunk_start, chunk_end, (1.0, 0.0))],
    )
    scope_id = project.snapshot([revision])

    result = hybrid_search_with_provenance(
        project,
        scope_id,
        [artifact_id],
        "NoLexicalMatch",
        (1.0, 0.0),
    )

    assert len(result) == 1
    hit = result[0]
    assert hit.unit_id == "caller-defined-chunk"
    assert hit.vector_rank == 1
    assert hit.lexical_rank is None
    row = _hit_lineage(project, result.retrieval_run_id)[0]
    assert (row["revision_id"], row["start_offset"], row["end_offset"]) == (
        revision,
        chunk_start,
        chunk_end,
    )
    assert (row["artifact_id"], row["unit_id"], row["vector_rank"]) == (
        artifact_id,
        "caller-defined-chunk",
        1,
    )
