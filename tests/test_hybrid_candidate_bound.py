import json

import pytest

from discourselens import Project
from discourselens.errors import ScopeViolationError
from discourselens.types import CoverageMode, CoverageTerminationState
from discourselens.vectors import (
    HybridHit,
    VectorChunk,
    build_vector_index,
    hybrid_search,
    hybrid_search_with_provenance,
    search_vectors,
)


def _exhaustive_rrf(project, lexical_rows, vector_hits, *, limit: int, rrf_constant: int):
    candidates: dict[tuple[str, int, int], list[int | None]] = {}
    unit_by_span: dict[tuple[str, int, int], str] = {}
    artifact_by_span: dict[tuple[str, int, int], str] = {}
    paragraph_ids = list(dict.fromkeys(str(row["paragraph_id"]) for row in lexical_rows))
    placeholders = ",".join("?" for _ in paragraph_ids)
    end_offsets = {
        str(row["id"]): int(row["end_offset"])
        for row in project.db.execute(
            f"SELECT id,end_offset FROM paragraphs WHERE id IN ({placeholders})", paragraph_ids
        )
    }
    for rank, row in enumerate(lexical_rows, start=1):
        span = (
            row["revision_id"],
            int(row["start_offset"]),
            end_offsets[str(row["paragraph_id"])],
        )
        candidates[span] = [rank, None]
        unit_by_span[span] = row["paragraph_id"]
    for rank, hit in enumerate(vector_hits, start=1):
        span = (hit.revision_id, hit.start, hit.end)
        ranks = candidates.setdefault(span, [None, None])
        if ranks[1] is None or rank < ranks[1]:
            ranks[1] = rank
            unit_by_span[span] = hit.chunk_id
            artifact_by_span[span] = hit.artifact_id

    ranked = [
        HybridHit(
            revision_id=span[0],
            unit_id=unit_by_span[span],
            lexical_rank=ranks[0],
            vector_rank=ranks[1],
            fusion_score=sum(1 / (rrf_constant + rank) for rank in ranks if rank is not None),
            artifact_id=(artifact_by_span[span] if ranks[1] is not None else None),
        )
        for span, ranks in candidates.items()
    ]
    return sorted(ranked, key=lambda hit: (-hit.fusion_score, hit.revision_id, hit.unit_id))[:limit]


def test_hybrid_search_bounds_lexical_retention_and_preserves_exhaustive_rrf(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "hybrid-candidate-bound")
    paragraphs_text = [f"green policy topic{index} {'context ' * index}" for index in range(40)]
    _, selected_revision = project.import_text("selected.txt", "\n\n".join(paragraphs_text))
    _, outside_revision = project.import_text("outside.txt", "green policy outside")
    selected_scope = project.snapshot([selected_revision])

    lexical_rows = project.search_exact_ranked("green policy", selected_scope)
    assert len(lexical_rows) == 40
    lexical_rank_by_unit = {
        row["paragraph_id"]: rank for rank, row in enumerate(lexical_rows, start=1)
    }
    limit = 4
    target_lexical_rank = limit + 2
    vector_by_rank = {
        target_lexical_rank: (1.0, 0.0),
        2: (0.99, 0.14),
        3: (0.96, 0.28),
        1: (0.8, 0.6),
    }
    chunks = [
        VectorChunk(
            str(row["paragraph_id"]),
            int(row["start_offset"]),
            int(row["end_offset"]),
            vector_by_rank.get(lexical_rank_by_unit[str(row["paragraph_id"])], (0.0, 1.0)),
        )
        for row in project.db.execute(
            "SELECT id AS paragraph_id,start_offset,end_offset FROM paragraphs "
            "WHERE revision_id=? AND end_offset>start_offset ORDER BY ordinal",
            (selected_revision,),
        )
    ]
    selected_index = build_vector_index(project, selected_revision, chunks)
    vector_hits = search_vectors(project, selected_scope, [selected_index], (1.0, 0.0), limit=limit)
    target_unit = str(lexical_rows[target_lexical_rank - 1]["paragraph_id"])
    assert vector_hits[0].chunk_id == target_unit

    lexical_head, additional_ranks, lexical_population = project._search_exact_ranked_candidates(
        "green policy",
        selected_scope,
        limit=limit,
        additional_spans={(hit.revision_id, hit.start, hit.end) for hit in vector_hits},
    )
    assert len(lexical_head) == limit
    assert [row["paragraph_id"] for row in lexical_head] == [
        row["paragraph_id"] for row in lexical_rows[:limit]
    ]
    assert lexical_population == len(lexical_rows)
    target_span = (selected_revision, vector_hits[0].start, vector_hits[0].end)
    assert additional_ranks[target_span] == target_lexical_rank

    rrf_constant = 17
    expected = _exhaustive_rrf(
        project, lexical_rows, vector_hits, limit=limit, rrf_constant=rrf_constant
    )

    def forbidden_exhaustive_search(*_args, **_kwargs):
        pytest.fail("hybrid search must use its bounded lexical retention path")

    monkeypatch.setattr(project, "search_exact_ranked", forbidden_exhaustive_search)
    actual = hybrid_search(
        project,
        selected_scope,
        [selected_index],
        "green policy",
        (1.0, 0.0),
        limit=limit,
        rrf_constant=rrf_constant,
    )
    assert [
        (hit.revision_id, hit.unit_id, hit.lexical_rank, hit.vector_rank) for hit in actual
    ] == [(hit.revision_id, hit.unit_id, hit.lexical_rank, hit.vector_rank) for hit in expected]
    assert [hit.fusion_score for hit in actual] == pytest.approx(
        [hit.fusion_score for hit in expected]
    )
    target_hit = next(hit for hit in actual if hit.unit_id == target_unit)
    assert target_hit.lexical_rank == target_lexical_rank
    assert target_hit.vector_rank == 1

    results = hybrid_search_with_provenance(
        project,
        selected_scope,
        [selected_index],
        "green policy",
        (1.0, 0.0),
        limit=limit,
        rrf_constant=rrf_constant,
    )
    run = project.db.execute(
        "SELECT r.*,c.config_json FROM retrieval_runs r "
        "JOIN retrieval_run_context c ON c.retrieval_run_id=r.id WHERE r.id=?",
        (results.retrieval_run_id,),
    ).fetchone()
    assert run is not None
    config = json.loads(run["config_json"])
    assert config["lexical_candidate_limit"] == limit
    assert config["lexical_candidate_count"] == len(lexical_rows)
    assert config["lexical_candidates_retained"] == limit
    assert config["lexical_candidate_limit_reached"] is True
    assert config["lexical_candidate_population_complete"] is True
    assert config["lexical_stream_fully_inspected"] is True
    assert "complete exact lexical stream" in config["lexical_retention_policy"]
    assert run["candidate_count"] == run["inspected_count"] == len(lexical_rows) + limit
    assert run["candidate_limit_reached"] == 1
    coverage = project.coverage_descriptor(run["coverage_id"])
    assert coverage.coverage_mode is CoverageMode.TARGETED
    assert coverage.inspection_count == run["inspected_count"]
    assert coverage.termination_state is CoverageTerminationState.LIMIT_REACHED

    outside_rows = project.db.execute(
        "SELECT id,start_offset,end_offset FROM paragraphs "
        "WHERE revision_id=? AND end_offset>start_offset",
        (outside_revision,),
    ).fetchall()
    outside_index = build_vector_index(
        project,
        outside_revision,
        [
            VectorChunk(
                str(row["id"]),
                int(row["start_offset"]),
                int(row["end_offset"]),
                (0.0, 1.0),
            )
            for row in outside_rows
        ],
    )
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        hybrid_search(
            project,
            selected_scope,
            [outside_index],
            "green policy",
            (1.0, 0.0),
            limit=limit,
        )


def test_hybrid_does_not_merge_same_unit_id_across_different_spans(tmp_path):
    project = Project.init(tmp_path / "hybrid-unit-id-collision")
    _, revision = project.import_text(
        "collision.txt", "green policy lexical paragraph\n\nvector-only paragraph"
    )
    paragraph_rows = project.db.execute(
        "SELECT id,ordinal,start_offset,end_offset FROM paragraphs "
        "WHERE revision_id=? AND end_offset>start_offset ORDER BY ordinal",
        (revision,),
    ).fetchall()
    assert len(paragraph_rows) == 2
    lexical_paragraph, vector_paragraph = paragraph_rows
    reused_id = "reused-unit-id"
    project.db.execute(
        "UPDATE paragraphs SET id=? WHERE id=?", (reused_id, lexical_paragraph["id"])
    )
    project.db.execute(
        "UPDATE revision_fts SET paragraph_id=? WHERE paragraph_id=?",
        (reused_id, lexical_paragraph["id"]),
    )
    project.db.commit()
    scope_id = project.snapshot([revision])
    artifact_id = build_vector_index(
        project,
        revision,
        [
            VectorChunk(
                reused_id,
                int(vector_paragraph["start_offset"]),
                int(vector_paragraph["end_offset"]),
                (1.0, 0.0),
            )
        ],
    )

    hits = hybrid_search(project, scope_id, [artifact_id], "green policy", (1.0, 0.0), limit=2)
    assert len(hits) == 2
    assert [(hit.unit_id, hit.lexical_rank, hit.vector_rank, hit.artifact_id) for hit in hits] == [
        (reused_id, 1, None, None),
        (reused_id, None, 1, artifact_id),
    ]

    result = hybrid_search_with_provenance(
        project, scope_id, [artifact_id], "green policy", (1.0, 0.0), limit=2
    )
    lineage = project.db.execute(
        "SELECT s.start_offset,s.end_offset,c.unit_id,c.lexical_rank,c.vector_rank,c.artifact_id "
        "FROM retrieval_hits h JOIN source_spans s ON s.id=h.source_span_id "
        "JOIN retrieval_hit_context c ON c.retrieval_hit_id=h.id "
        "WHERE h.retrieval_run_id=? ORDER BY h.rank",
        (result.retrieval_run_id,),
    ).fetchall()
    assert [
        (
            row["start_offset"],
            row["end_offset"],
            row["unit_id"],
            row["lexical_rank"],
            row["vector_rank"],
            row["artifact_id"],
        )
        for row in lineage
    ] == [
        (
            lexical_paragraph["start_offset"],
            lexical_paragraph["end_offset"],
            reused_id,
            1,
            None,
            None,
        ),
        (
            vector_paragraph["start_offset"],
            vector_paragraph["end_offset"],
            reused_id,
            None,
            1,
            artifact_id,
        ),
    ]


def test_hybrid_merges_equal_spans_and_keeps_vector_identity(tmp_path):
    project = Project.init(tmp_path / "hybrid-equal-span")
    _, revision = project.import_text("equal-span.txt", "green policy canonical paragraph")
    paragraph = project.db.execute(
        "SELECT start_offset,end_offset FROM paragraphs WHERE revision_id=?",
        (revision,),
    ).fetchone()
    assert paragraph is not None
    scope_id = project.snapshot([revision])
    vector_unit_id = "caller-defined-span-id"
    artifact_id = build_vector_index(
        project,
        revision,
        [
            VectorChunk(
                vector_unit_id,
                int(paragraph["start_offset"]),
                int(paragraph["end_offset"]),
                (1.0, 0.0),
            )
        ],
    )

    result = hybrid_search_with_provenance(
        project, scope_id, [artifact_id], "green policy", (1.0, 0.0), limit=1
    )
    assert len(result) == 1
    hit = result[0]
    assert (hit.unit_id, hit.lexical_rank, hit.vector_rank, hit.artifact_id) == (
        vector_unit_id,
        1,
        1,
        artifact_id,
    )
    lineage = project.db.execute(
        "SELECT s.start_offset,s.end_offset,c.unit_id,c.lexical_rank,c.vector_rank,c.artifact_id "
        "FROM retrieval_hits h JOIN source_spans s ON s.id=h.source_span_id "
        "JOIN retrieval_hit_context c ON c.retrieval_hit_id=h.id "
        "WHERE h.retrieval_run_id=?",
        (result.retrieval_run_id,),
    ).fetchone()
    assert lineage is not None
    assert (
        lineage["start_offset"],
        lineage["end_offset"],
        lineage["unit_id"],
        lineage["lexical_rank"],
        lineage["vector_rank"],
        lineage["artifact_id"],
    ) == (
        paragraph["start_offset"],
        paragraph["end_offset"],
        vector_unit_id,
        1,
        1,
        artifact_id,
    )
