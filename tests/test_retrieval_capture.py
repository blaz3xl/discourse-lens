from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import jsonschema
import pytest

from benchmarks.retrieval_capture import (
    CaptureQuery,
    VectorCaptureQuery,
    capture_lexical_rankings,
    capture_vector_rankings,
)
from benchmarks.retrieval_eval import evaluate_payloads
from discourselens import Project
from discourselens.vectors import VectorChunk, build_vector_index

ROOT = Path(__file__).resolve().parents[1]


def test_lexical_capture_is_frozen_scope_bound_and_evaluable(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, selected_revision = project.import_text(
            "selected.txt", "The shared marker appears in the selected report."
        )
        scope_id = project.snapshot([selected_revision])
        _, excluded_revision = project.import_text(
            "excluded.txt", "The shared marker appears in the excluded report."
        )

        query = CaptureQuery("q-shared", "shared marker", scope_id)
        results = capture_lexical_rankings(
            project, [query], retriever_id="public-lexical", limit=10
        )
        repeated = capture_lexical_rankings(
            project, [query], retriever_id="public-lexical", limit=10
        )

        assert results == repeated
        retrievers = cast(list[dict[str, Any]], results["retrievers"])
        rows = cast(list[dict[str, Any]], retrievers[0]["query_results"])
        ranked_units = cast(list[dict[str, Any]], rows[0]["results"])
        assert ranked_units == [{"unit_id": selected_revision, "rank": 1}]
        assert excluded_revision not in {item["unit_id"] for item in ranked_units}

        schema = json.loads(
            (ROOT / "benchmarks" / "schemas" / "retrieval-ranking-results-v1.json")
            .read_text(encoding="utf-8")
        )
        jsonschema.validate(results, schema)

        judgments = {
            "schema_version": "retrieval-judgments-v1",
            "unit_kind": "DOCUMENT_REVISION",
            "queries": [
                {
                    "query_id": "q-shared",
                    "judgments": [
                        {"unit_id": selected_revision, "grade": 3},
                        {"unit_id": excluded_revision, "grade": 1},
                    ],
                }
            ],
        }
        evaluation = evaluate_payloads(judgments, results, ks=(1,))
        metrics = cast(dict[str, Any], evaluation["metrics"])["public-lexical"]
        assert metrics["mean_reciprocal_rank"] == pytest.approx(1.0)
        assert metrics["recall_at"] == {"1": 0.5}
    finally:
        project.close()


def test_lexical_capture_rejects_duplicate_query_ids(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("selected.txt", "A shared marker.")
        scope_id = project.snapshot([revision])
        duplicate_queries = [
            CaptureQuery("same", "shared marker", scope_id),
            CaptureQuery("same", "shared marker", scope_id),
        ]

        with pytest.raises(ValueError, match="duplicate query_id"):
            capture_lexical_rankings(
                project, duplicate_queries, retriever_id="public-lexical"
            )
    finally:
        project.close()


def test_vector_capture_preserves_frozen_scope_and_evaluator_contract(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, selected_revision = project.import_text(
            "selected.txt", "Apples grow in the orchard."
        )
        _, excluded_revision = project.import_text(
            "excluded.txt", "A closer vector belongs to another document."
        )
        selected_artifact = build_vector_index(
            project,
            selected_revision,
            [VectorChunk("selected-chunk", 0, 6, (0.1, 0.0))],
        )
        excluded_artifact = build_vector_index(
            project,
            excluded_revision,
            [VectorChunk("excluded-chunk", 0, 7, (0.01, 0.0))],
        )
        query = VectorCaptureQuery(
            "q-orchard",
            project.snapshot([selected_revision]),
            (selected_artifact,),
            (1.0, 0.0),
        )

        results = capture_vector_rankings(
            project, [query], retriever_id="fixture-vector", limit=10
        )
        retrievers = cast(list[dict[str, Any]], results["retrievers"])
        rows = cast(list[dict[str, Any]], retrievers[0]["query_results"])
        ranked_units = cast(list[dict[str, Any]], rows[0]["results"])
        assert ranked_units == [{"unit_id": selected_revision, "rank": 1}]
        assert excluded_revision not in {item["unit_id"] for item in ranked_units}
        assert excluded_artifact not in query.artifact_ids

        schema = json.loads(
            (ROOT / "benchmarks" / "schemas" / "retrieval-ranking-results-v1.json")
            .read_text(encoding="utf-8")
        )
        jsonschema.validate(results, schema)

        judgments = {
            "schema_version": "retrieval-judgments-v1",
            "unit_kind": "DOCUMENT_REVISION",
            "queries": [
                {
                    "query_id": "q-orchard",
                    "judgments": [
                        {"unit_id": selected_revision, "grade": 3},
                        {"unit_id": excluded_revision, "grade": 1},
                    ],
                }
            ],
        }
        metrics = cast(
            dict[str, Any],
            cast(dict[str, Any], evaluate_payloads(judgments, results, ks=(1,))["metrics"])[
                "fixture-vector"
            ],
        )
        assert metrics["mean_reciprocal_rank"] == pytest.approx(1.0)
        assert metrics["recall_at"] == {"1": 0.5}
    finally:
        project.close()


def test_vector_capture_rejects_duplicate_query_ids(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "A source paragraph.")
        artifact = build_vector_index(
            project,
            revision,
            [VectorChunk("source", 0, 1, (1.0, 0.0))],
        )
        scope_id = project.snapshot([revision])
        query = VectorCaptureQuery("duplicate", scope_id, (artifact,), (1.0, 0.0))
        with pytest.raises(ValueError, match="duplicate query_id"):
            capture_vector_rankings(
                project, [query, query], retriever_id="fixture-vector"
            )
    finally:
        project.close()


def test_vector_capture_ranks_best_chunk_per_revision_before_global_limit(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision_a = project.import_text("a.txt", "abcdefghij")
        _, revision_b = project.import_text("b.txt", "klmnopqrst")
        _, revision_c = project.import_text("c.txt", "uvwxyz1234")
        artifact_a = build_vector_index(
            project,
            revision_a,
            [
                VectorChunk("a-best", 0, 2, (1.0, 0.001)),
                VectorChunk("a-second", 2, 4, (1.0, 0.01)),
            ],
        )
        artifact_b = build_vector_index(
            project,
            revision_b,
            [VectorChunk("b-best", 0, 2, (0.99, 0.1))],
        )
        artifact_c = build_vector_index(
            project,
            revision_c,
            [VectorChunk("c-best", 0, 2, (0.9, 0.44))],
        )
        scope_id = project.snapshot([revision_a, revision_b, revision_c])

        results = capture_vector_rankings(
            project,
            [
                VectorCaptureQuery(
                    "q-all",
                    scope_id,
                    (artifact_a, artifact_b, artifact_c),
                    (1.0, 0.0),
                )
            ],
            retriever_id="fixture-vector",
            limit=3,
        )
        retrievers = cast(list[dict[str, Any]], results["retrievers"])
        rows = cast(list[dict[str, Any]], retrievers[0]["query_results"])
        ranked_units = cast(list[dict[str, Any]], rows[0]["results"])
        assert ranked_units == [
            {"unit_id": revision_a, "rank": 1},
            {"unit_id": revision_b, "rank": 2},
            {"unit_id": revision_c, "rank": 3},
        ]
    finally:
        project.close()
