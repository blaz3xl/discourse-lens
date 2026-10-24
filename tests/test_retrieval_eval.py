from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import jsonschema
import pytest

from benchmarks.retrieval_eval import EvaluationError, evaluate_files, evaluate_payloads

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "benchmarks" / "datasets" / "retrieval-ranking-smoke"


def _judgments() -> dict[str, object]:
    return {
        "schema_version": "retrieval-judgments-v1",
        "unit_kind": "CHUNK",
        "chunker_provenance": {
            "name": "test-chunker",
            "version": "1",
            "configuration": "unit-tests",
        },
        "queries": [
            {
                "query_id": "q1",
                "judgments": [
                    {"unit_id": "unit-a", "grade": 3},
                    {"unit_id": "unit-b", "grade": 1},
                ],
            }
        ],
    }


def _rankings(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "retrieval-ranking-results-v1",
        "unit_kind": "CHUNK",
        "chunker_provenance": {
            "name": "test-chunker",
            "version": "1",
            "configuration": "unit-tests",
        },
        "retrievers": [
            {
                "retriever_id": "test",
                "query_results": [{"query_id": "q1", "results": results}],
            }
        ],
    }


def test_synthetic_fixture_hand_checkable_metrics_and_empty_policy() -> None:
    excluded = evaluate_files(
        FIXTURE_DIR / "judgments-v1.json",
        FIXTURE_DIR / "results-v1.json",
        ks=(1, 2, 3),
    )
    metrics = cast(dict[str, Any], excluded["metrics"])["synthetic-order"]
    assert excluded["unit_kind"] == "CHUNK"
    assert excluded["chunker_provenance"] == {
        "name": "synthetic-smoke-fixed-chunker",
        "version": "1",
        "configuration": "fixed-hand-authored-unit-ids",
    }
    assert metrics["queries_total"] == 3
    assert metrics["queries_with_relevance"] == 2
    assert metrics["queries_scored"] == 2
    assert metrics["queries_excluded_for_empty_relevance"] == 1
    assert metrics["mean_reciprocal_rank"] == pytest.approx(0.25)
    assert metrics["recall_at"] == {"1": 0.0, "2": 0.25, "3": 0.5}

    expected_q1_ndcg_at_2 = (1 / math.log2(3)) / (7 + 1 / math.log2(3))
    expected_q1_ndcg_at_3 = (1 / math.log2(3) + 7 / math.log2(4)) / (
        7 + 1 / math.log2(3)
    )
    assert metrics["ndcg_at"]["1"] == pytest.approx(0.0)
    assert metrics["ndcg_at"]["2"] == pytest.approx(expected_q1_ndcg_at_2 / 2)
    assert metrics["ndcg_at"]["3"] == pytest.approx(expected_q1_ndcg_at_3 / 2)

    zeroed = evaluate_files(
        FIXTURE_DIR / "judgments-v1.json",
        FIXTURE_DIR / "results-v1.json",
        ks=(3,),
        empty_relevance_policy="zero",
    )
    zero_metrics = cast(dict[str, Any], zeroed["metrics"])["synthetic-order"]
    assert zero_metrics["queries_scored"] == 3
    assert zero_metrics["queries_excluded_for_empty_relevance"] == 0
    assert zero_metrics["mean_reciprocal_rank"] == pytest.approx(1 / 6)
    assert zero_metrics["recall_at"]["3"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("queries", "message"),
    [
        (
            [
                {"query_id": "q1", "judgments": []},
                {"query_id": "q1", "judgments": []},
            ],
            "duplicate query_id",
        ),
        (
            [{"query_id": "q1", "judgments": [{"unit_id": "d", "grade": -1}]}],
            "grade must be an integer",
        ),
        (
            [{"query_id": "q1", "judgments": [{"unit_id": "d", "grade": 4}]}],
            "grade must be an integer",
        ),
        (
            [{"query_id": "q1", "judgments": [{"unit_id": "d", "grade": True}]}],
            "grade must be an integer",
        ),
        (
            [{"query_id": "q1", "judgments": [{"unit_id": "d", "grade": 1.5}]}],
            "grade must be an integer",
        ),
        (
            [
                {
                    "query_id": "q1",
                    "judgments": [
                        {"unit_id": "d", "grade": 1},
                        {"unit_id": "d", "grade": 2},
                    ],
                }
            ],
            "duplicate unit_id",
        ),
    ],
)
def test_invalid_judgments_are_rejected(
    queries: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(EvaluationError, match=message):
        evaluate_payloads(
            {
                "schema_version": "retrieval-judgments-v1",
                "unit_kind": "CHUNK",
                "chunker_provenance": {
                    "name": "test-chunker",
                    "version": "1",
                    "configuration": "unit-tests",
                },
                "queries": queries,
            },
            _rankings([]),
        )


@pytest.mark.parametrize(
    ("query_results", "message"),
    [
        ([{"query_id": "missing", "results": []}], "unknown query_id"),
        (
            [
                {"query_id": "q1", "results": []},
                {"query_id": "q1", "results": []},
            ],
            "duplicate query result row",
        ),
        (
            [
                {
                    "query_id": "q1",
                    "results": [
                        {"unit_id": "d", "rank": 1},
                        {"unit_id": "d", "rank": 2},
                    ],
                }
            ],
            "duplicate unit_id",
        ),
        (
            [
                {
                    "query_id": "q1",
                    "results": [
                        {"unit_id": "d1", "rank": 1},
                        {"unit_id": "d2", "rank": 1},
                    ],
                }
            ],
            "duplicate rank",
        ),
        (
            [{"query_id": "q1", "results": [{"unit_id": "d", "rank": 0}]}],
            "rank must be a positive integer",
        ),
        (
            [{"query_id": "q1", "results": [{"unit_id": "d", "rank": True}]}],
            "rank must be a positive integer",
        ),
        (
            [{"query_id": "q1", "results": [{"unit_id": "d", "rank": 1.5}]}],
            "rank must be a positive integer",
        ),
    ],
)
def test_invalid_ranking_rows_are_rejected(
    query_results: list[dict[str, object]], message: str
) -> None:
    rankings: dict[str, object] = {
        "schema_version": "retrieval-ranking-results-v1",
        "unit_kind": "CHUNK",
        "chunker_provenance": {
            "name": "test-chunker",
            "version": "1",
            "configuration": "unit-tests",
        },
        "retrievers": [{"retriever_id": "test", "query_results": query_results}],
    }
    with pytest.raises(EvaluationError, match=message):
        evaluate_payloads(_judgments(), rankings)


def test_duplicate_retriever_ids_and_invalid_metric_options_are_rejected() -> None:
    duplicate_retrievers = {
        "schema_version": "retrieval-ranking-results-v1",
        "unit_kind": "CHUNK",
        "chunker_provenance": {
            "name": "test-chunker",
            "version": "1",
            "configuration": "unit-tests",
        },
        "retrievers": [
            {"retriever_id": "same", "query_results": []},
            {"retriever_id": "same", "query_results": []},
        ],
    }
    with pytest.raises(EvaluationError, match="duplicate retriever_id"):
        evaluate_payloads(_judgments(), duplicate_retrievers)
    with pytest.raises(EvaluationError, match="duplicate k"):
        evaluate_payloads(_judgments(), _rankings([]), ks=(1, 1))
    with pytest.raises(EvaluationError, match="empty_relevance_policy"):
        evaluate_payloads(
            _judgments(), _rankings([]), empty_relevance_policy="ignore"  # type: ignore[arg-type]
        )


def test_unit_kind_and_chunker_provenance_must_match() -> None:
    mismatched_kind = _rankings([])
    mismatched_kind["unit_kind"] = "SOURCE_SPAN"
    mismatched_kind.pop("chunker_provenance")
    with pytest.raises(EvaluationError, match="unit_kind and results.unit_kind must match"):
        evaluate_payloads(_judgments(), mismatched_kind)

    mismatched_provenance = _rankings([])
    provenance = cast(dict[str, str], mismatched_provenance["chunker_provenance"])
    provenance["configuration"] = "different-chunking"
    with pytest.raises(EvaluationError, match="chunker_provenance must match"):
        evaluate_payloads(_judgments(), mismatched_provenance)

    missing_provenance = _judgments()
    missing_provenance.pop("chunker_provenance")
    with pytest.raises(EvaluationError, match="missing keys: chunker_provenance"):
        evaluate_payloads(missing_provenance, _rankings([]))


@pytest.mark.parametrize("unit_kind", ["DOCUMENT_REVISION", "SOURCE_SPAN"])
def test_non_chunk_unit_kinds_do_not_require_chunker_provenance(unit_kind: str) -> None:
    judgments = _judgments()
    rankings = _rankings([])
    judgments["unit_kind"] = unit_kind
    rankings["unit_kind"] = unit_kind
    judgments.pop("chunker_provenance")
    rankings.pop("chunker_provenance")
    report = evaluate_payloads(judgments, rankings, ks=(1,))
    assert report["unit_kind"] == unit_kind
    assert "chunker_provenance" not in report


def test_duplicate_json_object_keys_are_rejected(tmp_path: Path) -> None:
    judgments_path = tmp_path / "judgments.json"
    rankings_path = tmp_path / "rankings.json"
    judgments_path.write_text(
        '{"schema_version":"retrieval-judgments-v1",'
        '"unit_kind":"CHUNK","chunker_provenance":'
        '{"name":"x","version":"1","configuration":"test"},'
        '"schema_version":"retrieval-judgments-v1","queries":[]}',
        encoding="utf-8",
    )
    rankings_path.write_text(json.dumps(_rankings([])), encoding="utf-8")
    with pytest.raises(EvaluationError, match="duplicate JSON object key"):
        evaluate_files(judgments_path, rankings_path)


@pytest.mark.parametrize(
    ("fixture_name", "schema_name"),
    [
        ("judgments-v1.json", "retrieval-judgments-v1.json"),
        ("results-v1.json", "retrieval-ranking-results-v1.json"),
    ],
)
def test_versioned_schemas_accept_synthetic_fixtures(
    fixture_name: str, schema_name: str
) -> None:
    schema_path = ROOT / "benchmarks" / "schemas" / schema_name
    fixture_path = FIXTURE_DIR / fixture_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(fixture, schema)
