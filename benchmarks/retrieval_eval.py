"""Offline, mechanical evaluation of ranked retrieval results against graded judgments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, TextIO, cast

JUDGMENTS_VERSION = "retrieval-judgments-v1"
RESULTS_VERSION = "retrieval-ranking-results-v1"
PROTOCOL_VERSION = "retrieval-ranking-eval-v1"
MAX_GRADE = 3
DEFAULT_KS = (1, 3, 5)
EmptyRelevancePolicy = Literal["exclude", "zero"]
UnitKind = Literal["DOCUMENT_REVISION", "SOURCE_SPAN", "CHUNK"]
ChunkerProvenance = tuple[str, str, str]
RankedUnit = tuple[int, str]
Judgments = dict[str, dict[str, int]]
Rankings = dict[str, dict[str, tuple[RankedUnit, ...]]]


@dataclass(frozen=True)
class JudgmentDataset:
    unit_kind: UnitKind
    chunker_provenance: ChunkerProvenance | None
    grades_by_query: Judgments


@dataclass(frozen=True)
class RankingDataset:
    unit_kind: UnitKind
    chunker_provenance: ChunkerProvenance | None
    rankings_by_retriever: Rankings


class EvaluationError(ValueError):
    """Raised when an input file or evaluation configuration violates the protocol."""


def _fail_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(path: str | Path, label: str) -> object:
    source_path = Path(path)
    try:
        raw = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvaluationError(f"cannot read {label} file {source_path}: {exc}") from exc
    try:
        return json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_fail_constant,
        )
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise EvaluationError(f"{source_path}: invalid JSON: {exc}") from exc


def _record(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluationError(f"{location}: expected an object with string keys")
    return cast(dict[str, object], value)


def _exact_keys(value: dict[str, object], expected: set[str], location: str) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    details: list[str] = []
    if missing:
        details.append("missing keys: " + ", ".join(missing))
    if extra:
        details.append("unknown keys: " + ", ".join(extra))
    if details:
        raise EvaluationError(f"{location}: " + "; ".join(details))


def _identifier(value: object, name: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{location}: {name} must be a non-empty string")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise EvaluationError(f"{location}: {name} must contain valid Unicode scalar values")
    return value


def _parse_unit_header(
    root: dict[str, object], *, label: str, version: str, body_key: str
) -> tuple[UnitKind, ChunkerProvenance | None]:
    if root.get("schema_version") != version:
        raise EvaluationError(f"{label}.schema_version must be {version!r}")
    raw_kind = root.get("unit_kind")
    allowed_kinds = ("DOCUMENT_REVISION", "SOURCE_SPAN", "CHUNK")
    if not isinstance(raw_kind, str) or raw_kind not in allowed_kinds:
        raise EvaluationError(f"{label}.unit_kind must be one of: {', '.join(allowed_kinds)}")
    unit_kind = cast(UnitKind, raw_kind)

    expected = {"schema_version", "unit_kind", body_key}
    if unit_kind == "CHUNK":
        expected.add("chunker_provenance")
    _exact_keys(root, expected, label)

    if unit_kind != "CHUNK":
        return unit_kind, None
    raw_provenance = _record(root["chunker_provenance"], f"{label}.chunker_provenance")
    _exact_keys(
        raw_provenance,
        {"name", "version", "configuration"},
        f"{label}.chunker_provenance",
    )
    provenance = tuple(
        _identifier(raw_provenance[key], key, f"{label}.chunker_provenance")
        for key in ("name", "version", "configuration")
    )
    return unit_kind, cast(ChunkerProvenance, provenance)


def parse_judgments(payload: object) -> JudgmentDataset:
    """Validate versioned graded judgments and return their ranked-unit metadata."""
    root = _record(payload, "judgments")
    unit_kind, chunker_provenance = _parse_unit_header(
        root, label="judgments", version=JUDGMENTS_VERSION, body_key="queries"
    )
    raw_queries = root["queries"]
    if not isinstance(raw_queries, list) or not raw_queries:
        raise EvaluationError("judgments.queries must be a non-empty array")

    cases: Judgments = {}
    for query_index, raw_query in enumerate(raw_queries):
        location = f"judgments.queries[{query_index}]"
        query = _record(raw_query, location)
        _exact_keys(query, {"query_id", "judgments"}, location)
        query_id = _identifier(query["query_id"], "query_id", location)
        if query_id in cases:
            raise EvaluationError(f"{location}: duplicate query_id {query_id!r}")
        raw_judgments = query["judgments"]
        if not isinstance(raw_judgments, list):
            raise EvaluationError(f"{location}.judgments must be an array")
        grades: dict[str, int] = {}
        for judgment_index, raw_judgment in enumerate(raw_judgments):
            judgment_location = f"{location}.judgments[{judgment_index}]"
            judgment = _record(raw_judgment, judgment_location)
            _exact_keys(judgment, {"unit_id", "grade"}, judgment_location)
            unit_id = _identifier(judgment["unit_id"], "unit_id", judgment_location)
            if unit_id in grades:
                raise EvaluationError(
                    f"{judgment_location}: duplicate unit_id {unit_id!r}"
                )
            grade = judgment["grade"]
            if type(grade) is not int or not 0 <= grade <= MAX_GRADE:
                raise EvaluationError(
                    f"{judgment_location}: grade must be an integer from 0 to {MAX_GRADE}"
                )
            grades[unit_id] = grade
        cases[query_id] = grades
    return JudgmentDataset(unit_kind, chunker_provenance, cases)


def parse_rankings(payload: object, known_query_ids: set[str]) -> RankingDataset:
    """Validate versioned per-retriever ranking rows, including their query references."""
    root = _record(payload, "results")
    unit_kind, chunker_provenance = _parse_unit_header(
        root, label="results", version=RESULTS_VERSION, body_key="retrievers"
    )
    raw_retrievers = root["retrievers"]
    if not isinstance(raw_retrievers, list) or not raw_retrievers:
        raise EvaluationError("results.retrievers must be a non-empty array")

    rankings: Rankings = {}
    for retriever_index, raw_retriever in enumerate(raw_retrievers):
        location = f"results.retrievers[{retriever_index}]"
        retriever = _record(raw_retriever, location)
        _exact_keys(retriever, {"retriever_id", "query_results"}, location)
        retriever_id = _identifier(retriever["retriever_id"], "retriever_id", location)
        if retriever_id in rankings:
            raise EvaluationError(f"{location}: duplicate retriever_id {retriever_id!r}")
        raw_query_results = retriever["query_results"]
        if not isinstance(raw_query_results, list):
            raise EvaluationError(f"{location}.query_results must be an array")
        by_query: dict[str, tuple[RankedUnit, ...]] = {}
        for row_index, raw_row in enumerate(raw_query_results):
            row_location = f"{location}.query_results[{row_index}]"
            row = _record(raw_row, row_location)
            _exact_keys(row, {"query_id", "results"}, row_location)
            query_id = _identifier(row["query_id"], "query_id", row_location)
            if query_id not in known_query_ids:
                raise EvaluationError(f"{row_location}: unknown query_id {query_id!r}")
            if query_id in by_query:
                raise EvaluationError(f"{row_location}: duplicate query result row for {query_id!r}")
            raw_results = row["results"]
            if not isinstance(raw_results, list):
                raise EvaluationError(f"{row_location}.results must be an array")
            seen_units: set[str] = set()
            seen_ranks: set[int] = set()
            ranked_units: list[RankedUnit] = []
            for result_index, raw_result in enumerate(raw_results):
                result_location = f"{row_location}.results[{result_index}]"
                result = _record(raw_result, result_location)
                _exact_keys(result, {"unit_id", "rank"}, result_location)
                unit_id = _identifier(result["unit_id"], "unit_id", result_location)
                if unit_id in seen_units:
                    raise EvaluationError(f"{result_location}: duplicate unit_id {unit_id!r}")
                seen_units.add(unit_id)
                rank = result["rank"]
                if type(rank) is not int or rank < 1:
                    raise EvaluationError(f"{result_location}: rank must be a positive integer")
                if rank in seen_ranks:
                    raise EvaluationError(f"{result_location}: duplicate rank {rank}")
                seen_ranks.add(rank)
                ranked_units.append((rank, unit_id))
            by_query[query_id] = tuple(sorted(ranked_units))
        rankings[retriever_id] = by_query
    return RankingDataset(unit_kind, chunker_provenance, rankings)


def _normalize_ks(ks: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in ks:
        if type(value) is not int or value < 1:
            raise EvaluationError("each k must be a positive integer")
        if value in normalized:
            raise EvaluationError(f"duplicate k value {value}")
        normalized.append(value)
    if not normalized:
        raise EvaluationError("at least one k value is required")
    return tuple(sorted(normalized))


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_payloads(
    judgments_payload: object,
    rankings_payload: object,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    empty_relevance_policy: EmptyRelevancePolicy = "exclude",
) -> dict[str, object]:
    """Evaluate ranking payloads and return JSON-serializable aggregate metrics.

    Grades 1 through 3 are relevant; grade 0 and unjudged documents are nonrelevant.
    Empty-relevance queries are excluded from all means or scored as zero according to
    ``empty_relevance_policy``. A missing result row is equivalent to an empty ranking.
    """
    if empty_relevance_policy not in ("exclude", "zero"):
        raise EvaluationError("empty_relevance_policy must be 'exclude' or 'zero'")
    selected_ks = _normalize_ks(ks)
    cases = parse_judgments(judgments_payload)
    result_set = parse_rankings(rankings_payload, set(cases.grades_by_query))
    if result_set.unit_kind != cases.unit_kind:
        raise EvaluationError(
            "judgments.unit_kind and results.unit_kind must match "
            f"({cases.unit_kind!r} != {result_set.unit_kind!r})"
        )
    if result_set.chunker_provenance != cases.chunker_provenance:
        raise EvaluationError("judgments and results chunker_provenance must match")
    rankings = result_set.rankings_by_retriever

    summaries: dict[str, object] = {}
    for retriever_id in sorted(rankings):
        by_query = rankings[retriever_id]
        reciprocal_ranks: list[float] = []
        recall_sums = {k: 0.0 for k in selected_ks}
        ndcg_sums = {k: 0.0 for k in selected_ks}
        scored_queries = 0
        relevant_queries = 0
        for query_id in sorted(cases.grades_by_query):
            grades = cases.grades_by_query[query_id]
            relevant_grades = {unit_id: grade for unit_id, grade in grades.items() if grade > 0}
            has_relevance = bool(relevant_grades)
            if has_relevance:
                relevant_queries += 1
            elif empty_relevance_policy == "exclude":
                continue
            scored_queries += 1

            ranked = by_query.get(query_id, ())
            first_relevant_rank = next(
                (rank for rank, unit_id in ranked if grades.get(unit_id, 0) > 0),
                None,
            )
            reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

            for k in selected_ks:
                retrieved_relevant = {
                    unit_id
                    for rank, unit_id in ranked
                    if rank <= k and grades.get(unit_id, 0) > 0
                }
                recall = len(retrieved_relevant) / len(relevant_grades) if relevant_grades else 0.0
                recall_sums[k] += recall

                dcg = sum(
                    (2**grades.get(unit_id, 0) - 1) / math.log2(rank + 1)
                    for rank, unit_id in ranked
                    if rank <= k
                )
                ideal_grades = sorted(relevant_grades.values(), reverse=True)[:k]
                idcg = sum(
                    (2**grade - 1) / math.log2(rank + 1)
                    for rank, grade in enumerate(ideal_grades, start=1)
                )
                ndcg_sums[k] += dcg / idcg if idcg else 0.0

        summaries[retriever_id] = {
            "queries_total": len(cases.grades_by_query),
            "queries_with_relevance": relevant_queries,
            "queries_scored": scored_queries,
            "queries_excluded_for_empty_relevance": (
                len(cases.grades_by_query) - relevant_queries
                if empty_relevance_policy == "exclude"
                else 0
            ),
            "mean_reciprocal_rank": _average(reciprocal_ranks),
            "recall_at": {
                str(k): recall_sums[k] / scored_queries if scored_queries else None
                for k in selected_ks
            },
            "ndcg_at": {
                str(k): ndcg_sums[k] / scored_queries if scored_queries else None
                for k in selected_ks
            },
        }

    report: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "unit_kind": cases.unit_kind,
        "empty_relevance_policy": empty_relevance_policy,
        "ks": list(selected_ks),
        "metrics": summaries,
    }
    if cases.chunker_provenance is not None:
        name, version, configuration = cases.chunker_provenance
        report["chunker_provenance"] = {
            "name": name,
            "version": version,
            "configuration": configuration,
        }
    return report


def evaluate_files(
    judgments_path: str | Path,
    rankings_path: str | Path,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    empty_relevance_policy: EmptyRelevancePolicy = "exclude",
) -> dict[str, object]:
    """Load and evaluate versioned JSON input files."""
    return evaluate_payloads(
        _load_json(judgments_path, "judgments"),
        _load_json(rankings_path, "results"),
        ks=ks,
        empty_relevance_policy=empty_relevance_policy,
    )


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("k values must be comma-separated integers") from exc


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judgments", help="versioned graded judgments JSON")
    parser.add_argument("rankings", help="versioned per-retriever ranking results JSON")
    parser.add_argument("--ks", type=_parse_ks, default=DEFAULT_KS, help="comma-separated cutoffs")
    parser.add_argument(
        "--empty-relevance",
        choices=("exclude", "zero"),
        default="exclude",
        help="how to handle queries with no positive-grade judgments (default: exclude)",
    )
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    try:
        report = evaluate_files(
            args.judgments,
            args.rankings,
            ks=args.ks,
            empty_relevance_policy=args.empty_relevance,
        )
    except EvaluationError as exc:
        print(f"retrieval evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
