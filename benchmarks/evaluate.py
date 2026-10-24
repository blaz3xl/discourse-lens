"""Strict, mechanical evaluation for JSONL evidence-benchmark files.

This evaluator measures exact offsets, revision-bound evidence IDs, and declared
labels only. It does not measure semantic quality or validate CDA interpretations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TextIO, cast

from discourselens.json_safety import (
    MAX_JSON_NESTING_DEPTH,
    JsonNestingLimitError,
    validate_json_nesting,
)

MAX_JSONL_LINE_BYTES = 1_048_576
MAX_JSON_NESTING = MAX_JSON_NESTING_DEPTH
SUPPORTED_LANGUAGES = ("en", "es", "ar")
CASE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
LIMITATION_TAG_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

COVERAGE_MODES = ("EXAMPLE_BASED", "TARGETED", "BROAD", "EXHAUSTIVE")
# Smoke-protocol comparison convention for declared labels, not a retrieval metric.
COVERAGE_LABEL_ORDER = ("EXAMPLE_BASED", "TARGETED", "BROAD", "EXHAUSTIVE")
COVERAGE_LABEL_RANK = {mode: rank for rank, mode in enumerate(COVERAGE_LABEL_ORDER)}
CLAIM_TYPES = (
    "OBSERVATIONAL",
    "EXAMPLE_BASED",
    "INTERPRETIVE",
    "COMPARATIVE",
    "QUANTIFIED",
    "ABSENCE",
    "UNIVERSAL",
    "CONTEXTUAL",
)

NOTICE = (
    "Mechanical metrics only: exact spans, revision-bound evidence IDs, and declared labels. "
    "Coverage-label ordering is a "
    "smoke-protocol convention and does not verify retrieval/search coverage; this evaluator "
    "does not measure semantic quality or validate CDA."
)

Span = tuple[int, int]


class EvaluationError(ValueError):
    """Raised when a JSONL file violates the benchmark contract."""


@dataclass(frozen=True)
class GoldCase:
    support_spans: frozenset[Span]
    counterevidence_spans: frozenset[Span]
    support_evidence_ids: frozenset[str]
    counterevidence_evidence_ids: frozenset[str]
    allowed_claim_types: frozenset[str]
    prohibited_claim_types: frozenset[str]
    minimum_coverage: str
    requires_alternative_interpretation: bool
    required_limitations: frozenset[str]


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    language: str
    text: str
    task: str
    scope_revision_ids: frozenset[str]
    evidence_revisions: tuple[tuple[str, str], ...]
    gold: GoldCase


@dataclass(frozen=True)
class FindingPrediction:
    claim_types: frozenset[str]
    support_spans: frozenset[Span]
    counterevidence_spans: frozenset[Span]
    support_evidence_ids: tuple[tuple[str, str], ...]
    counterevidence_evidence_ids: tuple[tuple[str, str], ...]
    alternative_interpretation: str | None
    limitation_tags: frozenset[str]


@dataclass(frozen=True)
class CasePrediction:
    coverage_mode: str
    findings: tuple[FindingPrediction, ...]


def _fail_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_jsonl(path: str | Path, label: str) -> list[dict[str, object]]:
    source_path = Path(path)
    records: list[dict[str, object]] = []
    try:
        stream = source_path.open("rb")
    except OSError as exc:
        raise EvaluationError(f"cannot read {label} file {source_path}: {exc}") from exc

    with stream:
        line_number = 0
        while True:
            raw_line = stream.readline(MAX_JSONL_LINE_BYTES + 1)
            if not raw_line:
                break
            line_number += 1
            location = f"{source_path}:{line_number}"
            if len(raw_line) > MAX_JSONL_LINE_BYTES:
                raise EvaluationError(f"{location}: line exceeds {MAX_JSONL_LINE_BYTES} byte limit")
            if raw_line.endswith(b"\n"):
                raw_line = raw_line[:-1]
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            if not raw_line:
                raise EvaluationError(f"{location}: blank JSONL lines are not allowed")
            try:
                text_line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EvaluationError(f"{location}: line is not valid UTF-8") from exc
            try:
                validate_json_nesting(text_line, max_depth=MAX_JSON_NESTING)
            except JsonNestingLimitError as exc:
                raise EvaluationError(
                    f"{location}: JSON nesting exceeds the {MAX_JSON_NESTING}-container limit"
                ) from exc
            try:
                value = json.loads(
                    text_line,
                    object_pairs_hook=_object_without_duplicate_keys,
                    parse_constant=_fail_constant,
                )
            except RecursionError as exc:
                raise EvaluationError(
                    f"{location}: JSON nesting exceeds the {MAX_JSON_NESTING}-container limit"
                ) from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise EvaluationError(f"{location}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
                raise EvaluationError(f"{location}: each JSONL record must be an object")
            records.append(value)
    return records


def _record(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluationError(f"{location}: expected an object")
    return value


def _exact_keys(value: dict[str, object], expected: set[str], location: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if extra:
            details.append("unknown keys: " + ", ".join(extra))
        raise EvaluationError(f"{location}: " + "; ".join(details))


def _has_surrogate(value: str) -> bool:
    return any("\ud800" <= character <= "\udfff" for character in value)


def _non_empty_string(value: object, name: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{location}: {name} must be a non-empty string")
    if _has_surrogate(value):
        raise EvaluationError(f"{location}: {name} must contain valid Unicode scalar values")
    return value


def _enum(value: object, allowed: Sequence[str], name: str, location: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(allowed)
        raise EvaluationError(f"{location}: {name} must be one of: {choices}")
    return value


def _string_set(
    value: object,
    *,
    name: str,
    location: str,
    allowed: Sequence[str] | None = None,
    minimum_items: int = 0,
    unique_items: bool = False,
    tag_pattern: re.Pattern[str] | None = None,
) -> frozenset[str]:
    if not isinstance(value, list):
        raise EvaluationError(f"{location}: {name} must be an array")
    if len(value) < minimum_items:
        raise EvaluationError(f"{location}: {name} must contain at least {minimum_items} item(s)")
    result: set[str] = set()
    for index, item in enumerate(value):
        item_location = f"{location}.{name}[{index}]"
        if allowed is not None:
            selected = _enum(item, allowed, name, item_location)
        else:
            selected = _non_empty_string(item, name, item_location)
            if tag_pattern is not None and not tag_pattern.fullmatch(selected):
                raise EvaluationError(
                    f"{item_location}: {name} entries must match the benchmark tag format"
                )
        if unique_items and selected in result:
            raise EvaluationError(f"{location}: {name} entries must be unique")
        result.add(selected)
    return frozenset(result)


def _spans(
    value: object,
    *,
    name: str,
    text: str,
    location: str,
    minimum_items: int = 0,
) -> frozenset[Span]:
    if not isinstance(value, list):
        raise EvaluationError(f"{location}: {name} must be an array")
    if len(value) < minimum_items:
        raise EvaluationError(f"{location}: {name} must contain at least {minimum_items} item(s)")
    result: set[Span] = set()
    for index, raw_span in enumerate(value):
        span_location = f"{location}.{name}[{index}]"
        span = _record(raw_span, span_location)
        _exact_keys(span, {"start", "end"}, span_location)
        start = span["start"]
        end = span["end"]
        if type(start) is not int or type(end) is not int:
            raise EvaluationError(f"{span_location}: start and end must be integers")
        if start < 0 or end <= start or end > len(text):
            raise EvaluationError(
                f"{span_location}: offsets must be non-empty half-open spans within text"
            )
        if (start, end) in result:
            raise EvaluationError(f"{location}: {name} entries must be unique")
        result.add((start, end))
    return frozenset(result)


def _evidence_catalog(value: object, location: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise EvaluationError(f"{location}.evidence_catalog must be an array")
    catalog: dict[str, str] = {}
    for index, raw_reference in enumerate(value):
        reference_location = f"{location}.evidence_catalog[{index}]"
        reference = _record(raw_reference, reference_location)
        _exact_keys(reference, {"evidence_id", "revision_id"}, reference_location)
        evidence_id = _non_empty_string(
            reference["evidence_id"], "evidence_id", reference_location
        )
        revision_id = _non_empty_string(
            reference["revision_id"], "revision_id", reference_location
        )
        if evidence_id in catalog:
            raise EvaluationError(f"{location}.evidence_catalog evidence_id entries must be unique")
        catalog[evidence_id] = revision_id
    return tuple(sorted(catalog.items()))


def _evidence_references(
    value: object, *, name: str, location: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise EvaluationError(f"{location}.{name} must be an array")
    references: list[tuple[str, str]] = []
    for index, raw_reference in enumerate(value):
        reference_location = f"{location}.{name}[{index}]"
        reference = _record(raw_reference, reference_location)
        _exact_keys(reference, {"evidence_id", "revision_id"}, reference_location)
        references.append(
            (
                _non_empty_string(reference["evidence_id"], "evidence_id", reference_location),
                _non_empty_string(reference["revision_id"], "revision_id", reference_location),
            )
        )
    return tuple(references)


def _gold_evidence_ids(
    value: object,
    *,
    name: str,
    location: str,
    evidence_revisions: dict[str, str],
    scope_revision_ids: frozenset[str],
    minimum_items: int = 0,
) -> frozenset[str]:
    evidence_ids = _string_set(
        value,
        name=name,
        location=location,
        minimum_items=minimum_items,
        unique_items=True,
    )
    for evidence_id in sorted(evidence_ids):
        if evidence_id not in evidence_revisions:
            raise EvaluationError(f"{location}.{name} references unknown evidence_id {evidence_id!r}")
        if evidence_revisions[evidence_id] not in scope_revision_ids:
            raise EvaluationError(
                f"{location}.{name} evidence_id {evidence_id!r} is outside the declared revision scope"
            )
    return evidence_ids


def _parse_gold(
    value: object,
    text: str,
    location: str,
    *,
    evidence_revisions: dict[str, str],
    scope_revision_ids: frozenset[str],
    has_evidence_ids: bool,
) -> GoldCase:
    gold = _record(value, f"{location}.gold")
    expected_gold_keys = {
        "support_spans",
        "counterevidence_spans",
        "allowed_claim_types",
        "prohibited_claim_types",
        "minimum_coverage",
        "requires_alternative_interpretation",
        "required_limitations",
    }
    if has_evidence_ids:
        expected_gold_keys.update({"support_evidence_ids", "counterevidence_evidence_ids"})
    _exact_keys(gold, expected_gold_keys, f"{location}.gold")
    allowed_claim_types = _string_set(
        gold["allowed_claim_types"],
        name="allowed_claim_types",
        location=location,
        allowed=CLAIM_TYPES,
        minimum_items=1,
        unique_items=True,
    )
    prohibited_claim_types = _string_set(
        gold["prohibited_claim_types"],
        name="prohibited_claim_types",
        location=location,
        allowed=CLAIM_TYPES,
        minimum_items=1,
        unique_items=True,
    )
    overlap = sorted(allowed_claim_types & prohibited_claim_types)
    if overlap:
        raise EvaluationError(
            f"{location}.gold: claim types cannot be both allowed and prohibited: "
            + ", ".join(overlap)
        )
    requires_alternative = gold["requires_alternative_interpretation"]
    if type(requires_alternative) is not bool:
        raise EvaluationError(
            f"{location}.gold.requires_alternative_interpretation must be a boolean"
        )
    return GoldCase(
        support_spans=_spans(
            gold["support_spans"],
            name="support_spans",
            text=text,
            location=f"{location}.gold",
            minimum_items=1,
        ),
        counterevidence_spans=_spans(
            gold["counterevidence_spans"],
            name="counterevidence_spans",
            text=text,
            location=f"{location}.gold",
        ),
        support_evidence_ids=(
            _gold_evidence_ids(
                gold["support_evidence_ids"],
                name="support_evidence_ids",
                location=f"{location}.gold",
                evidence_revisions=evidence_revisions,
                scope_revision_ids=scope_revision_ids,
                minimum_items=1,
            )
            if has_evidence_ids
            else frozenset()
        ),
        counterevidence_evidence_ids=(
            _gold_evidence_ids(
                gold["counterevidence_evidence_ids"],
                name="counterevidence_evidence_ids",
                location=f"{location}.gold",
                evidence_revisions=evidence_revisions,
                scope_revision_ids=scope_revision_ids,
            )
            if has_evidence_ids
            else frozenset()
        ),
        allowed_claim_types=allowed_claim_types,
        prohibited_claim_types=prohibited_claim_types,
        minimum_coverage=_enum(
            gold["minimum_coverage"], COVERAGE_MODES, "minimum_coverage", location
        ),
        requires_alternative_interpretation=requires_alternative,
        required_limitations=_string_set(
            gold["required_limitations"],
            name="required_limitations",
            location=f"{location}.gold",
            unique_items=True,
            tag_pattern=LIMITATION_TAG_PATTERN,
        ),
    )


def _parse_dataset(path: str | Path) -> list[DatasetCase]:
    records = _read_jsonl(path, "dataset")
    if not records:
        raise EvaluationError("dataset must contain at least one case")
    cases: list[DatasetCase] = []
    seen_case_ids: set[str] = set()
    expected = {
        "case_id",
        "language",
        "text",
        "task",
        "gold",
    }
    evidence_contract_keys = {"scope_revision_ids", "evidence_catalog"}
    for index, raw_record in enumerate(records, start=1):
        location = f"{Path(path)}:{index}"
        record = _record(raw_record, location)
        has_evidence_ids = evidence_contract_keys <= set(record)
        _exact_keys(record, expected | (evidence_contract_keys if has_evidence_ids else set()), location)
        case_id = _non_empty_string(record["case_id"], "case_id", location)
        if case_id in seen_case_ids:
            raise EvaluationError(f"{location}: duplicate case_id {case_id!r}")
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise EvaluationError(f"{location}: case_id must be a lowercase hyphenated identifier")
        seen_case_ids.add(case_id)
        language = _enum(record["language"], SUPPORTED_LANGUAGES, "language", location)
        text = record["text"]
        if not isinstance(text, str) or not text:
            raise EvaluationError(f"{location}: text must be a non-empty string")
        if _has_surrogate(text):
            raise EvaluationError(f"{location}: text must contain valid Unicode scalar values")
        task = _non_empty_string(record["task"], "task", location)
        if has_evidence_ids:
            scope_revision_ids = _string_set(
                record["scope_revision_ids"],
                name="scope_revision_ids",
                location=location,
                minimum_items=1,
                unique_items=True,
            )
            evidence_catalog = _evidence_catalog(record["evidence_catalog"], location)
        else:
            if evidence_contract_keys & set(record):
                _exact_keys(record, expected | evidence_contract_keys, location)
            scope_revision_ids = frozenset()
            evidence_catalog = ()
        evidence_revisions = dict(evidence_catalog)
        cases.append(
            DatasetCase(
                case_id=case_id,
                language=language,
                text=text,
                task=task,
                scope_revision_ids=scope_revision_ids,
                evidence_revisions=evidence_catalog,
                gold=_parse_gold(
                    record["gold"],
                    text,
                    location,
                    evidence_revisions=evidence_revisions,
                    scope_revision_ids=scope_revision_ids,
                    has_evidence_ids=has_evidence_ids,
                ),
            )
        )
    return cases


def _parse_finding(value: object, text: str, location: str) -> FindingPrediction:
    finding = _record(value, location)
    expected = {
        "claim_types",
        "support_spans",
        "counterevidence_spans",
        "alternative_interpretation",
        "limitation_tags",
    }
    evidence_fields = {"support_evidence_ids", "counterevidence_evidence_ids"}
    has_evidence_ids = evidence_fields <= set(finding)
    _exact_keys(finding, expected | (evidence_fields if has_evidence_ids else set()), location)
    alternative = finding["alternative_interpretation"]
    if alternative is not None and not isinstance(alternative, str):
        raise EvaluationError(f"{location}.alternative_interpretation must be a string or null")
    if isinstance(alternative, str) and _has_surrogate(alternative):
        raise EvaluationError(
            f"{location}.alternative_interpretation must contain valid Unicode scalar values"
        )
    return FindingPrediction(
        claim_types=_string_set(
            finding["claim_types"],
            name="claim_types",
            location=location,
            allowed=CLAIM_TYPES,
            unique_items=True,
        ),
        support_spans=_spans(
            finding["support_spans"], name="support_spans", text=text, location=location
        ),
        counterevidence_spans=_spans(
            finding["counterevidence_spans"],
            name="counterevidence_spans",
            text=text,
            location=location,
        ),
        support_evidence_ids=(
            _evidence_references(
                finding["support_evidence_ids"], name="support_evidence_ids", location=location
            )
            if has_evidence_ids
            else ()
        ),
        counterevidence_evidence_ids=(
            _evidence_references(
                finding["counterevidence_evidence_ids"],
                name="counterevidence_evidence_ids",
                location=location,
            )
            if has_evidence_ids
            else ()
        ),
        alternative_interpretation=alternative,
        limitation_tags=_string_set(
            finding["limitation_tags"],
            name="limitation_tags",
            location=location,
            unique_items=True,
            tag_pattern=LIMITATION_TAG_PATTERN,
        ),
    )


def _parse_predictions(path: str | Path, cases: Sequence[DatasetCase]) -> dict[str, CasePrediction]:
    records = _read_jsonl(path, "prediction")
    dataset_by_id = {case.case_id: case for case in cases}
    predictions: dict[str, CasePrediction] = {}
    expected = {"case_id", "coverage_mode", "findings"}
    for index, raw_record in enumerate(records, start=1):
        location = f"{Path(path)}:{index}"
        record = _record(raw_record, location)
        _exact_keys(record, expected, location)
        case_id = _non_empty_string(record["case_id"], "case_id", location)
        if case_id not in dataset_by_id:
            raise EvaluationError(f"{location}: unknown prediction case_id {case_id!r}")
        if case_id in predictions:
            raise EvaluationError(f"{location}: duplicate prediction case_id {case_id!r}")
        coverage_mode = _enum(record["coverage_mode"], COVERAGE_MODES, "coverage_mode", location)
        raw_findings = record["findings"]
        if not isinstance(raw_findings, list):
            raise EvaluationError(f"{location}.findings must be an array")
        case = dataset_by_id[case_id]
        findings = tuple(
            _parse_finding(item, case.text, f"{location}.findings[{finding_index}]")
            for finding_index, item in enumerate(raw_findings)
        )
        predictions[case_id] = CasePrediction(coverage_mode, findings)
    return predictions


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _span_score(gold: frozenset[Span], predicted: frozenset[Span]) -> dict[str, object]:
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    predicted_count = len(predicted)
    gold_count = len(gold)
    denominator = predicted_count + gold_count
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
        "precision": _ratio(true_positive, predicted_count),
        "recall": _ratio(true_positive, gold_count),
        "f1": (2 * true_positive / denominator) if denominator else None,
    }


def _evidence_score(
    gold: frozenset[str],
    predicted_references: Sequence[tuple[str, str]],
    evidence_revisions: tuple[tuple[str, str], ...],
    scope_revision_ids: frozenset[str],
) -> dict[str, object]:
    catalog = dict(evidence_revisions)
    predicted = frozenset(predicted_references)
    gold_references = frozenset((evidence_id, catalog[evidence_id]) for evidence_id in gold)
    true_positive = len(gold_references & predicted)
    false_positive = len(predicted - gold_references)
    false_negative = len(gold_references - predicted)
    predicted_count = len(predicted)
    gold_count = len(gold_references)
    denominator = predicted_count + gold_count

    occurrence_counts: dict[str, int] = {}
    for evidence_id, _ in predicted_references:
        occurrence_counts[evidence_id] = occurrence_counts.get(evidence_id, 0) + 1
    duplicate_ids = sorted(
        evidence_id for evidence_id, count in occurrence_counts.items() if count > 1
    )
    unknown_ids: set[str] = set()
    out_of_scope_ids: set[str] = set()
    revision_mismatches: set[tuple[str, str, str]] = set()
    for evidence_id, predicted_revision_id in predicted:
        expected_revision_id = catalog.get(evidence_id)
        if expected_revision_id is None:
            unknown_ids.add(evidence_id)
        elif expected_revision_id not in scope_revision_ids:
            out_of_scope_ids.add(evidence_id)
        elif expected_revision_id != predicted_revision_id:
            revision_mismatches.add(
                (evidence_id, expected_revision_id, predicted_revision_id)
            )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
        "precision": _ratio(true_positive, predicted_count),
        "recall": _ratio(true_positive, gold_count),
        "f1": (2 * true_positive / denominator) if denominator else None,
        "diagnostics": {
            "unknown_ids": sorted(unknown_ids),
            "duplicate_ids": duplicate_ids,
            "out_of_scope_ids": sorted(out_of_scope_ids),
            "revision_mismatched_ids": [
                {
                    "evidence_id": evidence_id,
                    "expected_revision_id": expected_revision_id,
                    "predicted_revision_id": predicted_revision_id,
                }
                for evidence_id, expected_revision_id, predicted_revision_id in sorted(
                    revision_mismatches
                )
            ],
        },
    }


def _sum_span_scores(scores: Sequence[dict[str, object]]) -> dict[str, object]:
    true_positive = sum(cast(int, score["true_positive"]) for score in scores)
    false_positive = sum(cast(int, score["false_positive"]) for score in scores)
    false_negative = sum(cast(int, score["false_negative"]) for score in scores)
    predicted_count = true_positive + false_positive
    gold_count = true_positive + false_negative
    denominator = predicted_count + gold_count
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
        "precision": _ratio(true_positive, predicted_count),
        "recall": _ratio(true_positive, gold_count),
        "f1": (2 * true_positive / denominator) if denominator else None,
    }


def _sum_evidence_scores(scores: Sequence[dict[str, object]]) -> dict[str, object]:
    true_positive = sum(cast(int, score["true_positive"]) for score in scores)
    false_positive = sum(cast(int, score["false_positive"]) for score in scores)
    false_negative = sum(cast(int, score["false_negative"]) for score in scores)
    predicted_count = true_positive + false_positive
    gold_count = true_positive + false_negative
    denominator = predicted_count + gold_count
    unknown_count = 0
    duplicate_count = 0
    out_of_scope_count = 0
    revision_mismatch_count = 0
    for score in scores:
        diagnostics = score["diagnostics"]
        assert isinstance(diagnostics, dict)
        unknown_count += len(cast(list[object], diagnostics["unknown_ids"]))
        duplicate_count += len(cast(list[object], diagnostics["duplicate_ids"]))
        out_of_scope_count += len(cast(list[object], diagnostics["out_of_scope_ids"]))
        revision_mismatch_count += len(
            cast(list[object], diagnostics["revision_mismatched_ids"])
        )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
        "precision": _ratio(true_positive, predicted_count),
        "recall": _ratio(true_positive, gold_count),
        "f1": (2 * true_positive / denominator) if denominator else None,
        "diagnostic_counts": {
            "unknown_ids": unknown_count,
            "duplicate_ids": duplicate_count,
            "out_of_scope_ids": out_of_scope_count,
            "revision_mismatched_ids": revision_mismatch_count,
        },
    }


def _evaluate_case(case: DatasetCase, prediction: CasePrediction | None) -> dict[str, object]:
    findings = prediction.findings if prediction is not None else ()
    predicted_support = frozenset(span for finding in findings for span in finding.support_spans)
    predicted_counterevidence = frozenset(
        span for finding in findings for span in finding.counterevidence_spans
    )
    predicted_support_evidence_ids = tuple(
        reference for finding in findings for reference in finding.support_evidence_ids
    )
    predicted_counterevidence_evidence_ids = tuple(
        reference for finding in findings for reference in finding.counterevidence_evidence_ids
    )
    predicted_claim_types = frozenset(
        claim_type for finding in findings for claim_type in finding.claim_types
    )
    predicted_limitations = frozenset(
        tag for finding in findings for tag in finding.limitation_tags
    )
    prohibited = predicted_claim_types & case.gold.prohibited_claim_types
    not_allowed = predicted_claim_types - case.gold.allowed_claim_types
    required_limitations = case.gold.required_limitations
    matched_limitations = required_limitations & predicted_limitations
    missing_limitations = required_limitations - predicted_limitations
    alternative_present = any(
        isinstance(finding.alternative_interpretation, str)
        and bool(finding.alternative_interpretation.strip())
        for finding in findings
    )
    predicted_coverage = prediction.coverage_mode if prediction is not None else None
    label_below_minimum = (
        COVERAGE_LABEL_RANK[predicted_coverage] < COVERAGE_LABEL_RANK[case.gold.minimum_coverage]
        if predicted_coverage is not None
        else None
    )
    coverage_declaration_missing = prediction is None
    limitation_recall = _ratio(len(matched_limitations), len(required_limitations))
    required_alternative = case.gold.requires_alternative_interpretation
    return {
        "case_id": case.case_id,
        "language": case.language,
        "task": case.task,
        "support_spans": _span_score(case.gold.support_spans, predicted_support),
        "counterevidence_spans": _span_score(
            case.gold.counterevidence_spans, predicted_counterevidence
        ),
        "support_evidence_ids": _evidence_score(
            case.gold.support_evidence_ids,
            predicted_support_evidence_ids,
            case.evidence_revisions,
            case.scope_revision_ids,
        ),
        "counterevidence_evidence_ids": _evidence_score(
            case.gold.counterevidence_evidence_ids,
            predicted_counterevidence_evidence_ids,
            case.evidence_revisions,
            case.scope_revision_ids,
        ),
        "claims": {
            "predicted": sorted(predicted_claim_types),
            "prohibited": sorted(prohibited),
            "not_allowed": sorted(not_allowed),
            "prohibited_count": len(prohibited),
            "not_allowed_count": len(not_allowed),
        },
        "coverage_label_comparison": {
            "minimum_label": case.gold.minimum_coverage,
            "prediction_label": predicted_coverage,
            "label_below_minimum": label_below_minimum,
            "declaration_missing": coverage_declaration_missing,
            "minimum_label_requirement_met": label_below_minimum is False,
        },
        "alternative_interpretation": {
            "required": required_alternative,
            "present": alternative_present,
            "requirement_met": (alternative_present if required_alternative else True),
        },
        "required_limitations": {
            "required": sorted(required_limitations),
            "matched": sorted(matched_limitations),
            "missing": sorted(missing_limitations),
            "recall": limitation_recall,
        },
    }


def _aggregate(case_results: Sequence[dict[str, object]]) -> dict[str, object]:
    support_scores = [
        result["support_spans"]
        for result in case_results
        if isinstance(result["support_spans"], dict)
    ]
    counterevidence_scores = [
        result["counterevidence_spans"]
        for result in case_results
        if isinstance(result["counterevidence_spans"], dict)
    ]
    support_evidence_scores = [
        result["support_evidence_ids"]
        for result in case_results
        if isinstance(result["support_evidence_ids"], dict)
    ]
    counterevidence_evidence_scores = [
        result["counterevidence_evidence_ids"]
        for result in case_results
        if isinstance(result["counterevidence_evidence_ids"], dict)
    ]
    prohibited_cases = 0
    prohibited_count = 0
    not_allowed_cases = 0
    not_allowed_count = 0
    below_minimum_label_cases = 0
    missing_coverage_declaration_cases = 0
    required_alternative_cases = 0
    missing_alternative_cases = 0
    present_alternative_cases = 0
    required_limitation_count = 0
    matched_limitation_count = 0
    missing_limitation_count = 0
    for result in case_results:
        claims = result["claims"]
        coverage = result["coverage_label_comparison"]
        alternative = result["alternative_interpretation"]
        limitations = result["required_limitations"]
        assert isinstance(claims, dict)
        assert isinstance(coverage, dict)
        assert isinstance(alternative, dict)
        assert isinstance(limitations, dict)
        case_prohibited = int(claims["prohibited_count"])
        case_not_allowed = int(claims["not_allowed_count"])
        prohibited_count += case_prohibited
        not_allowed_count += case_not_allowed
        prohibited_cases += case_prohibited > 0
        not_allowed_cases += case_not_allowed > 0
        below_minimum_label_cases += coverage["label_below_minimum"] is True
        missing_coverage_declaration_cases += coverage["declaration_missing"] is True
        present_alternative_cases += alternative["present"] is True
        if alternative["required"] is True:
            required_alternative_cases += 1
            missing_alternative_cases += alternative["present"] is not True
        required_limitation_count += len(cast(list[object], limitations["required"]))
        matched_limitation_count += len(cast(list[object], limitations["matched"]))
        missing_limitation_count += len(cast(list[object], limitations["missing"]))

    case_count = len(case_results)
    return {
        "case_count": case_count,
        "support_spans": _sum_span_scores(support_scores),
        "counterevidence_spans": _sum_span_scores(counterevidence_scores),
        "support_evidence_ids": _sum_evidence_scores(support_evidence_scores),
        "counterevidence_evidence_ids": _sum_evidence_scores(counterevidence_evidence_scores),
        "claims": {
            "cases_with_prohibited_claims": prohibited_cases,
            "prohibited_claim_type_count": prohibited_count,
            "cases_with_not_allowed_claims": not_allowed_cases,
            "not_allowed_claim_type_count": not_allowed_count,
        },
        "coverage_label_comparison": {
            "below_minimum_label_cases": below_minimum_label_cases,
            "missing_coverage_declaration_cases": missing_coverage_declaration_cases,
            "minimum_label_requirement_unmet_cases": (
                below_minimum_label_cases + missing_coverage_declaration_cases
            ),
            "case_count": case_count,
            "minimum_label_requirement_unmet_rate": _ratio(
                below_minimum_label_cases + missing_coverage_declaration_cases, case_count
            ),
        },
        "alternative_interpretation": {
            "required_cases": required_alternative_cases,
            "present_cases": present_alternative_cases,
            "missing_required_cases": missing_alternative_cases,
            "requirement_recall": _ratio(
                required_alternative_cases - missing_alternative_cases,
                required_alternative_cases,
            ),
        },
        "required_limitations": {
            "required_tag_count": required_limitation_count,
            "matched_tag_count": matched_limitation_count,
            "missing_tag_count": missing_limitation_count,
            "recall": _ratio(matched_limitation_count, required_limitation_count),
        },
    }


def evaluate(dataset_path: str | Path, predictions_path: str | Path) -> dict[str, object]:
    """Evaluate predictions against a strict dataset contract and return JSON data."""
    cases = _parse_dataset(dataset_path)
    predictions = _parse_predictions(predictions_path, cases)
    case_results = [_evaluate_case(case, predictions.get(case.case_id)) for case in cases]
    return {
        "notice": NOTICE,
        "aggregation": _aggregate(case_results),
        "cases": case_results,
    }


def _render_report(report: dict[str, object], stream: TextIO) -> None:
    json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
    stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mechanically evaluate exact spans, revision-bound evidence IDs, claim labels, coverage, "
            "alternative interpretations, and limitation tags."
        )
    )
    parser.add_argument("dataset", type=Path, help="gold dataset JSONL")
    parser.add_argument("predictions", type=Path, help="prediction JSONL")
    arguments = parser.parse_args(argv)
    try:
        report = evaluate(arguments.dataset, arguments.predictions)
    except EvaluationError as exc:
        print(f"benchmark evaluation error: {exc}", file=sys.stderr)
        return 2
    _render_report(report, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
