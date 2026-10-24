import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.evaluate import (
    MAX_JSON_NESTING,
    MAX_JSONL_LINE_BYTES,
    EvaluationError,
    _read_jsonl,
    evaluate,
)


def _benchmark_subprocess_env(module_root: Path) -> dict[str, str]:
    """Expose the source tree to the child Python process without relying on a shell cwd."""
    env = os.environ.copy()
    source_root = str(module_root / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else os.pathsep.join((source_root, existing_pythonpath))
    )
    return env


def _gold(
    *,
    support_spans=None,
    counterevidence_spans=None,
    support_evidence_ids=None,
    counterevidence_evidence_ids=None,
    allowed_claim_types=None,
    prohibited_claim_types=None,
    minimum_coverage="EXAMPLE_BASED",
    requires_alternative_interpretation=False,
    required_limitations=None,
):
    result = {
        "support_spans": (support_spans if support_spans is not None else [{"start": 0, "end": 1}]),
        "counterevidence_spans": (
            counterevidence_spans if counterevidence_spans is not None else []
        ),
        "allowed_claim_types": (
            allowed_claim_types if allowed_claim_types is not None else ["OBSERVATIONAL"]
        ),
        "prohibited_claim_types": (
            prohibited_claim_types if prohibited_claim_types is not None else ["UNIVERSAL"]
        ),
        "minimum_coverage": minimum_coverage,
        "requires_alternative_interpretation": requires_alternative_interpretation,
        "required_limitations": required_limitations if required_limitations is not None else [],
    }
    if support_evidence_ids is not None or counterevidence_evidence_ids is not None:
        result["support_evidence_ids"] = support_evidence_ids or []
        result["counterevidence_evidence_ids"] = counterevidence_evidence_ids or []
    return result


def _case(case_id="case-1", *, text="abcdefghij", gold=None):
    return {
        "case_id": case_id,
        "language": "en",
        "text": text,
        "task": "bounded_claim_review",
        "gold": gold if gold is not None else _gold(),
    }


def _finding(
    *,
    claim_types=None,
    support_spans=None,
    counterevidence_spans=None,
    alternative_interpretation=None,
    limitation_tags=None,
):
    return {
        "claim_types": claim_types if claim_types is not None else [],
        "support_spans": support_spans if support_spans is not None else [],
        "counterevidence_spans": (
            counterevidence_spans if counterevidence_spans is not None else []
        ),
        "alternative_interpretation": alternative_interpretation,
        "limitation_tags": limitation_tags if limitation_tags is not None else [],
    }


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_evaluate_reports_mechanical_case_and_micro_aggregate_metrics(tmp_path):
    dataset = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            _case(
                "case-1",
                gold=_gold(
                    support_spans=[{"start": 0, "end": 2}, {"start": 2, "end": 4}],
                    counterevidence_spans=[{"start": 4, "end": 5}],
                    allowed_claim_types=["OBSERVATIONAL"],
                    prohibited_claim_types=["UNIVERSAL"],
                    minimum_coverage="TARGETED",
                    requires_alternative_interpretation=True,
                    required_limitations=["actor_not_identified", "not_population_prevalence"],
                ),
            ),
            _case(
                "case-2",
                gold=_gold(
                    support_spans=[{"start": 5, "end": 8}],
                    allowed_claim_types=["OBSERVATIONAL"],
                    prohibited_claim_types=["ABSENCE"],
                    minimum_coverage="EXHAUSTIVE",
                ),
            ),
        ],
    )
    predictions = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {
                "case_id": "case-1",
                "coverage_mode": "EXAMPLE_BASED",
                "findings": [
                    _finding(
                        claim_types=["OBSERVATIONAL", "UNIVERSAL"],
                        support_spans=[{"start": 0, "end": 2}, {"start": 4, "end": 6}],
                        alternative_interpretation="A bounded alternative reading.",
                        limitation_tags=["actor_not_identified"],
                    ),
                    _finding(support_spans=[{"start": 0, "end": 2}]),
                ],
            }
        ],
    )

    report = cast(dict[str, Any], evaluate(dataset, predictions))
    case_one, case_two = report["cases"]
    assert "does not measure semantic quality or validate CDA" in report["notice"]
    assert "does not verify retrieval/search coverage" in report["notice"]
    assert case_one["support_spans"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "predicted_count": 2,
        "gold_count": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert case_one["counterevidence_spans"]["recall"] == 0.0
    assert case_one["claims"]["prohibited"] == ["UNIVERSAL"]
    assert case_one["claims"]["not_allowed"] == ["UNIVERSAL"]
    assert case_one["coverage_label_comparison"]["label_below_minimum"] is True
    assert case_one["alternative_interpretation"]["requirement_met"] is True
    assert case_one["required_limitations"]["recall"] == 0.5
    assert case_two["support_spans"]["precision"] is None
    assert case_two["support_spans"]["recall"] == 0.0
    assert case_two["coverage_label_comparison"]["prediction_label"] is None
    assert case_two["coverage_label_comparison"]["label_below_minimum"] is None
    assert case_two["coverage_label_comparison"]["declaration_missing"] is True
    assert case_two["alternative_interpretation"]["requirement_met"] is True
    assert case_two["required_limitations"]["recall"] is None

    aggregate = report["aggregation"]
    assert aggregate["support_spans"]["true_positive"] == 1
    assert aggregate["support_spans"]["false_positive"] == 1
    assert aggregate["support_spans"]["false_negative"] == 2
    assert aggregate["support_spans"]["precision"] == 0.5
    assert aggregate["support_spans"]["recall"] == pytest.approx(1 / 3)
    assert aggregate["support_spans"]["f1"] == 0.4
    assert aggregate["claims"]["cases_with_prohibited_claims"] == 1
    assert aggregate["claims"]["not_allowed_claim_type_count"] == 1
    assert aggregate["coverage_label_comparison"]["below_minimum_label_cases"] == 1
    assert aggregate["coverage_label_comparison"]["missing_coverage_declaration_cases"] == 1
    assert aggregate["coverage_label_comparison"]["minimum_label_requirement_unmet_cases"] == 2
    assert aggregate["coverage_label_comparison"]["minimum_label_requirement_unmet_rate"] == 1.0
    assert aggregate["alternative_interpretation"]["requirement_recall"] == 1.0
    assert aggregate["required_limitations"]["recall"] == 0.5


@pytest.mark.parametrize(
    ("dataset_records", "prediction_records", "message"),
    [
        ([_case(), _case()], [], "duplicate case_id"),
        (
            [_case()],
            [{"case_id": "unknown", "coverage_mode": "BROAD", "findings": []}],
            "unknown prediction",
        ),
        (
            [_case()],
            [
                {"case_id": "case-1", "coverage_mode": "BROAD", "findings": []},
                {"case_id": "case-1", "coverage_mode": "BROAD", "findings": []},
            ],
            "duplicate prediction",
        ),
    ],
)
def test_parser_rejects_duplicate_or_unknown_case_ids(
    tmp_path, dataset_records, prediction_records, message
):
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", dataset_records)
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", prediction_records)
    with pytest.raises(EvaluationError, match=message):
        evaluate(dataset, predictions)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record["gold"].update(minimum_coverage="SAMPLED"), "minimum_coverage"),
        (
            lambda record: record["gold"].update(support_spans=[{"start": 2, "end": 2}]),
            "half-open spans",
        ),
        (
            lambda record: record["gold"].update(support_spans=[{"start": 0, "end": 11}]),
            "half-open spans",
        ),
        (
            lambda record: record["gold"].update(
                support_spans=[{"start": 0, "end": 1}, {"start": 0, "end": 1}]
            ),
            "entries must be unique",
        ),
        (
            lambda record: record["gold"].update(allowed_claim_types=["MADE_UP"]),
            "allowed_claim_types",
        ),
        (
            lambda record: record.update(unexpected="field"),
            "unknown keys",
        ),
    ],
)
def test_dataset_parser_rejects_invalid_contract_values(tmp_path, mutate, message):
    record = _case()
    mutate(record)
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [record])
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])
    with pytest.raises(EvaluationError, match=message):
        evaluate(dataset, predictions)


def test_prediction_parser_rejects_invalid_offsets_and_duplicate_json_keys(tmp_path):
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [_case()])
    predictions = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {
                "case_id": "case-1",
                "coverage_mode": "BROAD",
                "findings": [_finding(support_spans=[{"start": -1, "end": 2}])],
            }
        ],
    )
    with pytest.raises(EvaluationError, match="half-open spans"):
        evaluate(dataset, predictions)

    predictions.write_text(
        '{"case_id":"case-1","case_id":"case-1","coverage_mode":"BROAD","findings":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="duplicate JSON object key"):
        evaluate(dataset, predictions)


def test_prediction_parser_rejects_duplicate_span_and_claim_entries(tmp_path):
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [_case()])
    predictions = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {
                "case_id": "case-1",
                "coverage_mode": "BROAD",
                "findings": [
                    _finding(
                        claim_types=["OBSERVATIONAL", "OBSERVATIONAL"],
                        support_spans=[{"start": 0, "end": 1}, {"start": 0, "end": 1}],
                    )
                ],
            }
        ],
    )

    with pytest.raises(EvaluationError, match="entries must be unique"):
        evaluate(dataset, predictions)

    predictions = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {
                "case_id": "case-1",
                "coverage_mode": "BROAD",
                "findings": [_finding(claim_types=["OBSERVATIONAL", "OBSERVATIONAL"])],
            }
        ],
    )
    with pytest.raises(EvaluationError, match="entries must be unique"):
        evaluate(dataset, predictions)


def test_parser_rejects_escaped_unpaired_surrogate_strings(tmp_path):
    record = _case()
    record["task"] = "invalid unicode \ud800"
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])
    with pytest.raises(EvaluationError, match="valid Unicode scalar values"):
        evaluate(dataset, predictions)


def test_evidence_id_metrics_are_revision_bound_and_independent_of_span_scores(tmp_path):
    dataset_case = _case(
        gold=_gold(
            support_evidence_ids=["support-gold"],
            counterevidence_evidence_ids=["counter-gold"],
        )
    )
    dataset_case.update(
        scope_revision_ids=["revision-1"],
        evidence_catalog=[
            {"evidence_id": "support-gold", "revision_id": "revision-1"},
            {"evidence_id": "counter-gold", "revision_id": "revision-1"},
            {"evidence_id": "known-decoy", "revision_id": "revision-1"},
            {"evidence_id": "outside-scope", "revision_id": "revision-2"},
        ],
    )
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [dataset_case])
    finding = _finding(support_spans=[{"start": 1, "end": 2}])
    finding["support_evidence_ids"] = [
        {"evidence_id": "support-gold", "revision_id": "revision-1"},
        {"evidence_id": "support-gold", "revision_id": "revision-1"},
        {"evidence_id": "unknown-id", "revision_id": "revision-1"},
        {"evidence_id": "known-decoy", "revision_id": "revision-1"},
        {"evidence_id": "outside-scope", "revision_id": "revision-2"},
        {"evidence_id": "support-gold", "revision_id": "revision-2"},
    ]
    finding["counterevidence_evidence_ids"] = [
        {"evidence_id": "counter-gold", "revision_id": "revision-1"}
    ]
    predictions = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [{"case_id": "case-1", "coverage_mode": "BROAD", "findings": [finding]}],
    )

    report = cast(dict[str, Any], evaluate(dataset, predictions))
    result = report["cases"][0]
    assert result["support_spans"]["recall"] == 0.0
    assert result["support_evidence_ids"]["true_positive"] == 1
    assert result["support_evidence_ids"]["false_positive"] == 4
    assert result["support_evidence_ids"]["precision"] == 0.2
    assert result["support_evidence_ids"]["recall"] == 1.0
    assert result["support_evidence_ids"]["diagnostics"] == {
        "unknown_ids": ["unknown-id"],
        "duplicate_ids": ["support-gold"],
        "out_of_scope_ids": ["outside-scope"],
        "revision_mismatched_ids": [
            {
                "evidence_id": "support-gold",
                "expected_revision_id": "revision-1",
                "predicted_revision_id": "revision-2",
            }
        ],
    }
    assert result["counterevidence_evidence_ids"]["true_positive"] == 1
    assert result["counterevidence_evidence_ids"]["recall"] == 1.0
    assert report["aggregation"]["support_evidence_ids"]["diagnostic_counts"] == {
        "unknown_ids": 1,
        "duplicate_ids": 1,
        "out_of_scope_ids": 1,
        "revision_mismatched_ids": 1,
    }


def test_evidence_contract_rejects_unresolved_gold_and_duplicate_catalog_ids(tmp_path):
    dataset_case = _case()
    dataset_case.update(
        scope_revision_ids=["revision-1"],
        evidence_catalog=[
            {"evidence_id": "known", "revision_id": "revision-1"},
            {"evidence_id": "known", "revision_id": "revision-1"},
        ],
    )
    dataset_case["gold"]["support_evidence_ids"] = ["known"]
    dataset_case["gold"]["counterevidence_evidence_ids"] = []
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [dataset_case])
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])
    with pytest.raises(EvaluationError, match="catalog evidence_id entries must be unique"):
        evaluate(dataset, predictions)

    dataset_case["evidence_catalog"] = []
    dataset_case["gold"]["support_evidence_ids"] = ["unknown"]
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [dataset_case])
    with pytest.raises(EvaluationError, match="references unknown evidence_id"):
        evaluate(dataset, predictions)


def test_legacy_case_and_prediction_contracts_remain_supported(tmp_path):
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [_case()])
    predictions = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [{"case_id": "case-1", "coverage_mode": "BROAD", "findings": [_finding()]}],
    )

    report = cast(dict[str, Any], evaluate(dataset, predictions))
    assert report["cases"][0]["support_evidence_ids"]["gold_count"] == 0
    assert report["cases"][0]["support_evidence_ids"]["precision"] is None
    assert report["aggregation"]["support_evidence_ids"]["recall"] is None


def test_parser_rejects_oversized_jsonl_lines(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_bytes(b"{" + b" " * MAX_JSONL_LINE_BYTES + b"}\n")
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])

    with pytest.raises(EvaluationError, match="byte limit"):
        evaluate(dataset, predictions)


def test_json_nesting_limit_counts_containers_but_ignores_quoted_escapes(tmp_path):
    quoted_value = r'brackets [] {} and escaped quote: \" stay inside this string'
    nested_depth = MAX_JSON_NESTING - 1  # The root object is also a container.
    line = (
        '{"quoted":'
        + json.dumps(quoted_value)
        + ',"nested":'
        + "[" * nested_depth
        + "0"
        + "]" * nested_depth
        + "}\n"
    )
    path = tmp_path / "at-limit.jsonl"
    path.write_text(line, encoding="utf-8")

    records = _read_jsonl(path, "dataset")

    assert records[0]["quoted"] == quoted_value
    assert len(line.encode("utf-8")) <= MAX_JSONL_LINE_BYTES


def test_json_nesting_above_limit_is_rejected_before_json_parser(tmp_path):
    nested_depth = MAX_JSON_NESTING  # Plus the root object exceeds the limit.
    path = tmp_path / "over-limit.jsonl"
    path.write_text(
        '{"nested":' + "[" * nested_depth + "0" + "]" * nested_depth + "}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationError,
        match=f"JSON nesting exceeds the {MAX_JSON_NESTING}-container limit",
    ):
        _read_jsonl(path, "dataset")


def test_deeply_nested_json_fails_through_cli_without_traceback(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    nested_depth = 10_000
    nested = (
        b'{"case_id":"case-1","nested":' + b"[" * nested_depth + b"0" + b"]" * nested_depth + b"}\n"
    )
    assert len(nested) < MAX_JSONL_LINE_BYTES
    dataset.write_bytes(nested)
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])
    module_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.evaluate",
            str(dataset),
            str(predictions),
        ],
        cwd=module_root,
        env=_benchmark_subprocess_env(module_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "benchmark evaluation error" in result.stderr
    assert f"JSON nesting exceeds the {MAX_JSON_NESTING}-container limit" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.update(case_id="Case_1"), "lowercase hyphenated"),
        (lambda record: record.update(language="fr"), "language must be one of"),
        (lambda record: record.update(text=""), "text must be a non-empty"),
        (lambda record: record.update(task="   "), "task must be a non-empty"),
        (
            lambda record: record["gold"].update(prohibited_claim_types=["OBSERVATIONAL"]),
            "cannot be both allowed and prohibited",
        ),
        (
            lambda record: record["gold"].update(allowed_claim_types=[]),
            "at least 1 item",
        ),
        (
            lambda record: record["gold"].update(support_spans=[]),
            "at least 1 item",
        ),
        (
            lambda record: record["gold"].update(required_limitations=["Needs prose"]),
            "benchmark tag format",
        ),
    ],
)
def test_dataset_parser_enforces_case_schema_constraints(tmp_path, mutate, message):
    record = _case()
    mutate(record)
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [record])
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])
    with pytest.raises(EvaluationError, match=message):
        evaluate(dataset, predictions)


def test_dataset_parser_rejects_empty_dataset(tmp_path):
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [])
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])
    with pytest.raises(EvaluationError, match="at least one case"):
        evaluate(dataset, predictions)


def test_cli_output_is_deterministic_json_and_advertises_limits(tmp_path):
    dataset = _write_jsonl(tmp_path / "dataset.jsonl", [_case()])
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])
    module_root = Path(__file__).resolve().parents[1]

    first_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.evaluate",
            str(dataset),
            str(predictions),
        ],
        cwd=module_root,
        env=_benchmark_subprocess_env(module_root),
        capture_output=True,
        text=True,
        check=True,
    )
    second_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.evaluate",
            str(dataset),
            str(predictions),
        ],
        cwd=module_root,
        env=_benchmark_subprocess_env(module_root),
        capture_output=True,
        text=True,
        check=True,
    )

    assert first_result.stdout == second_result.stdout
    output = json.loads(first_result.stdout)
    assert output["aggregation"]["case_count"] == 1
    assert "does not measure semantic quality" in output["notice"]
    assert "does not verify retrieval/search coverage" in output["notice"]


def test_repository_smoke_dataset_matches_manifest_and_parses_unicode_spans(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    dataset_root = repository_root / "benchmarks/datasets/epistemic-contract-smoke"
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    predictions = _write_jsonl(tmp_path / "predictions.jsonl", [])

    report = cast(dict[str, Any], evaluate(dataset_root / manifest["case_file"], predictions))

    assert report["aggregation"]["case_count"] == manifest["case_count"] == 6
    assert {case["language"] for case in report["cases"]} == set(manifest["languages"])
    assert "does not measure semantic quality" in report["notice"]

    results_by_id = {case["case_id"]: case for case in report["cases"]}
    for case_id in ("en-02-agency-overclaim", "es-02-agency-overclaim", "ar-02-agency-overclaim"):
        result = results_by_id[case_id]
        assert result["counterevidence_spans"]["gold_count"] == 0
        assert result["counterevidence_spans"]["recall"] is None
        assert "motive_not_reported" in result["required_limitations"]["required"]


def test_case_schema_matches_runtime_cross_field_contract():
    repository_root = Path(__file__).resolve().parents[1]
    schema_path = repository_root / "benchmarks/schemas/case-v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    dataset = repository_root / "benchmarks/datasets/epistemic-contract-smoke/cases.jsonl"
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]

    assert all(validator.is_valid(case) for case in cases)
    invalid_task = _case()
    invalid_task["task"] = "   "
    assert not validator.is_valid(invalid_task)
    duplicate_support_span = _case()
    duplicate_support_span["gold"]["support_spans"] = [
        {"start": 0, "end": 1},
        {"start": 0, "end": 1},
    ]
    assert not validator.is_valid(duplicate_support_span)
    conflicting_claim_policy = _case()
    conflicting_claim_policy["gold"]["prohibited_claim_types"] = ["OBSERVATIONAL"]
    assert not validator.is_valid(conflicting_claim_policy)
    partial_evidence_scope = _case()
    partial_evidence_scope["scope_revision_ids"] = ["revision-1"]
    assert not validator.is_valid(partial_evidence_scope)
    unresolved_evidence_contract = _case()
    unresolved_evidence_contract["gold"]["support_evidence_ids"] = ["support-1"]
    unresolved_evidence_contract["gold"]["counterevidence_evidence_ids"] = []
    assert not validator.is_valid(unresolved_evidence_contract)
