import csv
import io
from decimal import Decimal
from itertools import permutations

import pytest

from discourselens.errors import ProjectError
from discourselens.output import render_rows


def test_machine_and_human_output_rendering_is_deterministic():
    rows = [{"token": "café", "count": 2}]
    assert render_rows(rows, "jsonl") == '{"token": "café", "count": 2}\n'
    assert render_rows(rows, "csv") == "token,count\r\ncafé,2\r\n"
    assert "café" in render_rows(rows, "table")


def test_csv_neutralizes_formula_like_corpus_values_and_headers():
    rendered = render_rows(
        [
            {
                "=column": '=HYPERLINK("https://example.invalid","open")',
                "plus": " +SUM(1,2)",
                "minus": "-1+2",
                "at": "@SUM(A1:A2)",
                "tab": "\t=1+1",
                "newline": "\n=1+1",
                "ordinary": "café",
            }
        ],
        "csv",
        spreadsheet_safe_csv=True,
    )

    assert "'=column" in rendered
    assert "'=HYPERLINK" in rendered
    assert "' +SUM(1,2)" in rendered
    assert "'-1+2" in rendered
    assert "'@SUM(A1:A2)" in rendered
    assert "'\t=1+1" in rendered
    assert "'\n=1+1" in rendered
    assert "café" in rendered


def test_csv_preserves_formula_like_values_and_headers_for_machine_consumers():
    rendered = render_rows(
        [{"=column": "=1+1", "formula": '=HYPERLINK("https://example.invalid","open")'}],
        "csv",
    )
    reader = csv.DictReader(io.StringIO(rendered))
    row = next(reader)

    assert reader.fieldnames == ["=column", "formula"]
    assert row == {"=column": "=1+1", "formula": '=HYPERLINK("https://example.invalid","open")'}


def test_csv_neutralizes_formula_like_stringification_of_non_string_cells():
    class FormulaLike:
        def __str__(self) -> str:
            return '=HYPERLINK("https://example.invalid","open")'

    rendered = render_rows([{"term": FormulaLike()}], "csv", spreadsheet_safe_csv=True)

    assert "'=HYPERLINK" in rendered


def test_csv_preserves_exact_text_and_negative_numeric_cells_by_default():
    rendered = render_rows(
        [{"integer": -2, "float": -1.5, "decimal": Decimal("-2.50"), "text": "-2"}],
        "csv",
    )
    row = next(csv.DictReader(io.StringIO(rendered)))

    assert row == {
        "integer": "-2",
        "float": "-1.5",
        "decimal": "-2.50",
        "text": "-2",
    }


def test_spreadsheet_safe_csv_escapes_text_without_changing_numeric_values():
    rendered = render_rows(
        [{"integer": -2, "text": "-2", "formula": "=1+1"}],
        "csv",
        spreadsheet_safe_csv=True,
    )
    row = next(csv.DictReader(io.StringIO(rendered)))

    assert row == {"integer": "-2", "text": "'-2", "formula": "'=1+1"}


@pytest.mark.parametrize("prefix", ["\v", "\f", "\u00a0", "\u2003", "\ufeff\v"])
def test_csv_neutralizes_formula_after_all_leading_unicode_whitespace(prefix):
    rendered = render_rows(
        [{"term": f'{prefix}=HYPERLINK("https://example.invalid")'}],
        "csv",
        spreadsheet_safe_csv=True,
    )

    assert "'" + prefix + "=HYPERLINK" in rendered


@pytest.mark.parametrize("prefix", ["\x00", "\x1b", "\u200b", "\ufeff"])
def test_csv_neutralizes_formula_after_leading_control_or_format_characters(prefix):
    rendered = render_rows([{"term": f"{prefix}=1+1"}], "csv", spreadsheet_safe_csv=True)

    assert "'" + prefix + "=1+1" in rendered


def test_json_outputs_preserve_formula_like_values_unchanged():
    rows = [{"term": "=SUM(1,2)"}]

    assert render_rows(rows, "json") == '[{"term": "=SUM(1,2)"}]'
    assert render_rows(rows, "jsonl") == '{"term": "=SUM(1,2)"}\n'


@pytest.mark.parametrize("output_format", ["json", "jsonl"])
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_json_outputs_reject_non_finite_numbers(output_format, non_finite):
    rows = [{"metrics": {"value": non_finite}}]

    with pytest.raises(ProjectError, match="finite numeric values"):
        render_rows(rows, output_format)


def test_tabular_formats_keep_columns_that_first_appear_on_later_rows():
    rows = [{"first": "one"}, {"later": "two", "first": "three"}]

    csv_output = render_rows(rows, "csv")
    table_output = render_rows(rows, "table")

    assert csv_output == "first,later\r\none,\r\nthree,two\r\n"
    assert "first  later" in table_output
    assert "one    " in table_output
    assert "three  two" in table_output


def test_csv_and_table_sort_set_values_for_deterministic_output():
    class HashOrderedValue:
        def __init__(self, text: str, hash_value: int) -> None:
            self.text = text
            self.hash_value = hash_value

        def __hash__(self) -> int:
            return self.hash_value

        def __repr__(self) -> str:
            return repr(self.text)

    labels = ("alpha", "beta", "gamma")
    members = next(
        (
            candidate
            for ordered_labels in permutations(labels)
            if (
                candidate := {
                    HashOrderedValue(label, index)
                    for index, label in enumerate(ordered_labels)
                }
            )
            and [member.text for member in candidate] != sorted(labels)
        ),
        None,
    )
    assert members is not None
    nested = {HashOrderedValue("z", 0), HashOrderedValue("y", 1)}
    rows = [{"members": members, "nested": ["x", nested]}]

    csv_output = render_rows(rows, "csv")
    table_output = render_rows(rows, "table")

    assert csv_output == (
        'members,nested\r\n"{\'alpha\', \'beta\', \'gamma\'}",'
        '"[\'x\', {\'y\', \'z\'}]"\r\n'
    )
    assert "{'alpha', 'beta', 'gamma'}" in table_output
    assert "['x', {'y', 'z'}]" in table_output


def test_csv_pages_are_self_contained_and_repeat_the_header():
    page = [{"token": "café", "count": 1}]
    first_render = render_rows(page, "csv")
    repeated_render = render_rows(page, "csv")

    assert first_render == repeated_render == "token,count\r\ncafé,1\r\n"


def test_table_escapes_terminal_controls():
    first_page = render_rows([{"name": "entry", "value": "safe\x1b[2J\nnext"}], "table")

    assert "\x1b" not in first_page
    assert "\\x1b[2J\\x0anext" in first_page
    assert first_page.startswith("name   value")


@pytest.mark.parametrize("output_format", ["csv", "table"])
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_tabular_outputs_reject_non_finite_numbers(output_format, non_finite):
    with pytest.raises(ProjectError, match="finite numeric values"):
        render_rows([{"metrics": {"value": non_finite}}], output_format)


def test_json_rejects_non_string_nested_mapping_keys_instead_of_coercing_them():
    with pytest.raises(ProjectError, match="string mapping keys"):
        render_rows([{"metrics": {1: "value"}}], "json")


def test_outputs_reject_signaling_decimal_nan_as_a_project_error():
    with pytest.raises(ProjectError, match="finite numeric values"):
        render_rows([{"metrics": {"value": Decimal("sNaN")}}], "json")


@pytest.mark.parametrize("output_format", ["json", "jsonl", "csv", "table"])
def test_outputs_reject_circular_values_as_a_project_error(output_format):
    cycle = []
    cycle.append(cycle)

    with pytest.raises(ProjectError, match="circular references"):
        render_rows([{"cycle": cycle}], output_format)


def test_json_accepts_repeated_references_that_are_not_circular():
    shared = ["café"]

    assert render_rows([{"left": shared, "right": shared}], "json") == (
        '[{"left": ["café"], "right": ["café"]}]'
    )


def test_jsonl_pages_are_individually_newline_terminated_and_concatenate():
    first_page = render_rows([{"token": "café"}], "jsonl")
    next_page = render_rows([{"token": "tea"}], "jsonl")

    assert first_page + next_page == '{"token": "café"}\n{"token": "tea"}\n'
