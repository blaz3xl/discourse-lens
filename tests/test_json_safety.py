from __future__ import annotations

import json

import pytest

from discourselens.json_safety import (
    MAX_JSON_NESTING_DEPTH,
    JsonNestingLimitError,
    validate_json_nesting,
)


def _nested_array(depth: int) -> str:
    return "[" * depth + "0" + "]" * depth


def test_json_nesting_limit_accepts_exact_limit() -> None:
    text = _nested_array(MAX_JSON_NESTING_DEPTH)

    validate_json_nesting(text)

    assert isinstance(json.loads(text), list)


def test_json_nesting_limit_rejects_one_more_container() -> None:
    with pytest.raises(JsonNestingLimitError, match="configured depth limit"):
        validate_json_nesting(_nested_array(MAX_JSON_NESTING_DEPTH + 1))


def test_delimiters_inside_escaped_json_strings_are_not_containers() -> None:
    content = 'brackets [{ ] plus quote " and backslash \\'
    text = json.dumps({"value": content})

    validate_json_nesting(text)

    assert json.loads(text)["value"] == content


@pytest.mark.parametrize("max_depth", [-1, True, 1.5])
def test_json_nesting_limit_rejects_invalid_limit(max_depth: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_json_nesting("{}", max_depth=max_depth)  # type: ignore[arg-type]
