"""Small deterministic renderers for CLI machine/human output."""

from __future__ import annotations

import csv
import io
import json
import math
import unicodedata
from collections.abc import Iterable, Mapping
from decimal import Decimal
from numbers import Real

from .errors import ProjectError


def _contains_set(value: object, seen: set[int] | None = None) -> bool:
    """Check whether a tabular value needs set-order normalization."""
    if isinstance(value, (set, frozenset)):
        return True
    if isinstance(value, (Mapping, list, tuple)):
        visited = seen if seen is not None else set()
        identity = id(value)
        if identity in visited:
            return False
        visited.add(identity)
        nested_values = (
            [item for pair in value.items() for item in pair]
            if isinstance(value, Mapping)
            else value
        )
        return any(_contains_set(item, visited) for item in nested_values)
    return False


def _stable_repr(value: object) -> str:
    """Preserve container reprs while sorting unordered set members."""
    if isinstance(value, set):
        items = sorted(_stable_repr(item) for item in value)
        return "{" + ", ".join(items) + "}" if items else "set()"
    if isinstance(value, frozenset):
        items = sorted(_stable_repr(item) for item in value)
        return f"frozenset({{{', '.join(items)}}})" if items else "frozenset()"
    if isinstance(value, Mapping):
        items = (f"{_stable_repr(key)}: {_stable_repr(item)}" for key, item in value.items())
        return "{" + ", ".join(items) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_stable_repr(item) for item in value) + "]"
    if isinstance(value, tuple):
        items = ", ".join(_stable_repr(item) for item in value)
        if len(value) == 1:
            items += ","
        return "(" + items + ")"
    return repr(value)


def _spreadsheet_safe_cell(value: object) -> object:
    """Keep untrusted CSV text from being interpreted as a spreadsheet formula.

    JSON/JSONL remain byte-for-byte data representations; this escaping is only
    applied to CSV cells, which spreadsheet applications may execute on open.
    """
    if value is None:
        return value
    if isinstance(value, (bool, int, float, Decimal)):
        return value
    text = value if isinstance(value, str) else (
        _stable_repr(value) if _contains_set(value) else str(value)
    )
    if not text:
        return text
    first_visible = 0
    while first_visible < len(text):
        character = text[first_visible]
        category = unicodedata.category(character)
        if character.isspace() or category in {"Cc", "Cf", "Cs", "Zl", "Zp", "Zs"}:
            first_visible += 1
            continue
        break
    begins_with_control = unicodedata.category(text[0]) == "Cc"
    begins_with_formula = first_visible < len(text) and text[first_visible] in "=+-@"
    if begins_with_control or begins_with_formula:
        return "'" + text
    return text


def _identity_cell(value: object) -> object:
    return _stable_repr(value) if _contains_set(value) else value


def _validate_finite_numbers(value: object, active: set[int] | None = None) -> None:
    if isinstance(value, Decimal):
        is_finite = value.is_finite()
    elif isinstance(value, Real):
        is_finite = math.isfinite(value)
    else:
        is_finite = True
    if not is_finite:
        raise ProjectError("output requires finite numeric values; NaN and Infinity are unsupported")
    if isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    else:
        return

    ancestors = active if active is not None else set()
    identity = id(value)
    if identity in ancestors:
        raise ProjectError("output values must not contain circular references")
    ancestors.add(identity)
    try:
        for nested_value in children:
            _validate_finite_numbers(nested_value, ancestors)
    finally:
        ancestors.remove(identity)


def _validate_json_mapping_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ProjectError("JSON output requires string mapping keys")
            _validate_json_mapping_keys(nested_value)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            _validate_json_mapping_keys(nested_value)


def render_json(value: object) -> str:
    """Serialize one JSON value using only standard JSON numbers and string keys."""
    _validate_finite_numbers(value)
    _validate_json_mapping_keys(value)
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProjectError("JSON output contains a value that cannot be represented as JSON") from error


def _tabular_columns(rows: list[dict[str, object]]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if not isinstance(key, str):
                raise ProjectError("tabular output requires string column names")
            if key not in seen:
                selected.append(key)
                seen.add(key)
    return selected


def _safe_table_text(value: object) -> str:
    text = _stable_repr(value) if _contains_set(value) else str(value)
    escaped: list[str] = []
    for character in text:
        if character.isprintable():
            escaped.append(character)
        else:
            codepoint = ord(character)
            if codepoint <= 0xFF:
                escaped.append(f"\\x{codepoint:02x}")
            elif codepoint <= 0xFFFF:
                escaped.append(f"\\u{codepoint:04x}")
            else:
                escaped.append(f"\\U{codepoint:08x}")
    return "".join(escaped)


def render_rows(
    rows: Iterable[Mapping[str, object]],
    output_format: str,
    *,
    spreadsheet_safe_csv: bool = False,
) -> str:
    """Render rows without mutating CSV values unless spreadsheet safety is requested.

    ``csv`` preserves the underlying text for machine consumers by default. Set
    ``spreadsheet_safe_csv`` when CSV may be opened by spreadsheet software that
    interprets formula-like cells.
    """
    materialized = [dict(row) for row in rows]
    for row in materialized:
        _validate_finite_numbers(row)
    if output_format == "json":
        return render_json(materialized)
    if output_format == "jsonl":
        return "".join(render_json(row) + "\n" for row in materialized)
    if output_format == "csv":
        if not materialized:
            return ""
        output = io.StringIO()
        fieldnames = _tabular_columns(materialized)
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        cell = _spreadsheet_safe_cell if spreadsheet_safe_csv else _identity_cell
        writer.writerow({key: cell(key) for key in fieldnames})
        writer.writerows({key: cell(row.get(key)) for key in fieldnames} for row in materialized)
        return output.getvalue()
    if output_format == "table":
        if not materialized:
            return "No results.\n"
        fieldnames = _tabular_columns(materialized)
        safe_headers = {column: _safe_table_text(column) for column in fieldnames}
        safe_rows = [
            {column: _safe_table_text(row.get(column, "")) for column in fieldnames}
            for row in materialized
        ]
        widths = {
            column: max(len(safe_headers[column]), *(len(row[column]) for row in safe_rows))
            for column in fieldnames
        }
        lines = ["  ".join(safe_headers[column].ljust(widths[column]) for column in fieldnames)]
        lines.append("  ".join("-" * widths[column] for column in fieldnames))
        lines.extend(
            "  ".join(row[column].ljust(widths[column]) for column in fieldnames)
            for row in safe_rows
        )
        return "\n".join(lines) + "\n"
    raise ProjectError("format must be json, jsonl, csv, or table")
