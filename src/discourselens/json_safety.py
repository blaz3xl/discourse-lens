"""Small, parser-independent guards for untrusted JSON input."""

from __future__ import annotations

MAX_JSON_NESTING_DEPTH = 64


class JsonNestingLimitError(ValueError):
    """Raised when JSON contains more nested containers than the contract allows."""


def validate_json_nesting(
    text: str, *, max_depth: int = MAX_JSON_NESTING_DEPTH
) -> None:
    """Reject excessive JSON container depth without depending on CPython's parser limit.

    This is a lexical preflight only; ``json.loads`` remains responsible for validating
    JSON syntax. Braces and brackets inside escaped or ordinary string content do not
    count as containers.
    """
    if not isinstance(text, str):
        raise TypeError("JSON nesting validation requires text")
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("maximum JSON nesting depth must be a non-negative integer")

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > max_depth:
                raise JsonNestingLimitError(
                    "JSON nesting exceeds the configured depth limit"
                )
        elif character in "]}":
            # Malformed delimiter order is left to json.loads; keeping this lexical
            # pass tolerant avoids turning it into a second JSON parser.
            depth = max(0, depth - 1)
