"""Small SemVer 2.0 validation shared by versioned project contracts."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

_SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)


@lru_cache(maxsize=1)
def software_version() -> str:
    """Return installed distribution version, or ``unknown`` from an unpackaged source tree."""
    try:
        return distribution_version("discourselens")
    except PackageNotFoundError:
        return "unknown"


def is_semantic_version(value: str) -> bool:
    """Return whether *value* follows Semantic Versioning 2.0.0.

    Numeric prerelease identifiers cannot have leading zeroes; build metadata
    identifiers may, as specified by SemVer.
    """
    match = _SEMVER.fullmatch(value)
    if match is None:
        return False
    prerelease = match.group(4)
    if prerelease is None:
        return True
    return all(
        not (identifier.isdigit() and len(identifier) > 1 and identifier.startswith("0"))
        for identifier in prerelease.split(".")
    )
