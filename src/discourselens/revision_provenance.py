"""Stable, path-free extraction identity helpers for document revisions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

COORDINATE_SCHEME = "unicode-codepoint-half-open-v1"


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize a small provenance/structure record with stable JSON rules."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def basic_extraction_manifest(
    extractor: str, coordinate_scheme: str = COORDINATE_SCHEME
) -> str:
    """Honest fallback for importers that expose only canonical text and identity."""
    return canonical_json(
        {
            "manifest_version": "document-extraction-v1",
            "provenance_scope": "recorded-fields-only",
            "extractor_identity": extractor,
            "coordinate_scheme": coordinate_scheme,
        }
    )
