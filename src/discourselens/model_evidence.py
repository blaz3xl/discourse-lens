"""Typed evidence serialization for model requests, never for citation generation."""

from __future__ import annotations

from .errors import EvidenceValidationError
from .types import EvidenceKind, EvidenceRole


def model_evidence_record(
    project,
    evidence_id: str,
    kind: EvidenceKind,
    role: EvidenceRole,
    span_id: str | None,
) -> dict[str, object]:
    """Build a bounded, typed prompt record from a Core-owned evidence artifact.

    External URLs are deliberately omitted from model requests: the model needs
    the supplied source label, retrieval date and excerpt to review context, not
    query strings, fragments, access tokens or transport metadata.
    """
    record: dict[str, object] = {
        "evidence_id": evidence_id,
        "evidence_kind": kind.value,
        "role": role.value,
        "citation": project.render_citation(evidence_id),
    }
    if kind is EvidenceKind.TEXTUAL:
        if not span_id:
            raise EvidenceValidationError("textual evidence has no canonical source span")
        record["excerpt"] = project.span_text(span_id)
        record["provenance_class"] = "CANONICAL_DOCUMENT_TEXT"
        return record

    content = project.evidence_content(evidence_id)
    if kind is EvidenceKind.EXTERNAL_SOURCE:
        source = content.get("source")
        retrieved_at = content.get("retrieved_at")
        excerpt = content.get("excerpt")
        if not all(
            isinstance(value, str) and value.strip() for value in (source, retrieved_at, excerpt)
        ):
            raise EvidenceValidationError("external source evidence is incomplete")
        title = content.get("title")
        record.update(
            {
                "source": source,
                "title": title if isinstance(title, str) else None,
                "retrieved_at": retrieved_at,
                "excerpt": excerpt,
                "provenance_class": "EXTERNAL_SOURCE_USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
            }
        )
        return record

    if kind is EvidenceKind.USER_CONTEXT:
        label = content.get("label")
        user_content = content.get("content")
        if not all(isinstance(value, str) and value.strip() for value in (label, user_content)):
            raise EvidenceValidationError("user context evidence is incomplete")
        record.update(
            {
                "label": label,
                "content": user_content,
                "provenance_class": "RESEARCHER_SUPPLIED_CONTEXT_NOT_CORPUS_TEXT",
            }
        )
        return record

    raise EvidenceValidationError(
        f"evidence kind {kind.value!r} is not supported by this model evidence serializer"
    )
