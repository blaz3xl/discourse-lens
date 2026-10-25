"""Bounded continuous reading with explicit coverage and canonical locations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import DocumentError

if TYPE_CHECKING:
    from .project import Project

MAX_DOCUMENT_READING_CHARS = 100_000
MAX_READING_UNITS = 128


def read_document(
    project: Project, revision_id: str, *, start_offset: int = 0, max_chars: int = 100_000
) -> dict[str, object]:
    """Read one continuous canonical interval, partitioned by stored source units."""
    if type(start_offset) is not int or start_offset < 0:
        raise DocumentError("document reading start_offset must be a nonnegative integer")
    if type(max_chars) is not int or not 1 <= max_chars <= MAX_DOCUMENT_READING_CHARS:
        raise DocumentError("document reading max_chars must be between 1 and 100000")
    summary = project.document_revision_summary(revision_id)
    text = project.revision_text(revision_id)
    if start_offset > len(text):
        raise DocumentError("document reading start_offset exceeds the canonical text")
    end_offset = min(len(text), start_offset + max_chars)
    pages = project.db.execute(
        "SELECT page_index,page_label,canonical_start,canonical_end FROM pages "
        "WHERE revision_id=? AND canonical_end>? AND canonical_start<? ORDER BY page_index LIMIT 129",
        (revision_id, start_offset, end_offset),
    ).fetchall()
    units = []
    if pages:
        if len(pages) > MAX_READING_UNITS:
            pages = pages[:MAX_READING_UNITS]
            end_offset = min(end_offset, pages[-1]["canonical_end"])
        for page in pages:
            start, end = max(start_offset, page["canonical_start"]), min(end_offset, page["canonical_end"])
            units.append({
                "revision_id": revision_id,
                "page_index": page["page_index"],
                "page_label": page["page_label"],
                "start_offset": start, "end_offset": end,
                "citation": f"[{summary['display_name']} · p. {page['page_label']}]",
                "text": text[start:end],
            })
    elif text and start_offset < end_offset:
        paragraphs = project.db.execute(
            "SELECT ordinal,start_offset,end_offset FROM paragraphs WHERE revision_id=? "
            "AND end_offset>? AND start_offset<? ORDER BY ordinal LIMIT 129",
            (revision_id, start_offset, end_offset),
        ).fetchall()
        if len(paragraphs) > MAX_READING_UNITS:
            paragraphs = paragraphs[:MAX_READING_UNITS]
            end_offset = min(end_offset, paragraphs[-1]["end_offset"])
        for paragraph in paragraphs:
            start, end = max(start_offset, paragraph["start_offset"]), min(end_offset, paragraph["end_offset"])
            units.append({
                "revision_id": revision_id, "start_offset": start, "end_offset": end,
                "citation": f"[{summary['display_name']} · ¶{paragraph['ordinal'] + 1}]",
                "text": text[start:end],
            })
    return {
        "revision_id": revision_id, "display_name": summary["display_name"],
        "source_sha256": summary["source_hash"], "canonical_sha256": summary["canonical_hash"],
        "total_characters": len(text), "total_pages": summary["page_count"],
        "start_offset": start_offset, "end_offset": end_offset,
        "coverage": "complete_document" if start_offset == 0 and end_offset == len(text) else "partial_document",
        "has_more": end_offset < len(text),
        "next_start_offset": end_offset if end_offset < len(text) else None,
        "units": units,
    }
