from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from xml.etree.ElementTree import ParseError
from zipfile import ZipFile

import pymupdf as fitz
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from .config import DEFAULT_IMPORT_MAX_BYTES
from .errors import DocumentError
from .observability import log_event
from .revision_provenance import (
    COORDINATE_SCHEME,
    basic_extraction_manifest,
    canonical_json,
    stable_payload_hash,
)

DEFAULT_MAX_IMPORT_BYTES = DEFAULT_IMPORT_MAX_BYTES
_LOGGER = logging.getLogger(__name__)
_MAX_HTML_IGNORED_NESTING = 1_024
HTML_SUFFIXES = {".html", ".htm"}
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdown", ".mkdn"}
DOCX_SUFFIXES = {".docx"}
DEFAULT_DOCX_XML_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_DOCX_XML_TOTAL_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_DOCX_EXPANDED_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_DOCX_ARCHIVE_MAX_ENTRIES = 10_000
_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
BBOX_PROJECTION_VERSION = "pymupdf-char-range-v1"
PDF_EXTRACTOR_IMPLEMENTATION_VERSION = "discourselens-pdf-import-v1"
PDF_NORMALIZATION_VERSION = "pdf-text-preserve-v1"
PDF_READING_ORDER_VERSION = "pymupdf-sorted-blocks-v1"
PDF_PARAGRAPH_SEGMENTATION_VERSION = "pymupdf-block-v1"
PDF_CANONICAL_ASSEMBLY_VERSION = "page-ff-newline-v1"
DEFAULT_OCR_DPI = 300
AUTO_OCR_LANGUAGES = "eng+spa+nld"
AUTO_OCR_MIN_PAGE_CHARS = 40
# Bound page-by-page parsing and retained extraction metadata for digital PDFs.
MAX_PDF_PAGES = 1_000
# Bound the canonical text and structural rows retained after PDF extraction.
# These defaults admit many ordinary books while keeping compressed text streams
# from expanding into unbounded project data. Callers may override them per import.
MAX_PDF_CANONICAL_CHARS = 10_000_000
MAX_PDF_LAYOUT_BLOCKS = 100_000
# A rendered A4/Letter page at 600 DPI is about 35 million pixels. Keep a
# per-page ceiling near that scale to bound raster memory, while permitting
# ordinary pages at the maximum supported DPI. The aggregate budget bounds
# CPU work across otherwise tiny, many-page PDFs; at 300 DPI it allows roughly
# 950 Letter pages. Explicit local OCR remains opt-in.
MAX_OCR_PAGE_PIXELS = 40_000_000
MAX_OCR_TOTAL_PIXELS = 8_000_000_000
MAX_OCR_PAGES = 1_000
_OCR_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+(?:\+[A-Za-z0-9_-]+)*$")


class _CanonicalHtmlParser(HTMLParser):
    """Small local HTML extractor with deterministic block boundaries.

    This is intentionally not a browser: scripts/styles are excluded, no URLs are
    fetched, and only visible text blocks become the canonical revision text.
    """

    _BLOCK_TAGS: ClassVar[set[str]] = {
        "article",
        "blockquote",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "pre",
    }
    _IGNORED_TAGS: ClassVar[set[str]] = {"script", "style", "template", "head"}
    _VOID_TAGS: ClassVar[set[str]] = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.parts: list[str] = []
        self.ignored_tags: list[str] = []
        self.saw_markup = False

    def handle_starttag(self, tag: str, attrs) -> None:
        self.saw_markup = True
        tag = tag.casefold()
        if tag not in self._VOID_TAGS and (
            self.ignored_tags or tag in self._IGNORED_TAGS or self._has_hidden_attribute(attrs)
        ):
            if len(self.ignored_tags) >= _MAX_HTML_IGNORED_NESTING:
                raise DocumentError("HTML ignored-content nesting exceeds the safety limit")
            self.ignored_tags.append(tag)
        if tag in self._BLOCK_TAGS or tag == "br":
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        self.saw_markup = True
        tag = tag.casefold()
        if self.ignored_tags:
            if tag in {"body", "html"}:
                self.ignored_tags.clear()
            else:
                self._close_ignored_tag(tag)
            if tag in self._BLOCK_TAGS or tag in {"body", "html"}:
                self._flush()
            return
        if tag in self._BLOCK_TAGS or tag in {"body", "html"}:
            self._flush()

    def handle_startendtag(self, tag: str, attrs) -> None:
        """Apply HTML, rather than XML, rules to self-closing syntax.

        In HTML, a trailing slash does not close non-void elements. Treating
        ``<div hidden/>`` as a closed element would expose its following text to
        extraction even though an HTML browser keeps that text inside the
        hidden ``div`` until its explicit end tag.
        """
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if not self.ignored_tags:
            self.parts.append(data)

    def text(self) -> str:
        self._flush()
        return "\n".join(self.blocks)

    def _flush(self) -> None:
        text = " ".join("".join(self.parts).split())
        self.parts.clear()
        if text:
            self.blocks.append(text)

    @staticmethod
    def _has_hidden_attribute(attrs) -> bool:
        attributes = {name.casefold(): value for name, value in attrs}
        if "hidden" in attributes or str(attributes.get("aria-hidden") or "").strip().casefold() == "true":
            return True
        style = str(attributes.get("style") or "")
        for declaration in style.split(";"):
            name, separator, value = declaration.partition(":")
            if not separator:
                continue
            property_name = name.strip().casefold()
            property_value = value.partition("!")[0].strip().casefold()
            if (property_name == "display" and property_value == "none") or (
                property_name == "visibility" and property_value in {"hidden", "collapse"}
            ):
                return True
        return False

    def _close_ignored_tag(self, tag: str) -> None:
        for index in range(len(self.ignored_tags) - 1, -1, -1):
            if self.ignored_tags[index] == tag:
                del self.ignored_tags[index:]
                return


@dataclass(frozen=True)
class ImportResult:
    document_id: str | None
    revision_id: str | None
    status: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PdfExtractionLimits:
    """Per-import caps for PDF canonical text and extracted layout blocks."""

    max_canonical_chars: int = MAX_PDF_CANONICAL_CHARS
    max_layout_blocks: int = MAX_PDF_LAYOUT_BLOCKS


BatchImportStatus = Literal["SUCCESS", "DUPLICATE", "FAILED"]


@dataclass(frozen=True, slots=True)
class BatchImportItemResult:
    """Serializable outcome for one input in a sequential batch import.

    ``index`` is zero-based and refers to the input's iteration order. Paths and
    exception messages are intentionally omitted so callers can safely retain or
    serialize these outcomes without exposing local filesystem details.
    """

    index: int
    status: BatchImportStatus
    document_id: str | None = None
    revision_id: str | None = None
    failure_code: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchImportResult:
    """Ordered batch outcomes and their aggregate status counts."""

    items: tuple[BatchImportItemResult, ...]
    total_count: int
    success_count: int
    duplicate_count: int
    failed_count: int


class _PdfBlockExtraction(list[tuple[str, tuple[float, float, float, float]]]):
    """List-compatible PDF blocks with a page-level reading-order diagnostic."""

    def __init__(
        self,
        blocks: list[tuple[str, tuple[float, float, float, float]]],
        *,
        reading_order_ambiguous: bool,
    ) -> None:
        super().__init__(blocks)
        self.reading_order_ambiguous = reading_order_ambiguous


def _log_import_result(result: ImportResult, byte_count: int) -> ImportResult:
    log_event(
        _LOGGER,
        "document.import.completed",
        document_id=result.document_id,
        revision_id=result.revision_id,
        status=result.status,
        byte_count=byte_count,
    )
    return result


def _reject_import_inside_caller_transaction(project) -> None:
    """Keep import APIs from committing unrelated caller-owned DB changes."""
    if project.db.in_transaction:
        raise DocumentError("document import cannot run inside an existing transaction")


def _read_import_source(path: str | Path, max_bytes: int) -> tuple[Path, bytes]:
    """Read one stable, bounded regular file without following source symlinks.

    The size limit is enforced against the opened file descriptor and while reading,
    not against a prior pathname stat. This prevents a replaced/growing file from
    bypassing the configured bound between validation and parsing.
    """
    source_path = Path(path)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise DocumentError("document import size limit must be a positive integer")
    try:
        path_stat = source_path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise DocumentError("document source path must not be a symlink")
        if not stat.S_ISREG(path_stat.st_mode):
            raise DocumentError("document path is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source_path, flags)
    except DocumentError:
        raise
    except OSError as exc:
        raise DocumentError("document path cannot be opened safely") from exc

    try:
        with os.fdopen(descriptor, "rb") as source:
            opened_stat = os.fstat(source.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise DocumentError("document path is not a regular file")
            if (opened_stat.st_dev, opened_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
                raise DocumentError("document source changed while it was being opened")
            if opened_stat.st_size > max_bytes:
                raise DocumentError("document exceeds configured import size limit")
            raw = source.read(max_bytes + 1)
            final_stat = os.fstat(source.fileno())
            if len(raw) > max_bytes or final_stat.st_size > max_bytes:
                raise DocumentError("document exceeds configured import size limit")
            if (
                final_stat.st_size != opened_stat.st_size
                or final_stat.st_mtime_ns != opened_stat.st_mtime_ns
                or final_stat.st_ctime_ns != opened_stat.st_ctime_ns
            ):
                raise DocumentError("document source changed while it was being read")
            if len(raw) != final_stat.st_size:
                raise DocumentError("document source changed while it was being read")
    except OSError as exc:
        raise DocumentError("document source cannot be read safely") from exc
    return source_path, raw


def _reserve_source_hash_import(project) -> bool:
    """Serialize standalone de-duplication checks with their ensuing insert.

    Without a write reservation, two project connections can both observe a
    missing source hash and create separate document/revision rows for the same
    bytes. ``import_text`` commits the reserved transaction when it inserts the
    revision. An already-open caller transaction is left untouched.
    """
    if getattr(project, "read_only", False) or project.db.in_transaction:
        return False
    project.db.execute("BEGIN IMMEDIATE")
    return True


def _release_source_hash_import(project, reserved: bool) -> None:
    if reserved and project.db.in_transaction:
        project.db.rollback()


def _find_duplicate_source_revision(
    project,
    *,
    source_hash: str,
    canonical_text: str,
    mime_type: str,
    extractor: str,
    document_id: str | None = None,
    extraction_manifest_json: str | None = None,
    structure_hash: str | None = None,
):
    """Find an exact duplicate using the import provenance as well as bytes.

    Identical bytes can be interpreted differently by format-specific importers
    (for example, HTML markup is preserved by the plain-text importer but
    discarded by the visible-HTML importer), or with different extraction
    settings (for example, explicit local OCR). Returning a duplicate based only
    on ``source_hash`` would silently reuse a revision with different canonical
    text or provenance and could even bypass importer validation. A duplicate
    therefore needs to match the source bytes, canonical text, media type, and
    extractor identity.
    """
    canonical_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    manifest = extraction_manifest_json or basic_extraction_manifest(extractor)
    structural_hash = structure_hash or canonical_hash
    query = (
        "SELECT document_id,id FROM document_revisions "
        "WHERE source_hash=? AND canonical_hash=? AND mime_type=? AND extractor=? "
        "AND extraction_manifest_json=? AND structure_hash=?"
    )
    parameters: tuple[str, ...] = (
        source_hash,
        canonical_hash,
        mime_type,
        extractor,
        manifest,
        structural_hash,
    )
    if document_id is not None:
        query += " AND document_id=?"
        parameters += (document_id,)
    return project.db.execute(
        query + " ORDER BY created_at,id LIMIT 1", parameters
    ).fetchone()


def _pdf_structure_payload(
    pages: list[
        tuple[int, float, float, int, str, list[tuple[str, tuple[float, float, float, float]]]]
    ],
) -> dict[str, Any]:
    """Describe the persisted PDF map without random IDs or host filesystem paths."""
    page_records: list[dict[str, Any]] = []
    paragraph_records: list[dict[str, Any]] = []
    canonical_offset = 0
    for position, (page_index, width, height, rotation, text, blocks) in enumerate(pages):
        page_start = canonical_offset
        page_end = page_start + len(text)
        page_records.append(
            {
                "page_index": page_index,
                "width": width,
                "height": height,
                "rotation": rotation,
                "canonical_start": page_start,
                "canonical_end": page_end,
                "text": text,
            }
        )
        page_cursor = 0
        for block_index, (content, bbox) in enumerate(blocks):
            relative_start = text.find(content, page_cursor)
            paragraph_records.append(
                {
                    "page_index": page_index,
                    "block_index": block_index,
                    "canonical_start": (
                        page_start + relative_start if relative_start >= 0 else None
                    ),
                    "canonical_end": (
                        page_start + relative_start + len(content)
                        if relative_start >= 0
                        else None
                    ),
                    "text": content,
                    "segmentation_version": PDF_PARAGRAPH_SEGMENTATION_VERSION,
                    "bounding_boxes": [list(bbox)],
                }
            )
            if relative_start >= 0:
                page_cursor = relative_start + len(content)
        canonical_offset = page_end + (2 if position < len(pages) - 1 else 0)
    return {"pages": page_records, "paragraphs": paragraph_records}


def _find_document_for_source(project, source_hash: str, mime_type: str) -> str | None:
    """Reuse document identity when the same media is re-extracted differently."""
    row = project.db.execute(
        "SELECT document_id FROM document_revisions WHERE source_hash=? AND mime_type=? "
        "ORDER BY created_at,id LIMIT 1",
        (source_hash, mime_type),
    ).fetchone()
    return row["document_id"] if row else None


def _discard_failed_pdf_import(project, revision_id: str) -> None:
    """Roll back a deferred PDF import and remove its newly written source files."""
    if project.db.in_transaction:
        project.db.rollback()
    if project.db.execute(
        "SELECT 1 FROM document_revisions WHERE id=?", (revision_id,)
    ).fetchone():
        return
    source_paths: list[Path] = []
    for stored_path in (f"source/{revision_id}.txt", f"source/{revision_id}.original"):
        try:
            source_paths.append(project._resolve_managed_source_file(stored_path, must_exist=False))
        except DocumentError:
            _LOGGER.warning("could not resolve source file from failed PDF import")
    project._remove_uncommitted_source_files(source_paths)


def project_pdf_text_boxes(
    source_bytes: bytes,
    page_index: int,
    paragraphs: list[dict[str, object]],
    start_offset: int,
    end_offset: int,
    *,
    ocr_language: str | None = None,
    ocr_dpi: int | None = None,
) -> tuple[list[tuple[float, float, float, float]], str, str | None]:
    """Project one page-local canonical range to best-effort text geometry.

    PyMuPDF's character boxes are transient extraction data: only boxes touched by
    the requested SourceSpan are returned and persisted. The paragraph/block text
    must exactly align first; otherwise geometry is reported unavailable rather
    than guessed from an enclosing paragraph rectangle.
    """
    if (ocr_language is None) != (ocr_dpi is None):
        raise DocumentError("OCR language and DPI must be provided together for PDF geometry")
    if ocr_language is not None and ocr_dpi is not None:
        _validate_ocr_options(ocr_language, ocr_dpi)
    try:
        with fitz.open(stream=source_bytes, filetype="pdf") as pdf:
            if page_index < 0 or page_index >= pdf.page_count:
                return [], "UNAVAILABLE", None
            page = pdf[page_index]
            if ocr_language is not None and ocr_dpi is not None:
                _validate_pdf_ocr_workload(pdf, ocr_dpi)
                text_page = page.get_textpage_ocr(language=ocr_language, dpi=ocr_dpi, full=False)
                raw_dict = cast(Any, page.get_text("rawdict", textpage=text_page, sort=True))
            else:
                raw_dict = cast(Any, page.get_text("rawdict", sort=True))
            raw_blocks = [block for block in raw_dict["blocks"] if block.get("type") == 0]
    except (fitz.FileDataError, ValueError, IndexError, RuntimeError):
        return [], "UNAVAILABLE", None

    cursor = 0
    boxes: list[tuple[float, float, float, float]] = []
    total_chars = 0
    mapped_chars = 0
    touched_paragraphs: list[str] = []
    for paragraph in paragraphs:
        raw_text = paragraph.get("text")
        raw_start = paragraph.get("start_offset")
        raw_end = paragraph.get("end_offset")
        if (
            not isinstance(raw_text, str)
            or not isinstance(raw_start, int)
            or not isinstance(raw_end, int)
        ):
            continue
        text = raw_text
        paragraph_start = raw_start
        paragraph_end = raw_end
        if paragraph_end <= start_offset or paragraph_start >= end_offset:
            continue
        overlap_start = max(start_offset, paragraph_start)
        overlap_end = min(end_offset, paragraph_end)
        local_start, local_end = overlap_start - paragraph_start, overlap_end - paragraph_start
        total_chars += sum(not char.isspace() for char in text[local_start:local_end])

        match: tuple[str, list[tuple[float, float, float, float] | None]] | None = None
        for block_index in range(cursor, len(raw_blocks)):
            block = raw_blocks[block_index]
            line_texts: list[str] = []
            line_boxes: list[list[tuple[float, float, float, float] | None]] = []
            for line in block.get("lines", []):
                chars = [char for span in line.get("spans", []) for char in span.get("chars", [])]
                line_text = "".join(str(char.get("c", "")) for char in chars)
                char_boxes: list[tuple[float, float, float, float] | None] = []
                for char in chars:
                    value = str(char.get("c", ""))
                    bbox = char.get("bbox")
                    rectangle = (
                        (
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        )
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
                        else None
                    )
                    char_boxes.extend([rectangle] * len(value))
                line_texts.append(line_text)
                line_boxes.append(char_boxes)

            block_text = "\n".join(line_texts)
            block_geometry: list[tuple[float, float, float, float] | None] = []
            for line_index, char_boxes in enumerate(line_boxes):
                if line_index:
                    block_geometry.append(None)  # canonical line break has no glyph box
                block_geometry.extend(char_boxes)
            left_trim = len(block_text) - len(block_text.lstrip())
            right_trim = len(block_text.rstrip())
            block_text = block_text[left_trim:right_trim]
            block_geometry = block_geometry[left_trim:right_trim]
            if block_text == text:
                cursor = block_index + 1
                match = (block_text, block_geometry)
                break

        if match is None:
            continue
        raw_paragraph_id = paragraph.get("id")
        if raw_paragraph_id is not None:
            touched_paragraphs.append(str(raw_paragraph_id))
        for char, rectangle in zip(
            match[0][local_start:local_end], match[1][local_start:local_end]
        ):
            if char.isspace():
                continue
            if rectangle is None:
                continue
            mapped_chars += 1
            if boxes and _boxes_are_adjacent(boxes[-1], rectangle):
                previous = boxes[-1]
                boxes[-1] = (
                    min(previous[0], rectangle[0]),
                    min(previous[1], rectangle[1]),
                    max(previous[2], rectangle[2]),
                    max(previous[3], rectangle[3]),
                )
            else:
                boxes.append(rectangle)

    if not boxes:
        status = "UNAVAILABLE"
    elif mapped_chars < total_chars:
        status = "PARTIAL"
    else:
        status = "AVAILABLE"
    paragraph_id = touched_paragraphs[0] if len(touched_paragraphs) == 1 else None
    return boxes, status, paragraph_id


def _boxes_are_adjacent(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
) -> bool:
    vertical_tolerance = max(1.0, (previous[3] - previous[1]) * 0.15)
    same_line = (
        abs(previous[1] - current[1]) <= vertical_tolerance
        and abs(previous[3] - current[3]) <= vertical_tolerance
    )
    horizontal_gap = max(previous[0], current[0]) - min(previous[2], current[2])
    return same_line and horizontal_gap <= 1.5


def _extract_pdf_blocks(
    page: fitz.Page,
    *,
    ocr_language: str | None,
    ocr_dpi: int,
) -> _PdfBlockExtraction:
    """Extract reading-order text blocks, optionally using local Tesseract OCR."""
    if ocr_language is None:
        text_dict = cast(Any, page.get_text("dict", sort=True))
        extracted = page.get_text("blocks", sort=True)
    else:
        text_page = page.get_textpage_ocr(language=ocr_language, dpi=ocr_dpi, full=False)
        text_dict = cast(Any, page.get_text("dict", textpage=text_page, sort=True))
        extracted = page.get_text("blocks", textpage=text_page, sort=True)
    blocks = [
        (
            block[4].strip(),
            (float(block[0]), float(block[1]), float(block[2]), float(block[3])),
        )
        for block in extracted
        if block[4].strip()
    ]
    return _PdfBlockExtraction(
        blocks,
        reading_order_ambiguous=_has_horizontal_lane_ambiguity(text_dict),
    )


def _has_horizontal_lane_ambiguity(page_dict: dict[str, Any]) -> bool:
    """Detect a narrow set of horizontal lanes that sorted blocks may interleave.

    This is not a general layout analyzer. It detects either two vertically
    distinct line pairs separated by a consistent wide gap, or one wide-gutter
    pair followed by more text in the same horizontal lane. The latter catches
    short sidebars/unequal columns that can be interleaved without repeated rows.
    """
    line_boxes: list[tuple[float, float, float, float]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            raw_box = line.get("bbox")
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                continue
            line_boxes.append(
                (
                    float(raw_box[0]),
                    float(raw_box[1]),
                    float(raw_box[2]),
                    float(raw_box[3]),
                )
            )

    line_boxes.sort(key=lambda box: (box[1], box[0]))
    active: list[tuple[float, float, float, float]] = []
    repeated_gaps: list[tuple[float, float]] = []
    lane_pairs: list[
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            float,
            float,
        ]
    ] = []
    for current in line_boxes:
        active = [previous for previous in active if previous[3] > current[1]]
        for previous in active:
            overlap = min(previous[3], current[3]) - max(previous[1], current[1])
            minimum_height = min(previous[3] - previous[1], current[3] - current[1])
            if minimum_height <= 0 or overlap < minimum_height * 0.6:
                continue

            left, right = sorted((previous, current), key=lambda box: box[0])
            gap = right[0] - left[2]
            if gap < 36.0:
                continue
            boundary = (left[2] + right[0]) / 2
            baseline = (previous[1] + previous[3] + current[1] + current[3]) / 4
            repeated_gaps.append((boundary, baseline))
            lane_pairs.append((left, right, boundary, baseline))

        active.append(current)

    repeated_gaps.sort(key=lambda item: item[0])
    boundary_clusters: list[tuple[float, list[float]]] = []
    for boundary, baseline in repeated_gaps:
        if not boundary_clusters or boundary - boundary_clusters[-1][0] > 24.0:
            boundary_clusters.append((boundary, [baseline]))
        else:
            boundary_clusters[-1][1].append(baseline)

    if any(
        max(baselines) - min(baselines) >= 5.0
        for _, baselines in boundary_clusters
        if len(baselines) >= 2
    ):
        return True

    for left, right, boundary, baseline in lane_pairs:
        if right[0] - left[2] < 72.0:
            continue
        for continuation in line_boxes:
            continuation_baseline = (continuation[1] + continuation[3]) / 2
            if continuation_baseline - baseline < 5.0:
                continue
            continues_left_lane = (
                abs(continuation[0] - left[0]) <= 24.0 and continuation[2] < boundary
            )
            continues_right_lane = (
                abs(continuation[0] - right[0]) <= 24.0 and continuation[0] > boundary
            )
            if continues_left_lane or continues_right_lane:
                return True

    return False


def import_pdf(
    project,
    path: str | Path,
    *,
    document_id: str | None = None,
    extractor_suffix: str = "",
    max_bytes: int | None = None,
    ocr_language: str | None = None,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    pdf_extraction_limits: PdfExtractionLimits | None = None,
) -> ImportResult:
    _reject_import_inside_caller_transaction(project)
    _validate_import_ocr_options(ocr_language, ocr_dpi)
    max_bytes = project.config.import_max_bytes if max_bytes is None else max_bytes
    source_path, raw = _read_import_source(path, max_bytes)
    return _import_pdf_bytes(
        project,
        source_path.name,
        raw,
        extractor_suffix=extractor_suffix,
        document_id=document_id,
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        pdf_extraction_limits=pdf_extraction_limits,
    )


def _import_pdf_bytes(
    project,
    filename: str,
    raw: bytes,
    *,
    extractor_suffix: str = "",
    document_id: str | None = None,
    ocr_language: str | None = None,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    auto_ocr: bool = False,
    pdf_extraction_limits: PdfExtractionLimits | None = None,
) -> ImportResult:
    extraction_limits = _resolve_pdf_extraction_limits(pdf_extraction_limits)
    if not raw.startswith(b"%PDF"):
        raise DocumentError("PDF signature is invalid")
    try:
        pdf = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise DocumentError("PDF cannot be parsed") from exc
    with pdf:
        if pdf.needs_pass:
            return ImportResult(None, None, "PASSWORD_REQUIRED", ("ENCRYPTED_PDF",))
        if pdf.page_count > MAX_PDF_PAGES:
            raise DocumentError("PDF exceeds configured page-count safety limit")
        if ocr_language is not None:
            _validate_pdf_ocr_workload(pdf, ocr_dpi)
        pages: list[
            tuple[int, float, float, int, str, list[tuple[str, tuple[float, float, float, float]]]]
        ] = []
        warnings: list[str] = []
        ocr_languages_used: set[str] = set()
        extracted_text_chars = 0
        extracted_layout_blocks = 0
        for index in range(pdf.page_count):
            page = pdf[index]
            try:
                text_blocks = _extract_pdf_blocks(
                    page, ocr_language=ocr_language, ocr_dpi=ocr_dpi
                )
                page_chars = sum(len(content.strip()) for content, _ in text_blocks)
                if auto_ocr and ocr_language is None and page_chars < AUTO_OCR_MIN_PAGE_CHARS:
                    _validate_pdf_ocr_workload(pdf, DEFAULT_OCR_DPI)
                    try:
                        ocr_blocks = _extract_pdf_blocks(
                            page,
                            ocr_language=AUTO_OCR_LANGUAGES,
                            ocr_dpi=DEFAULT_OCR_DPI,
                        )
                    except Exception as exc:
                        warnings.append("AUTOMATIC_OCR_UNAVAILABLE")
                        _LOGGER.warning("automatic PDF OCR could not process a page", exc_info=exc)
                    else:
                        ocr_chars = sum(len(content.strip()) for content, _ in ocr_blocks)
                        if ocr_chars > page_chars:
                            text_blocks = ocr_blocks
                            ocr_languages_used.add(AUTO_OCR_LANGUAGES)
                            warnings.append("AUTOMATIC_OCR_APPLIED")
                if getattr(text_blocks, "reading_order_ambiguous", False):
                    warnings.append("AMBIGUOUS_READING_ORDER")
            except Exception as exc:
                if ocr_language is not None:
                    raise DocumentError(
                        "local OCR failed; install Tesseract and the requested tessdata language"
                    ) from exc
                raise DocumentError("PDF text extraction failed") from exc
            page_text_chars = sum(len(content) for content, _ in text_blocks)
            if len(text_blocks) > 1:
                page_text_chars += len(text_blocks) - 1
            candidate_text_chars = extracted_text_chars + page_text_chars
            if pages:
                candidate_text_chars += 2  # frozen canonical page separator: "\f\n"
            if candidate_text_chars > extraction_limits.max_canonical_chars:
                raise DocumentError("PDF extracted text exceeds configured character limit")
            candidate_layout_blocks = extracted_layout_blocks + len(text_blocks)
            if candidate_layout_blocks > extraction_limits.max_layout_blocks:
                raise DocumentError("PDF extracted layout exceeds configured block limit")
            text = "\n".join(item[0] for item in text_blocks)
            if not text_blocks:
                warnings.append("PAGE_EXTRACTION_FAILED")
            extracted_text_chars = candidate_text_chars
            extracted_layout_blocks = candidate_layout_blocks
            pages.append(
                (index, page.rect.width, page.rect.height, page.rotation, text, text_blocks)
            )

    source_hash = hashlib.sha256(raw).hexdigest()
    reserved = _reserve_source_hash_import(project)
    revision_id: str | None = None
    try:
        if not any(page[4] for page in pages):
            no_text_warnings = ["NO_TEXT_FOUND"]
            if ocr_language is not None:
                no_text_warnings.append("OCR_NO_TEXT")
            return ImportResult(None, None, "OCR_REQUIRED", tuple(no_text_warnings))
        if ocr_language is not None:
            warnings.append("LOCAL_OCR_ENABLED")
        canonical = "\f\n".join(page[4] for page in pages)
        extractor = f"pymupdf@{fitz.VersionBind}{extractor_suffix}"
        used_ocr_languages = sorted(ocr_languages_used | ({ocr_language} if ocr_language else set()))
        if used_ocr_languages:
            extractor += f"+tesseract-ocr[lang={'+'.join(used_ocr_languages)},dpi={ocr_dpi},full=false]"
        extraction_manifest_json = canonical_json(
            {
                "manifest_version": "document-extraction-v1",
                "provenance_scope": "complete-pdf-pipeline",
                "engine": "PyMuPDF",
                "engine_version": str(fitz.VersionBind),
                "extractor_identity": extractor,
                "extractor_implementation_version": PDF_EXTRACTOR_IMPLEMENTATION_VERSION,
                "normalization_version": PDF_NORMALIZATION_VERSION,
                "reading_order_version": PDF_READING_ORDER_VERSION,
                "paragraph_segmentation_version": PDF_PARAGRAPH_SEGMENTATION_VERSION,
                "canonical_assembly_version": PDF_CANONICAL_ASSEMBLY_VERSION,
                "coordinate_scheme": COORDINATE_SCHEME,
                "bbox_projection_version": BBOX_PROJECTION_VERSION,
                "ocr": {
                    "enabled": bool(used_ocr_languages),
                    "language": "+".join(used_ocr_languages) if used_ocr_languages else None,
                    "dpi": ocr_dpi if used_ocr_languages else None,
                },
            }
        )
        structure_hash = stable_payload_hash(_pdf_structure_payload(pages))
        existing = _find_duplicate_source_revision(
            project,
            source_hash=source_hash,
            canonical_text=canonical,
            mime_type="application/pdf",
            extractor=extractor,
            document_id=document_id,
            extraction_manifest_json=extraction_manifest_json,
            structure_hash=structure_hash,
        )
        if existing:
            return ImportResult(existing["document_id"], existing["id"], "DUPLICATE")
        target_document_id = document_id or _find_document_for_source(
            project, source_hash, "application/pdf"
        )
        imported_document_id, revision_id = project.import_text(
            filename,
            canonical,
            raw,
            extractor,
            document_id=target_document_id,
            mime_type="application/pdf",
            create_paragraphs=False,
            extraction_status="SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS",
            ocr_language="+".join(used_ocr_languages) if used_ocr_languages else None,
            ocr_dpi=ocr_dpi if used_ocr_languages else None,
            _extraction_manifest_json=extraction_manifest_json,
            _structure_hash=structure_hash,
            _defer_commit=True,
        )
        offset = 0
        ordinal = 0
        for page_position, (index, width, height, rotation, text, blocks) in enumerate(pages):
            pid = project._new_page(
                revision_id, index, width, height, rotation, offset, offset + len(text), text
            )
            page_cursor = 0
            for content, bbox in blocks:
                relative_start = text.find(content, page_cursor)
                if relative_start < 0:
                    # We never fabricate coordinates.  The page text remains available, but
                    # this block cannot be projected accurately.
                    warnings.append("TEXT_MAPPING_PARTIAL")
                    continue
                start = offset + relative_start
                end = start + len(content)
                paragraph_id = project.uid()
                project.db.execute(
                    "INSERT INTO paragraphs(id,revision_id,ordinal,start_offset,end_offset,page_number,page_id,text,segmentation_version,bounding_boxes_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        paragraph_id,
                        revision_id,
                        ordinal,
                        start,
                        end,
                        index,
                        pid,
                        content,
                        PDF_PARAGRAPH_SEGMENTATION_VERSION,
                        json.dumps([bbox]),
                    ),
                )
                project.db.execute(
                    "INSERT INTO revision_fts VALUES(?,?,?)", (revision_id, paragraph_id, content)
                )
                ordinal += 1
                page_cursor = relative_start + len(content)
            offset += len(text)
            if page_position < len(pages) - 1:
                offset += 2  # frozen canonical page separator: "\f\n"
        project._persist_sentences(revision_id)
        extraction_status = "SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS"
        project.db.execute(
            "UPDATE document_revisions SET extraction_status=? WHERE id=?",
            (extraction_status, revision_id),
        )
        for warning in sorted(set(warnings)):
            project.record_extraction_warning(revision_id, warning, commit=False)
        project.db.commit()
    except BaseException:
        try:
            if revision_id is not None:
                _discard_failed_pdf_import(project, revision_id)
            elif project.db.in_transaction:
                project.db.rollback()
        except Exception:
            if project.db.in_transaction:
                project.db.rollback()
            _LOGGER.exception("could not clean up an incomplete PDF import")
        raise
    finally:
        _release_source_hash_import(project, reserved)
    return ImportResult(
        imported_document_id,
        revision_id,
        "SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS",
        tuple(sorted(set(warnings))),
    )


def _resolve_pdf_extraction_limits(
    limits: PdfExtractionLimits | None,
) -> PdfExtractionLimits:
    if limits is None:
        return PdfExtractionLimits()
    if not isinstance(limits, PdfExtractionLimits) or any(
        type(value) is not int or value < 1
        for value in (limits.max_canonical_chars, limits.max_layout_blocks)
    ):
        raise DocumentError("PDF extraction limits must be positive integers")
    return limits


def _validate_pdf_ocr_workload(pdf: Any, dpi: int) -> None:
    """Reject explicit local OCR requests with unbounded page/raster work.

    This estimates the pixels PyMuPDF must render from page geometry before
    invoking Tesseract. It is a conservative workload guard, not a guarantee
    of a particular runtime or memory ceiling.
    """
    page_count = pdf.page_count
    if page_count > MAX_OCR_PAGES:
        raise DocumentError("local OCR exceeds configured page-count safety limit")

    total_pixels = 0
    for index in range(page_count):
        try:
            rect = pdf[index].rect
            width_points = float(rect.width)
            height_points = float(rect.height)
        except (ValueError, RuntimeError, IndexError) as exc:
            raise DocumentError("local OCR page dimensions cannot be validated") from exc
        if (
            not math.isfinite(width_points)
            or not math.isfinite(height_points)
            or width_points <= 0
            or height_points <= 0
        ):
            raise DocumentError("local OCR page dimensions cannot be validated")

        # Each rendered dimension is at least one pixel. Reject dimensions that
        # would exceed the per-page budget before scaling them, since a huge but
        # finite PDF box can overflow to infinity during float multiplication.
        maximum_dimension_points = MAX_OCR_PAGE_PIXELS * 72 / dpi
        if width_points > maximum_dimension_points or height_points > maximum_dimension_points:
            raise DocumentError("local OCR page render exceeds configured pixel safety limit")
        width_pixels = math.ceil(width_points * dpi / 72)
        height_pixels = math.ceil(height_points * dpi / 72)
        page_pixels = width_pixels * height_pixels
        if page_pixels > MAX_OCR_PAGE_PIXELS:
            raise DocumentError("local OCR page render exceeds configured pixel safety limit")
        total_pixels += page_pixels
        if total_pixels > MAX_OCR_TOTAL_PIXELS:
            raise DocumentError("local OCR document exceeds configured pixel workload limit")


def _docx_declared_entry_count(raw: bytes, *, max_entries: int) -> int:
    """Count central-directory records before ZipFile materializes them.

    ``ZipFile`` materializes one ``ZipInfo`` per central-directory entry during
    construction. The end-of-directory count is attacker-controlled and can
    under-report the entries that ``ZipFile`` actually parses, so count the
    bounded central-directory records themselves first. ZIP64 is handled for
    ordinary single-disk packages; malformed or multi-disk records are rejected.
    """
    eocd_signature = b"PK\x05\x06"
    tail_size = min(len(raw), 22 + 65_535)
    tail_start = len(raw) - tail_size
    tail = raw[tail_start:]
    cursor = len(tail)
    while True:
        index = tail.rfind(eocd_signature, 0, cursor)
        if index < 0:
            raise DocumentError("DOCX package is invalid")
        if len(tail) - index >= 22:
            record = struct.unpack_from("<4s4H2LH", tail, index)
            comment_length = record[7]
            if index + 22 + comment_length == len(tail):
                break
        cursor = index

    _, disk_number, central_disk, disk_entries, total_entries, directory_size, _, _ = record
    if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
        raise DocumentError("multi-disk DOCX packages are not supported")

    absolute_eocd = tail_start + index
    directory_end = absolute_eocd
    if any(
        count == 0xFFFF for count in (disk_entries, total_entries)
    ) or directory_size == 0xFFFFFFFF or record[6] == 0xFFFFFFFF:
        locator_offset = absolute_eocd - 20
        if locator_offset < 0:
            raise DocumentError("DOCX ZIP64 directory is invalid")
        locator = struct.unpack_from("<4sIQI", raw, locator_offset)
        if locator[0] != b"PK\x06\x07" or locator[1] != 0 or locator[3] != 1:
            raise DocumentError("DOCX ZIP64 directory is invalid")
        zip64_offset = locator[2]
        if zip64_offset + 56 > len(raw):
            raise DocumentError("DOCX ZIP64 directory is invalid")
        zip64 = struct.unpack_from("<4sQ2H2I4Q", raw, zip64_offset)
        if zip64[0] != b"PK\x06\x06" or zip64[4] != 0 or zip64[5] != 0:
            raise DocumentError("multi-disk DOCX packages are not supported")
        if zip64[6] != zip64[7]:
            raise DocumentError("DOCX ZIP64 directory is invalid")
        total_entries = zip64[7]
        directory_size = zip64[8]
        directory_end = zip64_offset

    directory_start = directory_end - directory_size
    if directory_start < 0:
        raise DocumentError("DOCX central directory is invalid")
    cursor = directory_start
    directory_limit = directory_end
    actual_entries = 0
    while cursor < directory_limit:
        if actual_entries >= max_entries:
            raise DocumentError("DOCX package exceeds the archive entry count limit")
        if cursor + 46 > directory_limit:
            raise DocumentError("DOCX central directory is invalid")
        central_record = struct.unpack_from("<4s6H3I5H2I", raw, cursor)
        if central_record[0] != b"PK\x01\x02":
            raise DocumentError("DOCX central directory is invalid")
        filename_length, extra_length, comment_length = central_record[10:13]
        record_length = 46 + filename_length + extra_length + comment_length
        if cursor + record_length > directory_limit:
            raise DocumentError("DOCX central directory is invalid")
        cursor += record_length
        actual_entries += 1

    if actual_entries != total_entries:
        raise DocumentError("DOCX central directory entry count is inconsistent")
    return actual_entries


def _extract_docx_text(
    raw: bytes,
    *,
    max_xml_bytes: int = DEFAULT_DOCX_XML_MAX_BYTES,
    max_xml_total_bytes: int = DEFAULT_DOCX_XML_TOTAL_MAX_BYTES,
    max_expanded_bytes: int = DEFAULT_DOCX_EXPANDED_MAX_BYTES,
    max_entries: int = DEFAULT_DOCX_ARCHIVE_MAX_ENTRIES,
) -> str:
    """Extract paragraph text from a DOCX package without executing document content."""
    limits = (max_xml_bytes, max_xml_total_bytes, max_expanded_bytes, max_entries)
    if any(isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 for limit in limits):
        raise DocumentError("DOCX archive limits must be positive integers")
    _docx_declared_entry_count(raw, max_entries=max_entries)
    try:
        with ZipFile(BytesIO(raw)) as package:
            entries = package.infolist()
            entry_names: set[str] = set()
            for info in entries:
                if info.filename in entry_names:
                    raise DocumentError("DOCX package contains duplicate archive entries")
                entry_names.add(info.filename)
            expanded_bytes = sum(info.file_size for info in entries)
            if expanded_bytes > max_expanded_bytes:
                raise DocumentError("DOCX package exceeds the expanded size limit")
            xml_entries = [
                info
                for info in entries
                if info.filename.casefold().endswith((".xml", ".rels"))
            ]
            xml_bytes = sum(info.file_size for info in xml_entries)
            if xml_bytes > max_xml_total_bytes or any(
                info.file_size > max_xml_bytes for info in xml_entries
            ):
                raise DocumentError("DOCX XML parts exceed the expanded size limit")
            try:
                document_info = package.getinfo("word/document.xml")
            except KeyError as exc:
                raise DocumentError("DOCX package lacks word/document.xml") from exc
            if document_info.flag_bits & 0x1:
                raise DocumentError("encrypted DOCX packages are not supported")
            if document_info.file_size > max_xml_bytes:
                raise DocumentError("DOCX document XML exceeds the expanded size limit")
            with package.open(document_info) as document_stream:
                document_xml = document_stream.read(max_xml_bytes + 1)
            if len(document_xml) > max_xml_bytes:
                raise DocumentError("DOCX document XML exceeds the expanded size limit")
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("DOCX package is invalid") from exc
    # DOCX needs no DTD. Reject it before the hardened parser sees it, retaining a
    # clear document error while defusedxml remains the defence in depth boundary.
    if b"<!DOCTYPE" in document_xml.upper():
        raise DocumentError("DOCX document XML may not contain a DTD")
    try:
        root = ElementTree.fromstring(document_xml)
    except (DefusedXmlException, ParseError) as exc:
        raise DocumentError("DOCX document XML cannot be parsed") from exc
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_WORD_NAMESPACE}p"):
        pieces: list[str] = []
        # Text boxes can contain their own Word paragraphs below a normal
        # paragraph's drawing/run nodes. Keep those as separate paragraphs in
        # the outer root traversal instead of folding their text into the
        # parent and then emitting it a second time.
        pending = [iter(paragraph)]
        while pending:
            try:
                node = next(pending[-1])
            except StopIteration:
                pending.pop()
                continue
            if node.tag == f"{_WORD_NAMESPACE}p":
                continue
            if node.tag == f"{_WORD_NAMESPACE}t" and node.text:
                pieces.append(node.text)
            elif node.tag == f"{_WORD_NAMESPACE}tab":
                pieces.append("\t")
            elif node.tag in {f"{_WORD_NAMESPACE}br", f"{_WORD_NAMESPACE}cr"}:
                pieces.append("\n")
            else:
                pending.append(iter(node))
        text = "".join(pieces)
        if text:
            paragraphs.append(text)
    canonical = "\n".join(paragraphs)
    if not canonical:
        raise DocumentError("DOCX document contains no extractable text")
    return canonical


def import_docx(
    project, path: str | Path, *, max_bytes: int | None = None
) -> ImportResult:
    """Import a local DOCX as deterministic paragraph text; macros are never executed."""
    _reject_import_inside_caller_transaction(project)
    max_bytes = project.config.import_max_bytes if max_bytes is None else max_bytes
    source_path, raw = _read_import_source(path, max_bytes)
    if not raw.startswith(b"PK"):
        raise DocumentError("DOCX signature is invalid")
    canonical = _extract_docx_text(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    reserved = _reserve_source_hash_import(project)
    try:
        existing = _find_duplicate_source_revision(
            project,
            source_hash=source_hash,
            canonical_text=canonical,
            mime_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            extractor="docx-wordml@1",
        )
        if existing:
            return ImportResult(existing["document_id"], existing["id"], "DUPLICATE")
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        existing_document_id = _find_document_for_source(project, source_hash, mime_type)
        return _import_docx_bytes(
            project,
            source_path.name,
            raw,
            canonical_text=canonical,
            document_id=existing_document_id,
        )
    finally:
        _release_source_hash_import(project, reserved)


def _import_docx_bytes(
    project,
    filename: str,
    raw: bytes,
    *,
    canonical_text: str | None = None,
    document_id: str | None = None,
) -> ImportResult:
    if not raw.startswith(b"PK"):
        raise DocumentError("DOCX signature is invalid")
    canonical = canonical_text if canonical_text is not None else _extract_docx_text(raw)
    document_id, revision_id = project.import_text(
        filename,
        canonical,
        raw,
        extractor="docx-wordml@1",
        document_id=document_id,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    return ImportResult(document_id, revision_id, "SUCCESS")


def import_document(
    project,
    path: str | Path,
    *,
    max_bytes: int | None = None,
    ocr_language: str | None = None,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    auto_ocr: bool = False,
    pdf_extraction_limits: PdfExtractionLimits | None = None,
) -> ImportResult:
    """Import a supported local source through the single Core dispatch point."""
    _reject_import_inside_caller_transaction(project)
    _validate_import_ocr_options(ocr_language, ocr_dpi)
    max_bytes = project.config.import_max_bytes if max_bytes is None else max_bytes
    path, raw = _read_import_source(path, max_bytes)
    return import_document_bytes(
        project,
        path.name,
        raw,
        max_bytes=max_bytes,
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        auto_ocr=auto_ocr,
        pdf_extraction_limits=pdf_extraction_limits,
    )


def import_document_bytes(
    project,
    filename: str,
    raw: bytes,
    *,
    max_bytes: int | None = None,
    ocr_language: str | None = None,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    auto_ocr: bool = False,
    pdf_extraction_limits: PdfExtractionLimits | None = None,
) -> ImportResult:
    """Import already-read bytes without reopening a caller-validated pathname.

    This is used by adapters that must bind filesystem authorization to the
    opened file descriptor. ``filename`` is metadata/dispatch only; it is never
    interpreted as a filesystem path.
    """
    _reject_import_inside_caller_transaction(project)
    _validate_import_ocr_options(ocr_language, ocr_dpi)
    max_bytes = project.config.import_max_bytes if max_bytes is None else max_bytes
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
        or not isinstance(raw, bytes)
        or len(raw) > max_bytes
    ):
        raise DocumentError("document exceeds configured import size limit")
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise DocumentError("document filename must be a single safe name")

    log_event(
        _LOGGER,
        "document.import.started",
        level=logging.DEBUG,
        byte_count=len(raw),
    )
    suffix = Path(filename).suffix.casefold()
    if suffix == ".pdf" and not raw.startswith(b"%PDF"):
        raise DocumentError("file has a PDF extension but does not contain a PDF signature")
    if suffix in DOCX_SUFFIXES and not raw.startswith(b"PK"):
        raise DocumentError("file has a DOCX extension but does not contain a DOCX signature")
    if raw.startswith(b"%PDF"):
        result = _import_pdf_bytes(
            project,
            filename,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
            auto_ocr=auto_ocr,
            raw=raw,
            pdf_extraction_limits=pdf_extraction_limits,
        )
        return _log_import_result(result, len(raw))
    if pdf_extraction_limits is not None:
        raise DocumentError("PDF extraction limits are supported only for PDF documents")
    if ocr_language is not None:
        raise DocumentError("local OCR options are supported only for PDF documents")
    if suffix in DOCX_SUFFIXES:
        canonical = _extract_docx_text(raw)
        extractor = "docx-wordml@1"
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentError("only UTF-8 plain text is supported for non-PDF input") from exc
        if suffix in HTML_SUFFIXES:
            parser = _CanonicalHtmlParser()
            parser.feed(text)
            parser.close()
            if not parser.saw_markup:
                raise DocumentError("HTML extension requires HTML markup")
            canonical = parser.text()
            if not canonical:
                raise DocumentError("HTML document contains no extractable visible text")
            extractor = "html-visible-text@1"
            mime_type = "text/html"
        elif suffix in MARKDOWN_SUFFIXES:
            # Preserve Markdown source verbatim. A future semantic Markdown renderer
            # must create a new DocumentRevision rather than silently changing spans.
            canonical = text
            extractor = "markdown-source@1"
            mime_type = "text/markdown"
        else:
            canonical = text
            extractor = "plain-text@1"
            mime_type = "text/plain"

    source_hash = hashlib.sha256(raw).hexdigest()
    reserved = _reserve_source_hash_import(project)
    try:
        existing = _find_duplicate_source_revision(
            project,
            source_hash=source_hash,
            canonical_text=canonical,
            mime_type=mime_type,
            extractor=extractor,
        )
        if existing:
            return _log_import_result(
                ImportResult(existing["document_id"], existing["id"], "DUPLICATE"), len(raw)
            )
        existing_document_id = _find_document_for_source(project, source_hash, mime_type)
        if suffix in DOCX_SUFFIXES:
            return _log_import_result(
                _import_docx_bytes(
                    project,
                    filename,
                    raw,
                    canonical_text=canonical,
                    document_id=existing_document_id,
                ),
                len(raw),
            )
        document_id, revision_id = project.import_text(
            filename,
            canonical,
            raw,
            extractor=extractor,
            document_id=existing_document_id,
            mime_type=mime_type,
        )
        return _log_import_result(ImportResult(document_id, revision_id, "SUCCESS"), len(raw))
    finally:
        _release_source_hash_import(project, reserved)


def reprocess_pdf(
    project,
    document_id: str,
    path: str | Path,
    *,
    extractor_suffix: str = ".reprocessed",
    max_bytes: int | None = None,
    ocr_language: str | None = None,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    pdf_extraction_limits: PdfExtractionLimits | None = None,
) -> ImportResult:
    _reject_import_inside_caller_transaction(project)
    _validate_import_ocr_options(ocr_language, ocr_dpi)
    max_bytes = project.config.import_max_bytes if max_bytes is None else max_bytes
    source_path, raw = _read_import_source(path, max_bytes)
    return reprocess_pdf_bytes(
        project,
        document_id,
        source_path.name,
        raw,
        extractor_suffix=extractor_suffix,
        max_bytes=max_bytes,
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        pdf_extraction_limits=pdf_extraction_limits,
    )


def reprocess_pdf_bytes(
    project,
    document_id: str,
    filename: str,
    raw: bytes,
    *,
    extractor_suffix: str = ".reprocessed",
    max_bytes: int | None = None,
    ocr_language: str | None = None,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    pdf_extraction_limits: PdfExtractionLimits | None = None,
) -> ImportResult:
    """Create a new PDF revision from already-read bytes without reopening a path."""
    _reject_import_inside_caller_transaction(project)
    _validate_import_ocr_options(ocr_language, ocr_dpi)
    max_bytes = project.config.import_max_bytes if max_bytes is None else max_bytes
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
        or not isinstance(raw, bytes)
        or len(raw) > max_bytes
    ):
        raise DocumentError("document exceeds configured import size limit")
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise DocumentError("document filename must be a single safe name")
    if not project.db.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone():
        raise DocumentError("unknown document")

    return _import_pdf_bytes(
        project,
        filename,
        raw,
        document_id=document_id,
        extractor_suffix=extractor_suffix,
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        pdf_extraction_limits=pdf_extraction_limits,
    )


def _validate_ocr_options(language: str, dpi: int) -> None:
    if not isinstance(language, str) or not _OCR_LANGUAGE_PATTERN.fullmatch(language):
        raise DocumentError("OCR language must use Tesseract codes such as eng or eng+spa")
    if isinstance(dpi, bool) or not isinstance(dpi, int):
        raise DocumentError("OCR DPI must be an integer between 72 and 600")
    if not 72 <= dpi <= 600:
        raise DocumentError("OCR DPI must be between 72 and 600")


def _validate_import_ocr_options(language: str | None, dpi: int) -> None:
    if language is not None:
        _validate_ocr_options(language, dpi)
    else:
        if isinstance(dpi, bool) or not isinstance(dpi, int):
            raise DocumentError("OCR DPI must be an integer")
        if dpi != DEFAULT_OCR_DPI:
            raise DocumentError("OCR DPI requires an explicit OCR language")
