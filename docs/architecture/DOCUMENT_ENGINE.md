# Document Engine — Phase 2A

Digital PDFs are extracted locally with PyMuPDF. Originals are SHA-256 hashed and
stored through immutable document revisions; canonical text joins page text with
the frozen `\f\n` separator. Each revision stores an `extraction_manifest_json`
covering engine/version, extractor implementation, normalization, reading order,
paragraph segmentation, canonical assembly, coordinate scheme, bbox projection,
and OCR settings. Its `structure_hash` covers the path-free page geometry/text and
paragraph/block ranges and boxes. A changed canonical representation, structural
map, or pipeline manifest creates a new revision and never retargets historical
evidence. An exact repeat of source hash, canonical hash, media type, extractor,
manifest, and structural hash returns the existing revision as `DUPLICATE`.
Re-extracting the same PDF with a different extractor/OCR configuration creates
a revision under the existing document; repeating that exact configuration
deduplicates to its revision. The same bytes interpreted as different media types
(for example, plain text versus visible HTML) remain separate documents so their
canonical interpretations do not overwrite one another.

Project schema v49 adds these two revision identity fields and expands the
database uniqueness key. Existing revisions are retained; older revisions receive
a conservative recorded-fields-only manifest and canonical-hash structural
sentinel. Re-importing an older PDF under the now-explicit PDF pipeline therefore
creates a fully-provenanced revision rather than retroactively asserting unknown
layout provenance.
Digital PDF imports are capped at 1,000 pages before page-by-page extraction; larger
files fail without creating a partial revision. This parser-work limit is separate
from the raster-work limits applied when local OCR is explicitly requested.

Pages use zero-based internal indexes and one-based human labels. Each stable PDF
text block is a deterministic paragraph (`pymupdf-block-v1`) with canonical offsets,
page identity and a list of native PDF `(x0,y0,x1,y1)` boxes. Every paragraph must
equal its canonical text slice. Current pipeline versions are explicitly named
`discourselens-pdf-import-v1`, `pdf-text-preserve-v1`,
`pymupdf-sorted-blocks-v1`, `pymupdf-block-v1`, and
`page-ff-newline-v1`; a material algorithm change must bump the corresponding
version.

`SourceSpan` identity is a single contiguous canonical range. Projection intersects
that range with persisted page/paragraph ranges and produces one or more
`SourceSpanSegment`s; a cross-page span therefore renders as `pp. N–M`. For PDFs,
the engine aligns the selected paragraph text to PyMuPDF's transient character
geometry and stores only rectangles covering the selected glyphs, merged into
contiguous runs. It does not persist per-character geometry. Each segment records
`bbox_status` (`AVAILABLE`, `PARTIAL`, or `UNAVAILABLE`) and a projection version;
misaligned parser text yields no guessed box. Existing block-level segment boxes
are marked `APPROXIMATE` by migration and upgraded the next time they are resolved.

Image-only PDFs return `OCR_REQUIRED`; OCR is not invoked unless explicitly requested.
`--ocr-language` opts in to PyMuPDF's local Tesseract path (default 300 DPI,
bounded to 72–600 DPI); the local language code and DPI are revision provenance.
OCR failures stop import before a revision is created. OCR-derived revisions carry
`LOCAL_OCR_ENABLED` and `SUCCESS_WITH_WARNINGS`, and source-span geometry reruns the
same OCR settings; cloud OCR is never used. Tesseract and requested `tessdata`
language packs must be installed by the researcher. Invalid signatures are rejected
before parsing. Page/paragraph reading order uses PyMuPDF sorted blocks and
can be imperfect for complex multi-column or graphics-heavy layouts. Repeated
headers/footers and hyphenation are retained rather than destructively normalized.
Repeated, same-baseline text lines separated by a consistent wide horizontal gap
produce a persisted `AMBIGUOUS_READING_ORDER` warning and `SUCCESS_WITH_WARNINGS`;
the two detected baselines must be at least 5 pt apart, and the text is not silently
reordered. A single aligned pair can also trigger the warning when it has a gap of
at least 72 pt and is followed by more text in either horizontal lane. This catches
some short sidebars and unequal columns whose blocks can otherwise be interleaved.
This is a narrow layout diagnostic, not a reading-order engine. It can miss complex
layouts that do not expose these patterns, so a clean import status does not claim
full support for multi-column or graphics-heavy PDFs. Wrapped or indented
single-column text is not marked unless it has one of these wide-gap patterns;
aligned table-like rows may also be reported as ambiguous.
