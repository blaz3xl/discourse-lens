# SPIKE-006 — permissive digital-PDF backends

## Result

Neither candidate is a drop-in substitute for the Phase-2A PyMuPDF adapter.
`pdfplumber` is the strongest permissively licensed candidate for a follow-up
adapter because its search API returns matching character objects and a text
box. It can plausibly support the existing source-span projection contract,
but paragraph/block grouping, coordinate normalization, error translation and
canonical text rules would need explicit implementation and regression tests.
`pypdf` is much smaller and reads page text and rotation metadata, but its
visitor callback only supplied a text fragment and transform anchor for the
selected text in this fixture; it did not supply a selected-glyph box. It is
not a suitable sole parser for the current geometry contract without building
and validating a custom glyph-layout layer.

The Phase-2A document contract says an extractor-identity or canonical-text
change creates a new immutable revision. The candidates preserved page-level
extraction, offsets within their own output, page rotation information, and an
empty-text signal here; they did **not** preserve PyMuPDF's canonical text or
offset values. Any adoption therefore needs versioned extractor identity and
new-revision behavior, rather than reusing old offsets.

## Harness and scope

`experiments/spike_006_pdf_backend/check.py` creates original PDFs at runtime
with fixed text and geometry: a page with two text lanes and multiple
paragraph-like gaps, a rotated page, a blank page, and an image-only page. It
also generates AES-256 encrypted, truncated and malformed inputs. No external
documents, OCR, network APIs or paid services are used. The extraction timing
is a small local fixture benchmark, not a universal speed or quality claim.

The isolated uv environment ran Python 3.12.8 with PyMuPDF 1.28.2 (matching the
project's installed baseline), pypdf 6.19.0 and pdfplumber 0.11.10. The lockfile
and ignored `.venv` live under the experiment directory; no project dependency
or production code was changed.

## Observations

| Behavior on generated fixture | PyMuPDF 1.28.2 | pdfplumber 0.11.10 | pypdf 6.19.0 |
| --- | --- | --- | --- |
| Text and reading order | Sorted output interleaved the same-baseline side note and preserved wide spaces and blank block separators. Page 1 text length 159; selected phrase offset 139. | Four visual lines; the same-baseline side note appeared on the first line. Page 1 length 126; phrase offset 106. | Preserved content-stream insertion order, placing the side note last. Page 1 length 126; phrase offset 89. |
| Block / paragraph material | Four explicit text blocks; 122 character records in raw text output. | 122 character and 19 word records; text extraction returned lines, with no directly equivalent paragraph-block object. | Five visitor text fragments; these were content fragments, not paragraph blocks. |
| Selected text to geometry | `search_for` returned one rectangle for `UNIQUE QUOTE`. Raw character boxes are available. | `search` returned one box plus matched character objects for the same phrase. | Visitor returned a containing text fragment and its transform anchor, but no selected-text rectangle. |
| Page size and rotation | Exposes media dimensions and effective rotated page rectangle; rotated page was 600×400 at 90°. | Exposes 600×400 effective page dimensions and 90° rotation. | Exposes 400×600 media dimensions and 90° rotation; an adapter must transform into effective page coordinates. |
| Blank / image-only signal | Empty extracted text on both pages. | Empty extracted text on both pages. | Empty extracted text on both pages. |
| Malformed / truncated input | Both rejected with `FileDataError`. | Both rejected with `PdfminerException`. | Both rejected with `PdfStreamError`. |
| Encrypted input without password | Opened and reported `needs_pass=True`. | Opening failed with `PdfminerException`. | Opened and reported `is_encrypted=True`, unauthenticated. |

These outputs demonstrate two distinct issues. First, the default text
serialization differs enough that a highlighted phrase had three different
canonical offsets: 139, 106 and 89. Second, all three exposed page dimensions
and rotation in some form, but only PyMuPDF and pdfplumber returned a direct
box for a selected phrase. Empty extracted text is a useful `OCR_REQUIRED`
signal, but cannot distinguish a scanned page from a genuinely blank one and
does not provide OCR.

## Runtime and package weight

| Parser profile | Locked dependency distributions | Installed files measured |
| --- | ---: | ---: |
| PyMuPDF | 1 | 58.844 MiB |
| pypdf | 1 | 1.643 MiB |
| pdfplumber | 8, including pdfminer.six, Pillow, pypdfium2 and their dependencies | 50.899 MiB |

These are installed Python distribution-file totals for each parser's locked
dependency closure in this environment. They exclude Python itself and system
libraries. The pdfplumber closure is much larger than the parser package alone;
the locked dependency tree is recorded by `uv tree`.

Across 12 timed runs, with one warm-up excluded and all four pages opened and
text-extracted per run, median times were 2.765 ms for PyMuPDF, 4.336 ms for
pypdf and 9.273 ms for pdfplumber (min/max: 2.435/3.573, 3.566/5.182 and
7.083/12.472 ms respectively). This tiny fixture only records local overhead.

## License declarations

The upstream [pypdf project metadata](https://github.com/py-pdf/pypdf/blob/main/pyproject.toml)
declares BSD-3-Clause. The upstream [pdfplumber license](https://github.com/jsvine/pdfplumber/blob/stable/LICENSE.txt)
is MIT. The PyMuPDF project states that its open-source use is under
[GNU AGPL v3, with a commercial license also available](https://github.com/pymupdf).
These statements cover the named projects, not a complete legal review of each
candidate's transitive dependencies.

Upstream API documentation supports the observed ergonomic differences:
[PyMuPDF text modes](https://pymupdf.readthedocs.io/en/latest/tutorial.html#extracting-text)
include blocks and character-level raw output;
[pdfplumber text extraction](https://github.com/jsvine/pdfplumber/blob/stable/README.md#extracting-text)
exposes character/word positions and search results; and
[pypdf visitor callbacks](https://pypdf.readthedocs.io/en/stable/user/extract-text.html#using-a-visitor)
provide text fragments and transformation matrices, with a documented caveat
that positions can be difficult to determine in complicated documents.

## Reproduction

Commands, run from the repository root:

```bash
uv lock --project experiments/spike_006_pdf_backend
uv sync --project experiments/spike_006_pdf_backend --locked
uv run --project experiments/spike_006_pdf_backend python experiments/spike_006_pdf_backend/check.py
```

The final harness command exited successfully and reported 14 passing checks:
all three parsers read all four pages and returned backend-local phrase
offsets; all reported 90° rotation and empty text for blank and image-only
pages; PyMuPDF and pdfplumber returned a selected-text box while pypdf did not;
and all three produced distinct canonical offsets. Malformed, truncated and
encrypted cases were captured in the output rather than treated as harness
failures.

This spike does not test a representative corpus, complex forms, Unicode font
edge cases, OCR, security hardening, or long-run memory behavior. It establishes
only that `pdfplumber` merits an adapter prototype if replacing the AGPL
dependency is an active goal; it does not establish equivalent extraction
quality across real-world PDFs.
