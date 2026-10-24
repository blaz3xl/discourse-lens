"""Bounded generated-PDF comparison for the Phase-2A extractor contract."""

from __future__ import annotations

import json
import logging
import statistics
import tempfile
import time
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf as fitz
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pypdf import PdfReader


PASSWORD = "fixture-password"
SEPARATOR = "\f\n"
TARGET = "UNIQUE QUOTE"
REPEATS = 12


def _save_fixtures(root: Path) -> dict[str, Path]:
    digital = root / "generated-digital.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((54, 72), "First paragraph line one.", fontsize=11)
    page.insert_text((54, 88), "First paragraph line two.", fontsize=11)
    page.insert_text((54, 128), "Second paragraph starts here.", fontsize=11)
    page.insert_text((54, 160), "PREFIX UNIQUE QUOTE SUFFIX.", fontsize=11)
    page.insert_text((330, 72), "Right lane note.", fontsize=11)

    rotated = doc.new_page(width=400, height=600)
    rotated.insert_text((42, 70), "Rotated page anchor.", fontsize=11)
    rotated.set_rotation(90)
    doc.new_page(width=612, height=792)  # A genuinely blank page.

    image_page = doc.new_page(width=612, height=792)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), 0)
    pixmap.clear_with(235)
    image_page.insert_image(image_page.rect, pixmap=pixmap)
    doc.set_metadata(
        {
            "title": "Generated PDF backend fixture",
            "author": "DiscourseLens local spike",
            "creationDate": "D:20260101000000Z",
            "modDate": "D:20260101000000Z",
        }
    )
    doc.save(digital, garbage=4, deflate=True)
    doc.close()

    encrypted = root / "generated-encrypted.pdf"
    protected = fitz.open()
    protected.new_page().insert_text((72, 72), "Encrypted fixture text.")
    protected.save(
        encrypted,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=PASSWORD,
        owner_pw="fixture-owner-password",
    )
    protected.close()

    truncated = root / "generated-truncated.pdf"
    truncated.write_bytes(digital.read_bytes()[:64])
    malformed = root / "generated-malformed.pdf"
    malformed.write_bytes(b"%PDF-1.7\nnot a complete PDF object graph\n")
    return {
        "digital": digital,
        "encrypted": encrypted,
        "truncated": truncated,
        "malformed": malformed,
    }


def _strings(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _canonical_offsets(page_texts: list[str]) -> dict[str, Any]:
    normalized_pages = [text.rstrip("\n") for text in page_texts]
    offsets: list[int] = []
    current = 0
    for text in normalized_pages:
        offsets.append(current)
        current += len(text) + len(SEPARATOR)
    joined = SEPARATOR.join(normalized_pages)
    return {
        "page_text_lengths": [len(text) for text in page_texts],
        "normalized_page_lengths": [len(text) for text in normalized_pages],
        "page_start_offsets": offsets,
        "separator": repr(SEPARATOR),
        "target_page_0_local_offset": normalized_pages[0].find(TARGET),
        "target_canonical_offset": joined.find(TARGET),
    }


def _extract_pymupdf(path: Path) -> dict[str, Any]:
    doc = fitz.open(path)
    try:
        page_texts = [page.get_text("text", sort=True) for page in doc]
        page = doc[0]
        blocks = [block for block in page.get_text("blocks", sort=True) if block[4].strip()]
        matches = page.search_for(TARGET)
        raw = page.get_text("rawdict", sort=True)
        raw_chars = sum(
            len(span.get("chars", []))
            for block in raw["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        page_geometry = [
            {
                "rect_width": round(item.rect.width, 2),
                "rect_height": round(item.rect.height, 2),
                "mediabox_width": round(item.mediabox.width, 2),
                "mediabox_height": round(item.mediabox.height, 2),
                "rotation": item.rotation,
            }
            for item in doc
        ]
        return {
            "version": fitz.VersionBind,
            "page_texts": page_texts,
            "page_0_lines": _strings(page_texts[0]),
            "offsets": _canonical_offsets(page_texts),
            "page_0_explicit_text_blocks": len(blocks),
            "page_0_raw_character_records": raw_chars,
            "bbox_lookup": {
                "selected_text": TARGET,
                "match_count": len(matches),
                "boxes": [list(map(float, match)) for match in matches],
            },
            "page_geometry": page_geometry,
        }
    finally:
        doc.close()


def _extract_pdfplumber(path: Path) -> dict[str, Any]:
    with pdfplumber.open(path) as pdf:
        page_texts = [page.extract_text() or "" for page in pdf.pages]
        page = pdf.pages[0]
        matches = page.search(TARGET, regex=False)
        return {
            "version": metadata.version("pdfplumber"),
            "page_texts": page_texts,
            "page_0_lines": _strings(page_texts[0]),
            "offsets": _canonical_offsets(page_texts),
            "page_0_character_records": len(page.chars),
            "page_0_word_records": len(page.extract_words()),
            "bbox_lookup": {
                "selected_text": TARGET,
                "match_count": len(matches),
                "boxes": [
                    [
                        round(float(match[key]), 2)
                        for key in ("x0", "top", "x1", "bottom")
                    ]
                    for match in matches
                ],
                "matched_character_count": sum(len(match.get("chars", [])) for match in matches),
            },
            "page_geometry": [
                {
                    "width": round(float(page.width), 2),
                    "height": round(float(page.height), 2),
                    "rotation": getattr(page, "rotation", None),
                }
                for page in pdf.pages
            ],
        }


def _extract_pypdf(path: Path) -> dict[str, Any]:
    reader = PdfReader(path, strict=False)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    visitor_fragments: list[dict[str, Any]] = []

    def visitor(text: str, cm: Any, tm: Any, font: Any, size: Any) -> None:
        if text.strip():
            visitor_fragments.append(
                {
                    "text": text,
                    "anchor": [round(float(tm[4]), 2), round(float(tm[5]), 2)],
                    "matrix": [round(float(value), 2) for value in cm],
                    "font_size": round(float(size), 2),
                }
            )

    reader.pages[0].extract_text(visitor_text=visitor)
    hits = [item for item in visitor_fragments if TARGET in item["text"]]
    return {
        "version": metadata.version("pypdf"),
        "page_texts": page_texts,
        "page_0_lines": _strings(page_texts[0]),
        "offsets": _canonical_offsets(page_texts),
        "page_0_visitor_text_fragments": len(visitor_fragments),
        "bbox_lookup": {
            "selected_text": TARGET,
            "text_fragments_containing_selection": len(hits),
            "fragment_anchors": [item["anchor"] for item in hits],
            "selection_box_available": False,
        },
        "page_geometry": [
            {
                "mediabox_width": round(float(page.mediabox.width), 2),
                "mediabox_height": round(float(page.mediabox.height), 2),
                "rotation": int(page.rotation or 0),
            }
            for page in reader.pages
        ],
    }


def _attempt(call: Any) -> dict[str, Any]:
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            value = call()
            return {
                "status": "OPENED",
                "result": value,
                "warnings": [str(item.message)[:160] for item in seen],
            }
        except Exception as error:  # Capture parser-specific failure modes.
            message_lines = str(error).splitlines()
            return {
                "status": "ERROR",
                "exception": type(error).__name__,
                "message": (message_lines[0] if message_lines else type(error).__name__)[:200],
                "warnings": [str(item.message)[:160] for item in seen],
            }
        finally:
            logging.disable(previous_disable)


def _edge_cases(paths: dict[str, Path]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for case in ("truncated", "malformed"):
        path = paths[case]
        results[case] = {
            "pymupdf": _attempt(lambda: len(fitz.open(path))),
            "pypdf": _attempt(lambda: len(PdfReader(path, strict=False).pages)),
            "pdfplumber": _attempt(lambda: _count_pdfplumber_pages(path)),
        }

    encrypted = paths["encrypted"]
    results["encrypted"] = {
        "pymupdf": _attempt(lambda: _pymupdf_encrypted(encrypted)),
        "pypdf": _attempt(lambda: _pypdf_encrypted(encrypted)),
        "pdfplumber": _attempt(lambda: _pdfplumber_encrypted(encrypted)),
    }
    return results


def _count_pdfplumber_pages(path: Path) -> int:
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def _pymupdf_encrypted(path: Path) -> dict[str, Any]:
    doc = fitz.open(path)
    result = {"needs_password": bool(doc.needs_pass)}
    doc.close()
    return result


def _pypdf_encrypted(path: Path) -> dict[str, Any]:
    reader = PdfReader(path, strict=False)
    return {"is_encrypted": bool(reader.is_encrypted), "authenticated_without_password": not reader.is_encrypted}


def _pdfplumber_encrypted(path: Path) -> dict[str, Any]:
    with pdfplumber.open(path) as pdf:
        return {"opened_without_password": True, "page_count": len(pdf.pages)}


def _bench(path: Path) -> dict[str, Any]:
    adapters = {
        "pymupdf": lambda: _read_all_pymupdf(path),
        "pdfplumber": lambda: _read_all_pdfplumber(path),
        "pypdf": lambda: _read_all_pypdf(path),
    }
    result: dict[str, Any] = {"repeats": REPEATS, "pages_per_repeat": 4}
    for name, adapter in adapters.items():
        adapter()  # One warm-up, excluded from measurement.
        samples = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            adapter()
            samples.append((time.perf_counter() - start) * 1000)
        result[name] = {
            "median_ms": round(statistics.median(samples), 3),
            "min_ms": round(min(samples), 3),
            "max_ms": round(max(samples), 3),
        }
    return result


def _read_all_pymupdf(path: Path) -> int:
    doc = fitz.open(path)
    count = sum(len(page.get_text("text", sort=True)) for page in doc)
    doc.close()
    return count


def _read_all_pdfplumber(path: Path) -> int:
    with pdfplumber.open(path) as pdf:
        return sum(len(page.extract_text() or "") for page in pdf.pages)


def _read_all_pypdf(path: Path) -> int:
    return sum(len(page.extract_text() or "") for page in PdfReader(path, strict=False).pages)


def _install_profiles() -> dict[str, Any]:
    installed = {
        canonicalize_name(dist.metadata.get("Name", "")): dist
        for dist in metadata.distributions()
    }

    def closure(root_name: str) -> set[str]:
        seen: set[str] = set()
        todo = [canonicalize_name(root_name)]
        while todo:
            name = todo.pop()
            if name in seen or name not in installed:
                continue
            seen.add(name)
            dist = installed[name]
            for requirement_text in dist.requires or []:
                requirement = Requirement(requirement_text)
                if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                    todo.append(canonicalize_name(requirement.name))
        return seen

    result = {}
    for label, root in (("pymupdf", "PyMuPDF"), ("pypdf", "pypdf"), ("pdfplumber", "pdfplumber")):
        names = closure(root)
        size = 0
        for name in names:
            dist = installed[name]
            for item in dist.files or ():
                try:
                    size += dist.locate_file(item).stat().st_size
                except OSError:
                    pass
        result[label] = {
            "dependency_distributions": sorted(names),
            "distribution_count": len(names),
            "installed_file_bytes": size,
            "installed_file_mib": round(size / (1024 * 1024), 3),
        }
    return result


def _checks(observations: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    for backend in ("pymupdf", "pypdf", "pdfplumber"):
        data = observations[backend]
        assert len(data["page_texts"]) == 4
        checks.append(f"{backend}: extracted all four generated pages")
        assert data["offsets"]["target_page_0_local_offset"] >= 0
        checks.append(f"{backend}: produced a page-local target offset")
        assert data["page_texts"][2].strip() == ""
        assert data["page_texts"][3].strip() == ""
        checks.append(f"{backend}: blank and image-only pages have no extractable text")
        assert data["page_geometry"][1]["rotation"] == 90
        checks.append(f"{backend}: reported the generated page rotation")
    assert observations["pymupdf"]["bbox_lookup"]["match_count"] > 0
    assert observations["pdfplumber"]["bbox_lookup"]["match_count"] > 0
    assert not observations["pypdf"]["bbox_lookup"]["selection_box_available"]
    checks.append("geometry lookup: PyMuPDF and pdfplumber return a selection box; pypdf does not")
    offsets = {
        data["offsets"]["target_canonical_offset"]
        for data in observations.values()
    }
    assert len(offsets) == 3
    checks.append("canonical serialization differs across all three backends on the same fixture")
    return checks


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pdf-backend-spike-") as temp:
        paths = _save_fixtures(Path(temp))
        observations = {
            "pymupdf": _extract_pymupdf(paths["digital"]),
            "pdfplumber": _extract_pdfplumber(paths["digital"]),
            "pypdf": _extract_pypdf(paths["digital"]),
        }
        no_text = {
            backend: {
                "blank_page_text_length": len(data["page_texts"][2]),
                "image_only_page_text_length": len(data["page_texts"][3]),
            }
            for backend, data in observations.items()
        }
        output = {
            "scope": "Generated fixture feasibility only; not a universal PDF quality benchmark.",
            "runtime": {
                "python": __import__("sys").version.split()[0],
                "pymupdf": observations["pymupdf"]["version"],
                "pypdf": observations["pypdf"]["version"],
                "pdfplumber": observations["pdfplumber"]["version"],
            },
            "fixture": {
                "page_count": 4,
                "pages": ["text with side-lane", "rotated text", "blank", "image-only"],
                "generated_malformed_and_truncated_inputs": True,
                "encrypted_password_protected_with_aes256": True,
                "fixture_content_is_original": True,
            },
            "observations": observations,
            "no_text_signal": no_text,
            "edge_case_open_behavior": _edge_cases(paths),
            "small_text_extraction_benchmark": _bench(paths["digital"]),
            "installed_runtime_profiles": _install_profiles(),
        }
        output["checks"] = _checks(observations)
        print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
