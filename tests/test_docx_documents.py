from __future__ import annotations

import struct
from io import BytesIO
from zipfile import ZipFile

import pytest

from discourselens import Project, documents
from discourselens.config import ProjectConfig
from discourselens.documents import _extract_docx_text, import_document, import_docx
from discourselens.errors import DocumentError


def _docx_bytes(paragraphs: list[str]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("word/document.xml", document)
        package.writestr("[Content_Types].xml", "<Types/>")
    return output.getvalue()


def test_docx_import_creates_stable_paragraphs_and_preserves_original(tmp_path):
    source = tmp_path / "speech.docx"
    source.write_bytes(_docx_bytes(["Café migración.", "Second paragraph."]))
    project = Project.init(tmp_path / "p")
    result = import_document(project, source)
    assert result.status == "SUCCESS" and result.revision_id is not None
    revision = project.db.execute(
        "SELECT extractor,mime_type FROM document_revisions WHERE id=?", (result.revision_id,)
    ).fetchone()
    assert revision["extractor"] == "docx-wordml@1"
    assert revision["mime_type"].endswith("wordprocessingml.document")
    assert project.revision_text(result.revision_id) == "Café migración.\nSecond paragraph."
    assert project.original_bytes(result.revision_id) == source.read_bytes()
    assert [
        row["text"]
        for row in project.db.execute(
            "SELECT text FROM paragraphs WHERE revision_id=? ORDER BY ordinal",
            (result.revision_id,),
        )
    ] == ["Café migración.", "Second paragraph."]


def test_direct_docx_import_deduplicates_identical_source_bytes(tmp_path):
    source = tmp_path / "speech.docx"
    source.write_bytes(_docx_bytes(["Repeated source."]))
    project = Project.init(tmp_path / "p")

    first = import_docx(project, source)
    duplicate = import_docx(project, source)

    assert first.status == "SUCCESS"
    assert duplicate.status == "DUPLICATE"
    assert duplicate.document_id == first.document_id
    assert duplicate.revision_id == first.revision_id
    assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 1


def test_docx_magic_content_is_validated_even_if_its_hash_was_imported_as_text(tmp_path):
    raw = b"PK but still plain UTF-8 text"
    text_source = tmp_path / "notes.txt"
    text_source.write_bytes(raw)
    docx_source = tmp_path / "notes.docx"
    docx_source.write_bytes(raw)
    project = Project.init(tmp_path / "p")

    first = import_document(project, text_source)
    assert first.status == "SUCCESS"

    with pytest.raises(DocumentError, match="DOCX package is invalid"):
        import_document(project, docx_source)

    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 1


def test_docx_text_box_paragraphs_are_not_duplicated_into_parent(tmp_path):
    source = tmp_path / "text-box.docx"
    document_xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
        "<w:body><w:p><w:r><w:t>Outer text</w:t><w:drawing><wp:inline>"
        "<a:graphic><a:graphicData><wps:wsp><wps:txbx><w:txbxContent>"
        "<w:p><w:r><w:t>Embedded text</w:t></w:r></w:p>"
        "</w:txbxContent></wps:txbx></wps:wsp></a:graphicData></a:graphic>"
        "</wp:inline></w:drawing></w:r></w:p></w:body></w:document>"
    )
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("word/document.xml", document_xml)
        package.writestr("[Content_Types].xml", "<Types/>")
    source.write_bytes(output.getvalue())
    project = Project.init(tmp_path / "p")

    result = import_document(project, source)

    assert result.revision_id is not None
    assert project.revision_text(result.revision_id) == "Outer text\nEmbedded text"
    assert [
        row["text"]
        for row in project.db.execute(
            "SELECT text FROM paragraphs WHERE revision_id=? ORDER BY ordinal",
            (result.revision_id,),
        )
    ] == ["Outer text", "Embedded text"]


def test_docx_rejects_duplicate_archive_parts_before_selecting_document_xml():
    document = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Conflicting content</w:t></w:r></w:p></w:body></w:document>"
    )
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("word/document.xml", document)
        with pytest.warns(UserWarning, match="Duplicate name"):
            package.writestr("word/document.xml", document.replace("Conflicting", "Other"))

    with pytest.raises(DocumentError, match="duplicate archive entries"):
        _extract_docx_text(output.getvalue())


def test_docx_requires_real_package_and_never_creates_a_revision_on_failure(tmp_path):
    source = tmp_path / "not.docx"
    source.write_text("plain UTF-8", encoding="utf-8")
    project = Project.init(tmp_path / "p")
    with pytest.raises(DocumentError, match="DOCX extension"):
        import_document(project, source)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_docx_rejects_document_xml_with_a_dtd(tmp_path):
    source = tmp_path / "unsafe.docx"
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr(
            "word/document.xml",
            '<!DOCTYPE document [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body></w:document>",
        )
    source.write_bytes(output.getvalue())
    project = Project.init(tmp_path / "p")
    with pytest.raises(DocumentError, match="DTD"):
        import_document(project, source)


def test_docx_expanded_xml_is_bounded_before_parsing():
    archive = _docx_bytes(["Repeated text " * 100])

    with pytest.raises(DocumentError, match="expanded size limit"):
        _extract_docx_text(archive, max_xml_bytes=128)


def test_docx_entry_count_is_rejected_before_zip_directory_materialization(monkeypatch):
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("word/document.xml", "<document/>")
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr("word/extra.xml", "<extra/>")

    def unexpected_zip_open(*args, **kwargs):
        raise AssertionError("ZipFile must not materialize an over-limit directory")

    monkeypatch.setattr("discourselens.documents.ZipFile", unexpected_zip_open)
    with pytest.raises(DocumentError, match="entry count limit"):
        _extract_docx_text(output.getvalue(), max_entries=2)


def test_docx_underreported_entry_count_is_rejected_before_zipfile_open(monkeypatch):
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("word/document.xml", "<document/>")
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr("word/extra.xml", "<extra/>")
    archive = bytearray(output.getvalue())
    eocd_offset = archive.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    # The Python ZIP reader trusts central-directory size while parsing entries,
    # even when these attacker-controlled entry-count fields claim there are none.
    struct.pack_into("<HH", archive, eocd_offset + 8, 0, 0)

    def unexpected_zip_open(*args, **kwargs):
        raise AssertionError("ZipFile must not materialize an over-limit directory")

    monkeypatch.setattr("discourselens.documents.ZipFile", unexpected_zip_open)
    with pytest.raises(DocumentError, match="entry count limit"):
        _extract_docx_text(bytes(archive), max_entries=2)


def test_docx_unsupported_compression_is_reported_as_document_error():
    archive = bytearray(_docx_bytes(["Unsupported compression fixture."]))
    local_header = archive.find(b"PK\x03\x04")
    central_header = archive.find(b"PK\x01\x02")
    assert local_header >= 0 and central_header >= 0
    struct.pack_into("<H", archive, local_header + 8, 99)
    struct.pack_into("<H", archive, central_header + 10, 99)

    with pytest.raises(DocumentError, match="package is invalid"):
        _extract_docx_text(bytes(archive))


def test_docx_declared_total_expansion_is_bounded():
    # Highly compressible XML exercises the archive's expanded-size policy without
    # allocating a large test fixture or asking an XML parser to consume it.
    archive = _docx_bytes(["Repeated text " * 1000])

    with pytest.raises(DocumentError, match="package exceeds the expanded size limit"):
        _extract_docx_text(
            archive,
            max_xml_bytes=100_000,
            max_xml_total_bytes=100_000,
            max_expanded_bytes=512,
        )


def test_docx_all_xml_parts_are_bounded_not_only_document_xml():
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("word/document.xml", "<document/>")
        package.writestr("[Content_Types].xml", "<Types/>" + (" " * 200))

    with pytest.raises(DocumentError, match="XML parts exceed the expanded size limit"):
        _extract_docx_text(
            output.getvalue(),
            max_xml_bytes=100,
            max_xml_total_bytes=500,
        )


def test_direct_docx_import_enforces_open_file_size_limit(tmp_path):
    source = tmp_path / "bounded.docx"
    source.write_bytes(_docx_bytes(["A bounded source."]))
    project = Project.init(tmp_path / "project")

    with pytest.raises(DocumentError, match="size limit"):
        import_docx(project, source, max_bytes=1)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_direct_docx_import_honors_project_import_size_limit(tmp_path):
    source = tmp_path / "bounded.docx"
    source.write_bytes(_docx_bytes(["The project policy must cover direct DOCX imports."]))
    project = Project.init(tmp_path / "project")
    project.set_project_config(ProjectConfig(import_max_bytes=source.stat().st_size - 1))

    with pytest.raises(DocumentError, match="size limit"):
        import_docx(project, source)

    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_docx_expanded_xml_limit_failure_does_not_mutate_project(tmp_path, monkeypatch):
    source = tmp_path / "expanded.docx"
    source.write_bytes(_docx_bytes(["Expanded content " * 20]))
    project = Project.init(tmp_path / "project")
    extract = documents._extract_docx_text

    def enforce_small_xml_limit(raw):
        return extract(raw, max_xml_bytes=128)

    monkeypatch.setattr(documents, "_extract_docx_text", enforce_small_xml_limit)

    with pytest.raises(DocumentError, match="expanded size limit"):
        import_document(project, source)

    for table in (
        "documents",
        "document_revisions",
        "paragraphs",
        "sentences",
        "extraction_warnings",
        "revision_fts",
    ):
        assert project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert list((project.hidden / "source").iterdir()) == []
    assert not project.db.in_transaction
