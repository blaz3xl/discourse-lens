from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event
from typing import Any, cast

import pytest

from discourselens import Project
from discourselens.documents import import_document
from discourselens.errors import DocumentError


class _SingleRowCursor:
    def __init__(self, row: Any) -> None:
        self.row = row

    def fetchone(self) -> Any:
        return self.row


class _ImportRaceObserver:
    def __init__(
        self, connection: Any, events: Queue[str], release_empty_lookup: Event
    ) -> None:
        self.connection = connection
        self.events = events
        self.release_empty_lookup = release_empty_lookup

    @property
    def in_transaction(self) -> bool:
        return self.connection.in_transaction

    def execute(self, statement: str, parameters: Any = ()) -> Any:
        normalized = " ".join(statement.split()).casefold()
        if normalized == "begin immediate":
            self.events.put("begin")
            return self.connection.execute(statement, parameters)
        if normalized.startswith(
            "select document_id,id from document_revisions where source_hash=?"
        ):
            row = self.connection.execute(statement, parameters).fetchone()
            if row is None:
                self.events.put("empty_lookup")
                if not self.release_empty_lookup.wait(timeout=5):
                    raise TimeoutError("test did not release the import race observer")
            return _SingleRowCursor(row)
        return self.connection.execute(statement, parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)


def test_html_import_preserves_original_and_extracts_visible_structural_text(tmp_path):
    source = tmp_path / "speech.html"
    source.write_text(
        "<html><head><title>Ignored</title><style>p{color:red}</style></head>"
        "<body><h1>Public speech</h1><p>Café migración.</p><p>Second paragraph.</p>"
        "<script>do_not_extract()</script></body></html>",
        encoding="utf-8",
    )
    project = Project.init(tmp_path / "p")
    result = import_document(project, source)
    assert result.status == "SUCCESS" and result.revision_id is not None
    revision = project.db.execute(
        "SELECT extractor,mime_type FROM document_revisions WHERE id=?", (result.revision_id,)
    ).fetchone()
    assert tuple(revision) == ("html-visible-text@1", "text/html")
    assert (
        project.revision_text(result.revision_id)
        == "Public speech\nCafé migración.\nSecond paragraph."
    )
    assert project.original_bytes(result.revision_id) == source.read_bytes()
    assert [
        row["text"]
        for row in project.db.execute(
            "SELECT text FROM paragraphs WHERE revision_id=? ORDER BY ordinal",
            (result.revision_id,),
        )
    ] == ["Public speech", "Café migración.", "Second paragraph."]


def test_markdown_is_preserved_verbatim_and_duplicate_bytes_do_not_create_a_revision(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\nCafé migración.\n", encoding="utf-8")
    project = Project.init(tmp_path / "p")
    first = import_document(project, source)
    duplicate = import_document(project, source)
    assert first.status == "SUCCESS" and first.revision_id is not None
    assert duplicate.status == "DUPLICATE" and duplicate.revision_id == first.revision_id
    revision = project.db.execute(
        "SELECT extractor,mime_type FROM document_revisions WHERE id=?", (first.revision_id,)
    ).fetchone()
    assert tuple(revision) == ("markdown-source@1", "text/markdown")
    assert project.revision_text(first.revision_id) == "# Notes\n\nCafé migración.\n"


@pytest.mark.parametrize("first_suffix", [".txt", ".html"])
def test_same_bytes_with_different_text_importers_keep_distinct_canonical_revisions(
    tmp_path, first_suffix
):
    raw = b"<p>Visible text</p><script>not visible in HTML</script>"
    first_path = tmp_path / f"first{first_suffix}"
    second_suffix = ".html" if first_suffix == ".txt" else ".txt"
    second_path = tmp_path / f"second{second_suffix}"
    first_path.write_bytes(raw)
    second_path.write_bytes(raw)
    project = Project.init(tmp_path / "project")

    first = import_document(project, first_path)
    second = import_document(project, second_path)

    assert first.status == second.status == "SUCCESS"
    assert first.revision_id is not None and second.revision_id is not None
    assert first.revision_id != second.revision_id
    canonical_by_suffix = {
        first_suffix: project.revision_text(first.revision_id),
        second_suffix: project.revision_text(second.revision_id),
    }
    assert canonical_by_suffix[".txt"] == raw.decode("utf-8")
    assert canonical_by_suffix[".html"] == "Visible text"
    assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


def test_duplicate_source_does_not_bypass_html_validation(tmp_path):
    raw = b"plain text without HTML markup"
    text_path = tmp_path / "notes.txt"
    html_path = tmp_path / "notes.html"
    text_path.write_bytes(raw)
    html_path.write_bytes(raw)
    project = Project.init(tmp_path / "project")

    assert import_document(project, text_path).status == "SUCCESS"
    with pytest.raises(DocumentError, match="HTML extension requires HTML markup"):
        import_document(project, html_path)
    assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_document_import_does_not_commit_or_leak_files_into_an_outer_transaction(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Imported source.", encoding="utf-8")
    project = Project.init(tmp_path / "p")
    project.db.execute("BEGIN")
    project.db.execute(
        "INSERT INTO project_meta(key,value) VALUES(?,?)",
        ("outer_transaction_probe", "pending"),
    )

    with pytest.raises(DocumentError, match="inside an existing transaction"):
        import_document(project, source)

    assert project.db.in_transaction
    assert list((project.hidden / "source").iterdir()) == []
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0
    project.db.rollback()
    assert not project.db.execute(
        "SELECT 1 FROM project_meta WHERE key='outer_transaction_probe'"
    ).fetchone()


def test_html_extension_requires_markup(tmp_path):
    source = tmp_path / "not-html.html"
    source.write_text("plain UTF-8 text", encoding="utf-8")
    project = Project.init(tmp_path / "p")
    with pytest.raises(DocumentError, match="requires HTML markup"):
        import_document(project, source)


def test_html_import_excludes_explicitly_hidden_content_and_keeps_following_text(tmp_path):
    source = tmp_path / "hidden.html"
    source.write_text(
        "<html><body><p>Visible before.</p>"
        "<p hidden>Hidden attribute text.</p>"
        '<div aria-hidden="true">ARIA-hidden text.</div>'
        '<section style="display: none !important">Display-hidden text.</section>'
        '<p style="visibility: hidden">Visibility-hidden text.</p>'
        "<div hidden><div>Nested hidden text.</div>Still hidden text.</div>"
        "<input hidden><p>Visible after.</p></body></html>",
        encoding="utf-8",
    )
    project = Project.init(tmp_path / "p")

    result = import_document(project, source)

    assert result.revision_id is not None
    assert project.revision_text(result.revision_id) == "Visible before.\nVisible after."


def test_html_self_closing_syntax_does_not_reveal_hidden_nonvoid_content(tmp_path):
    source = tmp_path / "self-closing-hidden.html"
    source.write_text(
        "<html><body><div hidden/>Hidden text.</div><p>Visible after.</p></body></html>",
        encoding="utf-8",
    )
    project = Project.init(tmp_path / "p")

    result = import_document(project, source)

    assert result.revision_id is not None
    assert project.revision_text(result.revision_id) == "Visible after."


def test_html_hidden_nesting_has_a_bounded_parser_stack(tmp_path):
    source = tmp_path / "deep-hidden.html"
    depth = 1_024
    source.write_text(
        "<div hidden>" + "<span>" * depth + "Hidden" + "</span>" * depth + "</div>",
        encoding="utf-8",
    )
    project = Project.init(tmp_path / "p")

    with pytest.raises(DocumentError, match="nesting exceeds the safety limit"):
        import_document(project, source)

    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_concurrent_identical_imports_are_deduplicated_across_project_connections(tmp_path):
    root = tmp_path / "project"
    initialized = Project.init(root)
    initialized.close()
    writer = Project.open(root)
    source = tmp_path / "notes.txt"
    raw = b"Same source bytes across both imports."
    source.write_bytes(raw)

    events: Queue[str] = Queue()
    release_empty_lookup = Event()
    importer_ready = Event()
    start_import = Event()
    def import_in_background() -> Any:
        importing = Project.open(root)
        cast(Any, importing).db = _ImportRaceObserver(importing.db, events, release_empty_lookup)
        importer_ready.set()
        start_import.wait(timeout=5)
        try:
            return import_document(importing, source)
        finally:
            importing.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(import_in_background)
        assert importer_ready.wait(timeout=5)
        # Hold SQLite's writer reservation so the test can deterministically place
        # a second identical import between the check and insert if the importer
        # performs an unlocked check-then-write.
        writer.db.execute("BEGIN IMMEDIATE")
        start_import.set()
        operation = events.get(timeout=5)
        try:
            writer.import_text(
                source.name,
                raw.decode("utf-8"),
                raw,
                extractor="plain-text@1",
                mime_type="text/plain",
            )
        finally:
            release_empty_lookup.set()
        result = future.result(timeout=5)

    assert operation == "begin"
    assert result.status == "DUPLICATE"
    assert writer.db.execute(
        "SELECT COUNT(*) FROM document_revisions WHERE source_hash=?",
        (hashlib.sha256(raw).hexdigest(),),
    ).fetchone()[0] == 1
    writer.close()
