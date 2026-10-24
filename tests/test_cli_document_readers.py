from __future__ import annotations

import json

import fitz
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.documents import import_pdf


def _write_two_page_pdf(path):
    document = fitz.open()
    first_page = document.new_page()
    first_page.insert_textbox(
        fitz.Rect(36, 36, 550, 750),
        "PAGE ZERO marker. " + "First page body. " * 20 + "TARGET ending.",
        fontsize=10,
    )
    second_page = document.new_page()
    second_page.insert_textbox(
        fitz.Rect(36, 36, 550, 750),
        "PAGE ONE marker. " + "Second page body. " * 20,
        fontsize=10,
    )
    document.save(path)


def test_document_readers_bound_persisted_page_paragraph_and_context_text(tmp_path):
    root = tmp_path / "reader-project"
    project = Project.init(root)
    source = tmp_path / "two-pages.pdf"
    _write_two_page_pdf(source)
    imported = import_pdf(project, source)
    assert imported.revision_id is not None
    revision_id = imported.revision_id

    first_page = project.page(revision_id, 0)
    second_page = project.page(revision_id, 1)
    paragraph = project.db.execute(
        "SELECT id,text FROM paragraphs WHERE revision_id=? ORDER BY ordinal LIMIT 1",
        (revision_id,),
    ).fetchone()
    assert paragraph is not None
    canonical_text = project.revision_text(revision_id)
    target_start = canonical_text.index("TARGET")
    span_id = project.create_span(revision_id, target_start, target_start + len("TARGET"))
    expected_context = project.span_context(span_id, before=5, after=4)
    project.close()

    runner = CliRunner()
    page_zero_result = runner.invoke(
        app,
        ["document", "page", str(root), revision_id, "0", "--max-chars", "12"],
    )
    page_one_result = runner.invoke(
        app,
        ["document", "page", str(root), revision_id, "1", "--max-chars", "12"],
    )
    paragraph_result = runner.invoke(
        app,
        ["document", "paragraph", str(root), str(paragraph["id"]), "--max-chars", "13"],
    )
    context_result = runner.invoke(
        app,
        [
            "document",
            "context",
            str(root),
            span_id,
            "--before",
            "5",
            "--after",
            "4",
            "--max-chars",
            "3",
        ],
    )

    for result in (page_zero_result, page_one_result, paragraph_result, context_result):
        assert result.exit_code == 0, result.output
    page_zero = json.loads(page_zero_result.output)
    page_one = json.loads(page_one_result.output)
    paragraph_data = json.loads(paragraph_result.output)
    context = json.loads(context_result.output)

    assert page_zero["page_index"] == 0
    assert page_zero["text"] == str(first_page["text"])[:12]
    assert page_zero["text_truncated"] is (len(str(first_page["text"])) > 12)
    assert page_one["page_index"] == 1
    assert page_one["text"] == str(second_page["text"])[:12]
    assert "PAGE ONE" in str(second_page["text"])
    assert page_one["text_truncated"] is (len(str(second_page["text"])) > 12)

    paragraph_text = str(paragraph["text"])
    assert paragraph_data["id"] == paragraph["id"]
    assert paragraph_data["text"] == paragraph_text[:13]
    assert paragraph_data["text_truncated"] is (len(paragraph_text) > 13)

    assert context["span_id"] == span_id
    assert context["before"] == expected_context["before"]
    assert context["exact_text"] == "TAR"
    assert context["exact_text_truncated"] is True
    assert context["after"] == expected_context["after"]


def test_document_readers_reject_unknown_ids_pages_and_context_bounds(tmp_path):
    root = tmp_path / "reader-errors"
    project = Project.init(root)
    source = tmp_path / "one-page.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "A persisted page for reader errors.")
    document.save(source)
    imported = import_pdf(project, source)
    assert imported.revision_id is not None
    revision_id = imported.revision_id
    project.close()

    runner = CliRunner()
    bad_page = runner.invoke(app, ["document", "page", str(root), revision_id, "1"])
    bad_revision = runner.invoke(app, ["document", "page", str(root), "unknown-revision", "0"])
    bad_paragraph = runner.invoke(app, ["document", "paragraph", str(root), "unknown-paragraph"])
    bad_span = runner.invoke(app, ["document", "context", str(root), "unknown-span"])
    bad_before = runner.invoke(
        app,
        ["document", "context", str(root), "unknown-span", "--before", "-1"],
    )
    bad_after = runner.invoke(
        app,
        ["document", "context", str(root), "unknown-span", "--after", "10001"],
    )

    assert bad_page.exit_code != 0
    assert "unknown page" in bad_page.output.lower()
    assert bad_revision.exit_code != 0
    assert "unknown page" in bad_revision.output.lower()
    assert bad_paragraph.exit_code != 0
    assert "unknown paragraph" in bad_paragraph.output.lower()
    assert bad_span.exit_code != 0
    assert "unknown span" in bad_span.output.lower()
    assert bad_before.exit_code == 2
    assert bad_after.exit_code == 2
