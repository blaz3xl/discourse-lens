import pymupdf as fitz

from discourselens import Project
from discourselens.documents import import_pdf


def _unequal_columns_pdf(path, *, sidebar_baseline):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((55, 72), "LEFT COLUMN FIRST LINE.", fontsize=10)
    page.insert_text((330, sidebar_baseline), "RIGHT SIDEBAR NOTE.", fontsize=10)
    page.insert_text((55, 96), "LEFT COLUMN SECOND LINE.", fontsize=10)
    page.insert_text((55, 120), "LEFT COLUMN THIRD LINE.", fontsize=10)
    doc.save(path)
    doc.close()


def test_unequal_column_sidebar_with_one_aligned_pair_warns_without_reordering(tmp_path):
    source = tmp_path / "unequal-columns.pdf"
    _unequal_columns_pdf(source, sidebar_baseline=72)
    project = Project.init(tmp_path / "project")

    result = import_pdf(project, source)

    assert result.status == "SUCCESS_WITH_WARNINGS"
    assert result.revision_id is not None
    assert result.warnings == ("AMBIGUOUS_READING_ORDER",)
    canonical = project.revision_text(result.revision_id)
    assert canonical == (
        "LEFT COLUMN FIRST LINE.\n"
        "RIGHT SIDEBAR NOTE.\n"
        "LEFT COLUMN SECOND LINE.\n"
        "LEFT COLUMN THIRD LINE."
    )
    assert [row["code"] for row in project.extraction_warnings(result.revision_id)] == [
        "AMBIGUOUS_READING_ORDER"
    ]

    sidebar_start = canonical.index("RIGHT SIDEBAR NOTE.")
    sidebar_span = project.create_span(
        result.revision_id,
        sidebar_start,
        sidebar_start + len("RIGHT SIDEBAR NOTE."),
    )
    assert project.span_text(sidebar_span) == "RIGHT SIDEBAR NOTE."
    evidence = project.textual_evidence(sidebar_span)
    assert "p. 1" in project.render_citation(evidence)


def test_staggered_sidebar_with_one_aligned_row_warns(tmp_path):
    source = tmp_path / "staggered-columns.pdf"
    _unequal_columns_pdf(source, sidebar_baseline=96)
    project = Project.init(tmp_path / "project")

    result = import_pdf(project, source)

    assert result.status == "SUCCESS_WITH_WARNINGS"
    assert result.revision_id is not None
    assert "AMBIGUOUS_READING_ORDER" in result.warnings


def test_single_column_paragraph_stays_unmarked(tmp_path):
    source = tmp_path / "single-column.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((55, 72), "A single-column paragraph begins here.", fontsize=10)
    page.insert_text((82, 96), "Its indented continuation remains below it.", fontsize=10)
    page.insert_text((55, 120), "The paragraph returns to its original margin.", fontsize=10)
    doc.save(source)
    doc.close()
    project = Project.init(tmp_path / "project")

    result = import_pdf(project, source)

    assert result.status == "SUCCESS"
    assert result.warnings == ()
